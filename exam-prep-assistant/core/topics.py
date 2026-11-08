"""Main-topic extraction, two ways.

1. Keyphrases (classic NLP, no API key):
   RAKE-style candidates (stop words and punctuation split the text into phrases)
   -> TF-IDF weight of each phrase across the document's chunks
   -> re-rank by embedding similarity to the whole document
   -> MMR, so the list is not ten versions of the same phrase.
2. AI topic summary: the LLM reads a digest of the document and returns structured
   topics through a Pydantic output parser.
"""

import re
from collections import Counter

import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from core.config import DIGEST_CHAR_BUDGET
from core.documents import make_digest
from core.llm import invoke_structured
from core.preprocess import lemma
from core.vectorstore import embed

# Words that are common in lecture notes but never a topic on their own:
# filler words, generic nouns and everyday verbs.
NOTE_STOP_WORDS = set("""
use used using uses example examples eg etc ie like also one two three four five first second
third new way ways make makes made get gets got need needs needed help helps helpful page slide
fig figure image lab question questions answer answers following given different various type
types based specific important able called known let step steps thing things good better best
many much lot lots solution link https http www com colab ipynb yes no just really very
create creates created creating take takes taking give gives giving write writes written find
finds check checks know knows come comes go goes change changes allow allows provide provides
provided solve solves solved remove removes removed generate generates generated extract
extracts show shows say says said see try tries want wants work works working start starts
started mean means understand understands learn learns learned add adds added include includes
including keep keeps run runs set sets put call calls perform performs performed apply applies
day days time times number numbers part parts kind kinds case cases point points problem
problems approach approaches method methods technique techniques idea ideas result results
output outputs input inputs information data list lists simple simply small large big long
high low great main basic basically common easy hard possible certain particular general
generally usually often always sometimes today now still actually able unable less little
""".split())
STOP_WORDS = ENGLISH_STOP_WORDS | NOTE_STOP_WORDS

# Anything that is not a letter, hyphen or space ends a phrase (punctuation, digits, bullets)
PHRASE_BREAK_RE = re.compile(r"[^A-Za-z\- ]+|\s-\s")


def candidate_phrases(text: str, surface: dict[str, Counter] | None = None) -> list[str]:
    """RAKE-style: runs of 1-3 content words with no stop word or punctuation inside.

    Each phrase is returned in lemmatized lower case ("Transformers" -> "transformer") so
    singular/plural forms count as one phrase; `surface` records how it was written.
    """
    phrases = []
    for segment in PHRASE_BREAK_RE.split(text):
        run: list[str] = []
        for word in segment.split() + [""]:
            w = word.strip("-")
            if w and w.lower() not in STOP_WORDS and len(w) > 2:
                run.append(w)
                continue
            if 1 <= len(run) <= 3:
                key = " ".join(lemma(x.lower()) for x in run)
                phrases.append(key)
                if surface is not None:
                    surface.setdefault(key, Counter())[" ".join(run)] += 1
            run = []
    return phrases


def _mmr(cand_vecs: np.ndarray, relevance: np.ndarray, top_n: int,
         diversity: float = 0.5) -> list[int]:
    """Maximal Marginal Relevance: pick relevant candidates that differ from ones already picked."""
    chosen = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(relevance)) if i != chosen[0]]
    while remaining and len(chosen) < top_n:
        sim_to_chosen = cand_vecs[remaining] @ cand_vecs[chosen].T
        scores = (1 - diversity) * relevance[remaining] - diversity * sim_to_chosen.max(axis=1)
        best = remaining[int(np.argmax(scores))]
        chosen.append(best)
        remaining.remove(best)
    return chosen


def extract_keyphrases(chunks: list[Document], top_n: int = 15) -> list[tuple[str, float]]:
    """Return [(phrase, score 0-1)] for one document, best first.

    Phrases keep their most common spelling in the notes (e.g. "RNN", "LoRA").
    """
    texts = [c.page_content for c in chunks if c.page_content.strip()]
    if not texts:
        return []
    surface: dict[str, Counter] = {}
    vectorizer = TfidfVectorizer(analyzer=lambda t: candidate_phrases(t, surface),
                                 sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:  # no usable phrases at all
        return []
    vocab = vectorizer.get_feature_names_out()

    # Sum over chunks = how strongly the phrase characterises this document;
    # multiply by document frequency so one-off phrases don't win.
    weight = np.asarray(tfidf.sum(axis=0)).ravel()
    doc_freq = np.asarray((tfidf > 0).sum(axis=0)).ravel()
    weight *= np.log1p(doc_freq) * np.array([1.0 + 0.3 * p.count(" ") for p in vocab])
    candidates_idx = np.argsort(-weight)[:80]
    candidates = [vocab[i] for i in candidates_idx]
    cand_weight = weight[candidates_idx] / weight[candidates_idx].max()

    chunk_vecs = embed(texts)
    doc_vec = chunk_vecs.mean(axis=0)
    doc_vec /= np.linalg.norm(doc_vec) + 1e-9
    cand_vecs = embed(candidates)
    semantic = cand_vecs @ doc_vec
    relevance = 0.5 * semantic + 0.5 * cand_weight

    picked = _mmr(cand_vecs, relevance, top_n=top_n)
    top = relevance[picked].max()
    return [(surface[candidates[i]].most_common(1)[0][0], float(relevance[i] / top))
            for i in picked]


class Topic(BaseModel):
    title: str = Field(description="Short topic name, 2-6 words, as it would appear in a syllabus")
    summary: str = Field(description="1-2 sentence explanation of what this topic covers")
    subtopics: list[str] = Field(description="3-6 key subtopics, terms or concepts under this topic")
    importance: str = Field(description="High, Medium or Low - how central this topic is to the document")


class DocumentTopics(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the whole document")
    main_topics: list[Topic] = Field(description="The 4-10 main topics, in the order they appear")


TOPIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher who turns lecture notes into a clear syllabus-style "
               "topic list for students revising for exams. Use only what is in the notes."),
    ("human", "Document: {name}\n"
              "Frequent key phrases (from TF-IDF): {keyphrases}\n\n"
              "Notes (sampled from every page):\n{digest}\n\n"
              "List the main topics of this document.\n{format_instructions}"),
])


def summarize_topics(llm: BaseChatModel, name: str, pages: list[Document],
                     keyphrases: list[tuple[str, float]]) -> DocumentTopics:
    return invoke_structured(llm, TOPIC_PROMPT, DocumentTopics, {
        "name": name,
        "keyphrases": ", ".join(p for p, _ in keyphrases) or "(none)",
        "digest": make_digest(pages, DIGEST_CHAR_BUDGET),
    })


def topics_to_markdown(name: str, topics: DocumentTopics) -> str:
    lines = [f"# Main topics: {name}", "", topics.overview, ""]
    for i, t in enumerate(topics.main_topics, 1):
        lines += [f"## {i}. {t.title}  ({t.importance})", t.summary, ""]
        lines += [f"- {s}" for s in t.subtopics] + [""]
    return "\n".join(lines)


def clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .:-")
