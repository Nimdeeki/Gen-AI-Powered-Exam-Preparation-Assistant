"""Compare notes with previous-year questions and predict upcoming exam questions.

How it works
1. Topic inventory: the main topics of every notes PDF (AI topics if available,
   otherwise TF-IDF keyphrases).
2. Semantic matching: every PYQ and every notes chunk is embedded; each question is
   assigned to its closest topic (cosine similarity). Questions far from all notes are
   reported as "not covered by your notes".
3. Scoring: topics asked in more years, more recently, for more marks, and with more
   pages in the notes get a higher priority score. Topics never asked but heavily
   covered in the notes are flagged as "not asked yet" (surprise candidates).
4. Generation: the LLM sees this evidence plus notes excerpts and writes new
   questions in the same style as the real papers.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core.config import PREDICT_CHAR_BUDGET
from core.llm import invoke_structured
from core.preprocess import preprocess
from core.pyq import PYQuestion
from core.topics import DocumentTopics, clean_label
from core.vectorstore import embed

NOTES_MATCH_MIN = 0.35     # question <-> best notes chunk similarity, else "not in notes"
STRONG_MATCH = 0.55        # above this the question is covered even if a word is missing

# Words every exam question uses; they say nothing about the topic
EXAM_WORDS = set("""
explain describe discuss define write short note notes example examples diagram neat suitable
differentiate compare difference distinguish brief briefly detail detailed state list mention
justify answer question following give illustrate elaborate significance various type types
""".split())


@dataclass
class TopicItem:
    label: str
    source: str
    embed_text: str


@dataclass
class TopicStats:
    topic: TopicItem
    questions: list[PYQuestion] = field(default_factory=list)
    coverage: float = 0.0               # share of notes chunks about this topic
    excerpts: list[Document] = field(default_factory=list)
    score: float = 0.0
    priority: str = "Low"

    @property
    def years(self) -> list[int]:
        return sorted({q.year for q in self.questions if q.year})

    @property
    def total_marks(self) -> float:
        return sum(q.marks or 0 for q in self.questions)


@dataclass
class CoverageGap:
    """A past question your notes do not (fully) cover."""
    question: PYQuestion
    similarity: float           # best match with any notes chunk
    missing_terms: list[str]    # question keywords that never appear in the notes


@dataclass
class Analysis:
    topics: list[TopicStats]
    gaps: list[CoverageGap]
    years: list[int]
    type_counts: Counter
    typical_marks: list[float]
    n_questions: int

    @property
    def asked(self) -> list[TopicStats]:
        return [t for t in self.topics if t.questions]

    @property
    def not_asked(self) -> list[TopicStats]:
        return [t for t in self.topics if not t.questions]


def build_topic_inventory(keyphrases: dict[str, list[tuple[str, float]]],
                          ai_topics: dict[str, DocumentTopics]) -> list[TopicItem]:
    items: list[TopicItem] = []
    for source, phrases in keyphrases.items():
        if source in ai_topics:
            for t in ai_topics[source].main_topics:
                items.append(TopicItem(clean_label(t.title), source,
                                       f"{t.title}: {', '.join(t.subtopics)}. {t.summary}"))
        else:
            for phrase, _ in phrases[:12]:
                label = clean_label(phrase)
                items.append(TopicItem(label[:1].upper() + label[1:], source, phrase))
    if not items:
        return items
    # Merge near-duplicate topics coming from different PDFs
    vecs = embed([i.embed_text for i in items])
    keep: list[int] = []
    for idx in range(len(items)):
        if all(float(vecs[idx] @ vecs[k]) < 0.88 for k in keep):
            keep.append(idx)
    return [items[i] for i in keep]


def _index_matrix(vectorstore: FAISS) -> tuple[np.ndarray, list[Document]]:
    """Read back the stored chunk vectors so we don't have to embed the notes again."""
    n = vectorstore.index.ntotal
    mat = vectorstore.index.reconstruct_n(0, n)
    docs = [vectorstore.docstore.search(vectorstore.index_to_docstore_id[i]) for i in range(n)]
    return np.asarray(mat, dtype=np.float32), docs


def analyse(topics: list[TopicItem], questions: list[PYQuestion], vectorstore: FAISS) -> Analysis:
    chunk_vecs, chunks = _index_matrix(vectorstore)
    topic_vecs = embed([t.embed_text for t in topics])
    stats = [TopicStats(topic=t) for t in topics]

    # Notes coverage: assign each chunk to its nearest topic
    chunk_topic_sim = chunk_vecs @ topic_vecs.T                  # (chunks, topics)
    nearest = chunk_topic_sim.argmax(axis=1)
    counts = np.bincount(nearest, minlength=len(topics))
    for i, s in enumerate(stats):
        s.coverage = counts[i] / max(1, len(chunks))
        best_chunks = np.argsort(-chunk_topic_sim[:, i])[:2]
        s.excerpts = [chunks[j] for j in best_chunks]

    # Match each question to a topic. A question is a "gap" when it is far from every
    # notes chunk (embeddings), or when its keywords are missing from the notes
    # vocabulary (classic preprocessing: tokenize -> stop words -> lemmatize).
    gaps: list[CoverageGap] = []
    if questions:
        notes_vocab = set()
        for c in chunks:
            notes_vocab.update(preprocess(c.page_content))
        q_vecs = embed([q.text for q in questions])
        q_topic = q_vecs @ topic_vecs.T
        q_notes = (q_vecs @ chunk_vecs.T).max(axis=1)
        for qi, q in enumerate(questions):
            terms = [t for t in dict.fromkeys(preprocess(q.text)) if t not in EXAM_WORDS]
            missing = [t for t in terms if t not in notes_vocab]
            sim = float(q_notes[qi])
            weak = sim < NOTES_MATCH_MIN or (
                missing and sim < STRONG_MATCH and len(missing) >= max(1, len(terms) // 4))
            if weak:
                gaps.append(CoverageGap(q, sim, missing))
            else:
                stats[int(q_topic[qi].argmax())].questions.append(q)

    years = sorted({q.year for q in questions if q.year})
    _score(stats, years)
    stats.sort(key=lambda s: -s.score)
    marks = [q.marks for q in questions if q.marks]
    return Analysis(topics=stats, gaps=gaps, years=years,
                    type_counts=Counter(q.qtype for q in questions),
                    typical_marks=sorted(set(marks)), n_questions=len(questions))


def _score(stats: list[TopicStats], years: list[int]) -> None:
    max_cov = max((s.coverage for s in stats), default=0) or 1
    max_marks = max((s.total_marks for s in stats), default=0)
    max_count = max((len(s.questions) for s in stats), default=0) or 1
    latest = years[-1] if years else None
    for s in stats:
        cov = s.coverage / max_cov
        if not s.questions:
            s.score = 0.35 * cov                  # never asked: rank by how much the notes cover it
        else:
            freq = len(s.years) / len(years) if years else len(s.questions) / max_count
            recency = max((0.6 ** (latest - y) for y in s.years), default=0.5) if latest else 0.5
            marks = s.total_marks / max_marks if max_marks else freq
            s.score = 0.45 * freq + 0.2 * recency + 0.15 * marks + 0.2 * cov
        s.priority = "High" if s.score >= 0.6 else "Medium" if s.score >= 0.35 else "Low"


class PredictedQuestion(BaseModel):
    question: str = Field(description="The exam question, phrased like a real university paper")
    topic: str = Field(description="Topic it belongs to (use the topic names given)")
    marks: int | None = Field(default=None, description="Likely marks, following the paper's pattern")
    question_type: str = Field(description="e.g. Definition, Explain, Compare, Diagram, Numerical, Code")
    likelihood: str = Field(description="High, Medium or Low")
    reason: str = Field(description="One sentence of evidence: years asked, repeat pattern or notes coverage")


class PredictedPaper(BaseModel):
    exam_pattern_insights: list[str] = Field(description="3-5 short observations about how this exam is set")
    questions: list[PredictedQuestion]


class MoreQuestions(BaseModel):
    questions: list[PredictedQuestion]


PREDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an experienced university exam paper setter and a study coach. You analyse "
     "previous year papers against the course notes and predict what is likely to be asked "
     "next. Be specific and honest: base every prediction on the evidence given."),
    ("human",
     "{course_line}"
     "EVIDENCE\n{evidence}\n\n"
     "TASK\nPredict exactly {n} questions for the upcoming exam.\n"
     "- Follow the priority ranking above: the top-ranked topics get the most questions.\n"
     "- About 60% from High/Medium priority topics: ask them from a new angle, combine "
     "related subtopics, or change the question type (do not copy past questions word for "
     "word; if a topic is asked almost every year you may repeat the classic question and "
     "say so in the reason).\n"
     "- About 25% from topics NOT asked yet but well covered in the notes.\n"
     "- Match the style, question types and marks of the past papers. "
     "Every question must be answerable from the notes.\n"
     "- Order from most to least likely.\n"
     "- In each reason, cite only facts from the EVIDENCE (years asked, marks, coverage). "
     "Never invent exam history.\n\n"
     "{format_instructions}"),
])

# Models often stop short of the requested count; this asks for the rest.
TOP_UP_PROMPT = ChatPromptTemplate.from_messages([
    PREDICT_PROMPT.messages[0],
    ("human",
     "{course_line}"
     "EVIDENCE\n{evidence}\n\n"
     "These questions are already predicted:\n{existing}\n\n"
     "TASK\nWrite exactly {n} MORE questions that are different from the ones above. "
     "Prefer topics NOT asked yet but covered in the notes, then Medium priority topics. "
     "Match the style and marks of the past papers, and cite only facts from the EVIDENCE "
     "in each reason.\n\n"
     "{format_instructions}"),
])


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def build_evidence(a: Analysis, budget: int = PREDICT_CHAR_BUDGET) -> str:
    parts: list[str] = []
    if a.n_questions:
        types = ", ".join(f"{k} {v}" for k, v in a.type_counts.most_common())
        marks = ", ".join(f"{m:g}" for m in a.typical_marks) or "unknown"
        parts.append(f"Exam pattern: {a.n_questions} past questions from years "
                     f"{', '.join(map(str, a.years)) or 'unknown'}. Question types: {types}. "
                     f"Marks used: {marks}.")
    else:
        parts.append("No previous year papers were provided. Predict from the notes alone: "
                     "topics with the most coverage are the most important.")

    parts.append("\nTOPICS RANKED BY PRIORITY (score 0-1):")
    for i, s in enumerate(a.asked[:10], 1):
        lines = [f"{i}. {s.topic.label} [{s.priority}, score {s.score:.2f}] - asked "
                 f"{len(s.questions)}x in years {', '.join(map(str, s.years)) or '?'}, "
                 f"{s.total_marks:g} marks total, notes coverage {s.coverage:.0%}"]
        for q in s.questions[:4]:
            tag = ", ".join(x for x in [str(q.year) if q.year else "", f"{q.marks:g}M" if q.marks else ""] if x)
            lines.append(f"   past ({tag}): {_short(q.text, 160)}")
        if s.excerpts:
            e = s.excerpts[0]
            lines.append(f"   notes ({e.metadata['source']} p.{e.metadata['page']}): "
                         f"{_short(e.page_content, 350)}")
        parts.append("\n".join(lines))

    parts.append("\nNOT ASKED YET BUT COVERED IN NOTES:")
    for s in a.not_asked[:6]:
        e = s.excerpts[0] if s.excerpts else None
        note = f" - notes ({e.metadata['source']} p.{e.metadata['page']}): {_short(e.page_content, 250)}" if e else ""
        parts.append(f"- {s.topic.label} (coverage {s.coverage:.0%}){note}")

    if a.gaps:
        parts.append("\nPAST QUESTIONS NOT COVERED BY THE NOTES (do not predict these):")
        parts += [f"- {_short(g.question.text, 120)}" for g in a.gaps[:5]]

    text, used = [], 0
    for p in parts:
        if used + len(p) > budget:
            break
        text.append(p)
        used += len(p)
    return "\n".join(text)


def predict_questions(llm: BaseChatModel, analysis: Analysis, n: int = 10,
                      course: str = "") -> PredictedPaper:
    inputs = {"course_line": f"Course: {course}\n\n" if course else "",
              "evidence": build_evidence(analysis)}
    paper = invoke_structured(llm, PREDICT_PROMPT, PredictedPaper, {**inputs, "n": n})
    missing = n - len(paper.questions)
    if missing > 0:
        existing = "\n".join(f"- {q.question}" for q in paper.questions)
        try:
            more = invoke_structured(llm, TOP_UP_PROMPT, MoreQuestions,
                                     {**inputs, "n": missing, "existing": existing})
            paper.questions += more.questions[:missing]
        except Exception:
            pass  # keep the questions we already have
    paper.questions = paper.questions[:n]
    return paper


def paper_to_markdown(paper: PredictedPaper, analysis: Analysis) -> str:
    lines = ["# Predicted exam questions", "",
             "_Generated by comparing your notes with previous year papers. "
             "Use it to prioritise revision; it is not a guarantee._", "",
             "## Exam pattern insights"]
    lines += [f"- {i}" for i in paper.exam_pattern_insights] + ["", "## Questions"]
    for i, q in enumerate(paper.questions, 1):
        marks = f" [{q.marks} marks]" if q.marks else ""
        lines += [f"{i}. **{q.question}**{marks}",
                  f"   - Topic: {q.topic} | Type: {q.question_type} | Likelihood: {q.likelihood}",
                  f"   - Why: {q.reason}"]
    lines += ["", "## Topic priority table", "",
              "| Topic | Priority | Times asked | Years | Notes coverage |",
              "|---|---|---|---|---|"]
    for s in analysis.topics:
        lines.append(f"| {s.topic.label} | {s.priority} ({s.score:.2f}) | {len(s.questions)} | "
                     f"{', '.join(map(str, s.years)) or '-'} | {s.coverage:.0%} |")
    return "\n".join(lines)
