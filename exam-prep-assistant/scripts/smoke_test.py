"""Run the whole pipeline from the command line (no Streamlit) to check everything works.

    python scripts/smoke_test.py --notes "path/to/notes" --pyq samples
    python scripts/smoke_test.py --notes notes --pyq samples --provider "Groq (free API, fast)"

Without --provider it uses a fake LLM, which checks the wiring but not answer quality.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from core.chat import StudyChatbot  # noqa: E402
from core.config import PROVIDERS  # noqa: E402
from core.documents import group_by_source, load_pdf, load_pdfs, split_documents  # noqa: E402
from core.llm import get_llm  # noqa: E402
from core.predictor import analyse, build_evidence, build_topic_inventory, predict_questions  # noqa: E402
from core.pyq import extract_questions  # noqa: E402
from core.topics import extract_keyphrases, summarize_topics  # noqa: E402
from core.vectorstore import build_index  # noqa: E402


def read_dir(folder: str) -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(Path(folder).glob("*.pdf"))]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--pyq")
    ap.add_argument("--provider", choices=list(PROVIDERS))
    ap.add_argument("--model")
    ap.add_argument("--ai-topics", action="store_true", help="also run AI topic summaries")
    args = ap.parse_args()

    pages, warnings = load_pdfs(read_dir(args.notes))
    chunks = split_documents(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks. Warnings: {warnings or 'none'}")
    store = build_index(chunks)

    keyphrases = {src: extract_keyphrases(cs) for src, cs in group_by_source(chunks).items()}
    for src, kp in keyphrases.items():
        print(f"\n[{src}] " + ", ".join(p for p, _ in kp[:10]))

    if args.provider:
        info = PROVIDERS[args.provider]
        llm = get_llm(args.provider, args.model or info["model"],
                      os.getenv(info["key_env"]) if info["key_env"] else None)
    else:
        llm = FakeListChatModel(responses=["(fake answer)"] * 50)

    ai_topics = {}
    if args.provider and args.ai_topics:
        by_src = group_by_source(pages)
        for src in list(keyphrases)[:2]:
            ai_topics[src] = summarize_topics(llm, src, by_src[src], keyphrases[src])
            print(f"\nAI topics for {src}:")
            for t in ai_topics[src].main_topics:
                print(f"  - {t.title} ({t.importance}): {', '.join(t.subtopics)}")

    bot = StudyChatbot(llm, store, catalog=", ".join(keyphrases))
    res = bot.ask("What is self-attention in transformers?", [])
    print("\nChat sources:", [(d.metadata["source"], d.metadata["page"]) for d in res["sources"]])
    print("Chat answer:", res["answer"][:800])

    questions = []
    if args.pyq:
        for name, data in read_dir(args.pyq):
            text = "\n".join(p.page_content for p in load_pdf(name, data))
            questions += extract_questions(name, text)
        for q in questions:
            print(f"  {q.year} {q.marks} {q.qtype:24} {q.text[:90]}")
    print(f"\nExtracted {len(questions)} PYQs")

    analysis = analyse(build_topic_inventory(keyphrases, ai_topics), questions, store)
    print("\nTop topics:")
    for s in analysis.topics[:12]:
        print(f"  {s.score:.2f} {s.priority:6} asked={len(s.questions)} years={s.years} "
              f"cov={s.coverage:.0%}  {s.topic.label}  <- {s.topic.source}")
    for g in analysis.gaps:
        print(f"  GAP sim={g.similarity:.2f} missing={g.missing_terms}  {g.question.text[:70]}")
    print("\nEvidence chars:", len(build_evidence(analysis)))

    if args.provider:
        paper = predict_questions(llm, analysis, n=10)
        print(f"\nGot {len(paper.questions)} of 10 requested questions")
        print("\nInsights:", *paper.exam_pattern_insights, sep="\n  - ")
        for q in paper.questions:
            print(f"  [{q.likelihood}] ({q.marks}M, {q.question_type}) {q.question}\n      why: {q.reason}")


if __name__ == "__main__":
    main()
