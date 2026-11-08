"""Exam Prep Assistant - chat with your notes, get main topics, predict exam questions.

Run:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from core.chat import StudyChatbot
from core.config import PROVIDERS
from core.documents import group_by_source, load_pdf, load_pdfs, split_documents
from core.llm import get_llm
from core.predictor import analyse, build_topic_inventory, paper_to_markdown, predict_questions
from core.preprocess import pipeline_report
from core.pyq import PYQuestion, classify_question, extract_questions
from core.topics import extract_keyphrases, summarize_topics, topics_to_markdown
from core.vectorstore import build_index

load_dotenv()
st.set_page_config(page_title="Exam Prep Assistant", page_icon="📚", layout="wide")

DEFAULTS = {
    "pages": [], "chunks": [], "store": None, "keyphrases": {}, "ai_topics": {},
    "extra": {}, "pyqs": [], "messages": [], "analysis": None, "paper": None,
    "answers": {}, "warnings": [], "pyq_version": 0,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
S = st.session_state


# ---------------------------------------------------------------- helpers
def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in ("401", "invalid api key", "api_key", "authentication", "permission")):
        return "The API key was rejected. Check that you pasted the right key for this provider."
    if any(k in low for k in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")):
        return ("The free-tier rate limit was reached. Wait a minute and try again, or pick a "
                "smaller/faster model in the sidebar.")
    if "model" in low and any(k in low for k in ("not found", "decommissioned", "does not exist")):
        return "That model name is not available. Change the model name in the sidebar."
    return f"Something went wrong: {msg[:600]}"


def badge(text: str, color: str) -> str:
    safe = str(text).replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"


PRIORITY_COLOR = {"High": "red", "Medium": "orange", "Low": "gray"}


def active_llm():
    """Return the chat model chosen in the sidebar, or None if it is not usable yet."""
    info = PROVIDERS[S.provider]
    if info["key_env"] and not S.get("api_key"):
        return None
    try:
        return get_llm(S.provider, S.model, S.get("api_key"))
    except Exception as exc:  # e.g. missing package, bad model id
        st.error(friendly_error(exc))
        return None


def need_llm_message() -> None:
    info = PROVIDERS[S.provider]
    st.info(f"Add your **{S.provider}** API key in the sidebar to use this. "
            f"Free key: {info['key_url']}  \nNo key? Choose *Hugging Face local* (slower, offline).")


def catalog() -> str:
    by_src = group_by_source(S.pages)
    return "\n".join(
        f"- {src} ({len(pages)} pages). Key terms: "
        f"{', '.join(p for p, _ in S.keyphrases.get(src, [])[:8])}"
        for src, pages in by_src.items())


def chatbot(llm) -> StudyChatbot:
    return StudyChatbot(llm, S.store, catalog())


# ---------------------------------------------------------------- processing
def process_notes(files) -> None:
    with st.status("Reading your notes...", expanded=True) as status:
        st.write("1/4 Extracting text from PDFs (PyPDFLoader)")
        pages, warnings = load_pdfs([(f.name, f.getvalue()) for f in files])
        if not pages:
            status.update(label="No text found in these PDFs", state="error")
            S.warnings = warnings
            return
        st.write(f"2/4 Splitting {len(pages)} pages into chunks")
        chunks = split_documents(pages)
        st.write(f"3/4 Embedding {len(chunks)} chunks into a FAISS vector store")
        store = build_index(chunks)
        st.write("4/4 Finding key phrases (TF-IDF + embeddings)")
        keyphrases = {src: extract_keyphrases(cs) for src, cs in group_by_source(chunks).items()}
        S.pages, S.chunks, S.store, S.keyphrases, S.warnings = pages, chunks, store, keyphrases, warnings
        S.ai_topics = {k: v for k, v in S.ai_topics.items() if k in keyphrases}
        S.analysis, S.paper, S.answers = None, None, {}
        status.update(label=f"Ready: {len(keyphrases)} PDFs, {len(pages)} pages, {len(chunks)} chunks",
                      state="complete", expanded=False)


def process_pyqs(files) -> None:
    questions: list[PYQuestion] = []
    for f in files:
        try:
            text = "\n".join(p.page_content for p in load_pdf(f.name, f.getvalue()))
        except Exception as exc:
            st.warning(f"Could not read {f.name}: {exc}")
            continue
        found = extract_questions(f.name, text)
        if not found:
            st.warning(f"No questions recognised in {f.name}. You can type them in the table "
                       "on the Exam predictor tab.")
        questions += found
    S.pyqs = questions
    S.pyq_version += 1          # fresh table, so old edits don't apply to new papers
    S.analysis, S.paper, S.answers = None, None, {}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 Exam Prep Assistant")
    st.caption("NLP + GenAI with LangChain")

    st.subheader("1. Choose the AI model")
    st.selectbox("Provider", list(PROVIDERS), key="provider")
    info = PROVIDERS[S.provider]
    st.text_input("Model name", value=info["model"], key=f"model::{S.provider}")
    S.model = S[f"model::{S.provider}"]
    if info["key_env"]:
        # A key from .env stays on the server: it is never put into the page,
        # so nobody who opens the app in a browser can read it.
        env_key = os.getenv(info["key_env"], "")
        typed = st.text_input("API key", type="password", key=f"key::{S.provider}",
                              placeholder="Using the key from .env" if env_key else "Paste your key",
                              help=f"Get a free key at {info['key_url']} . You can also put "
                                   f"{info['key_env']}=... in a .env file.")
        S.api_key = typed or env_key
        if env_key and not typed:
            st.caption("✓ Using the key from your .env file")
    else:
        S.api_key = None
        st.caption("Runs on your laptop. First use downloads the model (~3 GB) and answers "
                   "are slow, but no key or internet is needed afterwards.")

    st.subheader("2. Upload your notes")
    notes_files = st.file_uploader("Notes / slides (PDF)", type="pdf", accept_multiple_files=True)
    if st.button("Process notes", type="primary", disabled=not notes_files, width="stretch"):
        process_notes(notes_files)

    st.subheader("3. Previous year papers (optional)")
    pyq_files = st.file_uploader("PYQ papers (PDF)", type="pdf", accept_multiple_files=True,
                                 help="Put the year in the file name (e.g. NLP_2023.pdf) "
                                      "if the paper itself does not show it.")
    if st.button("Read question papers", disabled=not pyq_files, width="stretch"):
        process_pyqs(pyq_files)

    if S.keyphrases:
        st.divider()
        st.markdown("**Loaded**")
        for src, pages in group_by_source(S.pages).items():
            st.caption(f"📄 {src} · {len(pages)} pages")
        if S.pyqs:
            years = sorted({q.year for q in S.pyqs if q.year})
            st.caption(f"📝 {len(S.pyqs)} past questions · years {', '.join(map(str, years)) or '?'}")
        if st.button("Clear everything", width="stretch"):
            for key, value in DEFAULTS.items():
                S[key] = type(value)() if value is not None else None
            st.rerun()

for w in S.warnings:
    st.warning(w)

# ---------------------------------------------------------------- main
st.title("Exam Prep Assistant")
if not S.keyphrases:
    st.markdown(
        "Upload your course PDFs and this assistant will:\n"
        "- **Chat** — answer questions from *your* notes, with page references\n"
        "- **Main topics** — list the key topics of any PDF\n"
        "- **Exam predictor** — compare your notes with previous year papers and predict "
        "likely questions\n\n"
        "👈 Start in the sidebar: pick a model, upload notes, click **Process notes**.")
    st.stop()

chat_tab, topics_tab, predict_tab = st.tabs(["💬 Chat with notes", "📌 Main topics", "🔮 Exam predictor"])

# ---------------------------------------------------------------- chat
def show_sources(sources) -> None:
    with st.expander("Sources from your notes"):
        for i, (src, page, snippet) in enumerate(sources, 1):
            st.markdown(f"**[{i}] {src}, page {page}**  \n{snippet}")


with chat_tab:
    if S.messages and st.button("Clear chat"):
        S.messages = []
        st.rerun()
    if not S.messages and "pending_prompt" not in S:
        st.caption("Try one of these:")
        cols = st.columns(3)
        starters = ["What are the main topics in my notes?",
                    "Explain self-attention in simple words with an example",
                    "Give me 5 short-answer questions on fine-tuning"]
        for col, text in zip(cols, starters):
            if col.button(text, width="stretch"):
                S.pending_prompt = text
                st.rerun()

    # Messages go in this box so the input box always stays underneath them
    conversation = st.container()
    with conversation:
        for m in S.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    show_sources(m["sources"])

    prompt = st.chat_input("Ask anything about your notes...") or S.pop("pending_prompt", None)
    if prompt:
        llm = active_llm()
        with conversation:
            with st.chat_message("user"):
                st.markdown(prompt)
            if llm is None:
                need_llm_message()
            else:
                history = [HumanMessage(m["content"]) if m["role"] == "user"
                           else AIMessage(m["content"]) for m in S.messages]
                bot = chatbot(llm)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching your notes..."):
                            _, docs = bot.retrieve(prompt, history)
                        answer = st.write_stream(bot.stream(prompt, history, docs))
                        sources = [(d.metadata["source"], d.metadata["page"],
                                    d.page_content[:300].replace("\n", " ") + "...") for d in docs]
                        show_sources(sources)
                        S.messages += [{"role": "user", "content": prompt},
                                       {"role": "assistant", "content": answer, "sources": sources}]
                    except Exception as exc:
                        st.error(friendly_error(exc))

# ---------------------------------------------------------------- topics
with topics_tab:
    left, right = st.columns([2, 1])
    with right:
        other = st.file_uploader("Or analyse any other PDF", type="pdf", key="other_pdf",
                                 help="Only for topics: it is not added to the chat.")
        if other and other.name not in S.extra and other.name not in S.keyphrases:
            with st.spinner(f"Reading {other.name}..."):
                pages = load_pdf(other.name, other.getvalue())
                chunks = split_documents(pages)
                S.extra[other.name] = {"pages": pages, "keyphrases": extract_keyphrases(chunks)}
    names = list(S.keyphrases) + list(S.extra)
    with left:
        name = st.selectbox("Document", names,
                            index=names.index(other.name) if other and other.name in names else 0)

    if name in S.extra:
        pages, keyphrases = S.extra[name]["pages"], S.extra[name]["keyphrases"]
    else:
        pages, keyphrases = group_by_source(S.pages)[name], S.keyphrases[name]

    st.markdown(f"#### {name}")
    st.caption(f"{len(pages)} pages")
    st.markdown("**Key phrases** (no AI needed: TF-IDF + sentence embeddings + MMR)")
    st.markdown(" ".join(badge(p, "blue") for p, _ in keyphrases) or "_none found_")

    st.divider()
    topics = S.ai_topics.get(name)
    if st.button("✨ Generate AI topic summary" if not topics else "🔄 Regenerate AI topic summary",
                 type="primary" if not topics else "secondary"):
        llm = active_llm()
        if llm is None:
            need_llm_message()
        else:
            try:
                with st.spinner("Reading the document and listing its main topics..."):
                    S.ai_topics[name] = summarize_topics(llm, name, pages, keyphrases)
                st.rerun()
            except Exception as exc:
                st.error(friendly_error(exc))

    if topics:
        st.markdown(f"**Overview.** {topics.overview}")
        for i, t in enumerate(topics.main_topics, 1):
            with st.container(border=True):
                level = t.importance.strip().title()
                st.markdown(f"**{i}. {t.title}** {badge(level, PRIORITY_COLOR.get(level, 'gray'))}")
                st.write(t.summary)
                if t.subtopics:
                    st.caption(" · ".join(t.subtopics))
        st.download_button("Download topics (.md)", topics_to_markdown(name, topics),
                           file_name=f"{os.path.splitext(name)[0]}_topics.md")

    with st.expander("🔬 NLP pipeline view (what preprocessing does to this document)"):
        report = pipeline_report("\n".join(p.page_content for p in pages))
        c = st.columns(4)
        c[0].metric("Tokens", f"{report['tokens']:,}")
        c[1].metric("Vocabulary", f"{report['vocabulary']:,}")
        c[2].metric("After stop-word removal", f"{report['after_stopwords']:,}")
        c[3].metric("Vocab after lemmatization", f"{report['vocab_after_lemmatization']:,}",
                    delta=report["vocab_after_lemmatization"] - report["vocabulary"])
        st.caption("Steps: lowercase → remove HTML tags → remove URLs → remove punctuation → "
                   "NLTK word_tokenize → remove stop words → Porter stemming / WordNet lemmatization")
        st.dataframe(pd.DataFrame(report["stem_vs_lemma_examples"],
                                  columns=["Word", "Porter stem", "WordNet lemma"]),
                     hide_index=True)
        st.markdown("**Key phrase scores**")
        st.dataframe(pd.DataFrame(keyphrases, columns=["Phrase", "Score"]), hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                             format="%.2f")})

# ---------------------------------------------------------------- predictor
with predict_tab:
    st.markdown("Compares your notes with previous year questions (PYQs): which topics "
                "repeat, which are due, which were never asked, and then predicts questions.")

    st.markdown(f"**Past questions** ({len(S.pyqs)})")
    if not S.pyqs:
        st.caption("No question papers uploaded, so predictions will use your notes only. "
                   "Upload PYQ PDFs in the sidebar, or type questions into the table.")
    table = pd.DataFrame([{"Year": q.year, "Marks": q.marks, "Question": q.text, "Source": q.source}
                          for q in S.pyqs], columns=["Year", "Marks", "Question", "Source"])
    table[["Year", "Marks"]] = table[["Year", "Marks"]].apply(pd.to_numeric, errors="coerce")
    edited = st.data_editor(
        table, num_rows="dynamic", hide_index=True, width="stretch",
        key=f"pyq_editor_{S.pyq_version}",
        column_config={
            "Year": st.column_config.NumberColumn(format="%d", width="small"),
            "Marks": st.column_config.NumberColumn(width="small"),
            "Question": st.column_config.TextColumn(width="large"),
            "Source": st.column_config.TextColumn(width="small"),
        })
    questions = [
        PYQuestion(text=str(r.Question).strip(), source=str(r.Source or "typed"),
                   year=int(r.Year) if pd.notna(r.Year) else None,
                   marks=float(r.Marks) if pd.notna(r.Marks) else None,
                   qtype=classify_question(str(r.Question)))
        for r in edited.itertuples() if isinstance(r.Question, str) and r.Question.strip()]

    c1, c2, c3 = st.columns([2, 1, 1])
    course = c1.text_input("Course name (optional)", placeholder="e.g. NLP and Generative AI")
    n_questions = c2.slider("Questions to predict", 5, 20, 10)
    use_ai_topics = c3.toggle("AI topics", value=not S.provider.startswith("Hugging Face"),
                              help="Use AI-generated topic lists for matching (better, one API "
                                   "call per notes PDF). Off = key phrases only.")

    if st.button("🔮 Analyse & predict", type="primary"):
        llm = active_llm()
        try:
            if use_ai_topics and llm is not None:
                by_src = group_by_source(S.pages)
                todo = [s for s in S.keyphrases if s not in S.ai_topics]
                bar = st.progress(0.0, text="Listing topics in your notes...")
                for i, src in enumerate(todo, 1):
                    try:
                        S.ai_topics[src] = summarize_topics(llm, src, by_src[src], S.keyphrases[src])
                    except Exception as exc:
                        st.warning(f"Using key phrases for {src}: {friendly_error(exc)}")
                    bar.progress(i / len(todo), text=f"Listed topics for {i}/{len(todo)} PDFs")
                bar.empty()
            ai = S.ai_topics if use_ai_topics else {}
            with st.spinner("Matching past questions to your notes..."):
                S.analysis = analyse(build_topic_inventory(S.keyphrases, ai), questions, S.store)
            S.paper, S.answers = None, {}
            if llm is None:
                need_llm_message()
            else:
                with st.spinner("Predicting questions..."):
                    S.paper = predict_questions(llm, S.analysis, n=n_questions, course=course)
        except Exception as exc:
            st.error(friendly_error(exc))

    a = S.analysis
    if a:
        st.divider()
        m = st.columns(4)
        m[0].metric("Past questions", a.n_questions)
        m[1].metric("Years", ", ".join(map(str, a.years)) or "—")
        m[2].metric("Topics asked", len(a.asked))
        m[3].metric("Not in your notes", len(a.gaps))

        paper = S.paper
        if paper:
            st.subheader("Predicted questions")
            st.caption("Ranked by likelihood using the evidence below. Use it to prioritise "
                       "revision — it is a prediction, not a leaked paper!")
            if paper.exam_pattern_insights:
                with st.container(border=True):
                    st.markdown("**Exam pattern insights**")
                    for line in paper.exam_pattern_insights:
                        st.markdown(f"- {line}")
            llm = active_llm()
            for i, q in enumerate(paper.questions):
                with st.container(border=True):
                    level = q.likelihood.strip().title()
                    tags = [badge(level, PRIORITY_COLOR.get(level, "gray")), badge(q.topic, "blue"),
                            badge(q.question_type, "violet")]
                    if q.marks:
                        tags.append(badge(f"{q.marks} marks", "green"))
                    st.markdown(f"**Q{i + 1}. {q.question}**")
                    st.markdown(" ".join(tags))
                    st.caption(f"Why: {q.reason}")
                    if i in S.answers:
                        with st.expander("Answer from your notes", expanded=True):
                            st.markdown(S.answers[i])
                    elif st.button("Answer from my notes", key=f"ans_{i}"):
                        if llm is None:
                            need_llm_message()
                            continue
                        try:
                            with st.spinner("Writing an answer from your notes..."):
                                S.answers[i] = chatbot(llm).ask(
                                    f"Write an exam answer for: {q.question}", [])["answer"]
                            st.rerun()
                        except Exception as exc:
                            st.error(friendly_error(exc))
            st.download_button("Download predictions (.md)", paper_to_markdown(paper, a),
                               file_name="predicted_questions.md")

        st.subheader("Topic priority")
        st.dataframe(
            pd.DataFrame([{
                "Topic": s.topic.label, "Priority": s.priority, "Score": round(s.score, 2),
                "Times asked": len(s.questions), "Years": ", ".join(map(str, s.years)) or "—",
                "Marks": s.total_marks or None, "Notes coverage": s.coverage,
                "From": s.topic.source} for s in a.topics]),
            hide_index=True, width="stretch",
            column_config={
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Notes coverage": st.column_config.NumberColumn(format="percent"),
            })
        with st.expander("How is the score calculated?"):
            st.markdown(
                "Every past question and every chunk of your notes is turned into a sentence "
                "embedding (all-MiniLM-L6-v2). Each question is matched to its closest topic by "
                "cosine similarity. Then for each topic:\n\n"
                "`score = 0.45 × (years asked / years available) + 0.20 × recency + "
                "0.15 × marks share + 0.20 × notes coverage`\n\n"
                "Topics never asked score `0.35 × notes coverage` — big topics in your notes "
                "that haven't come yet are good surprise candidates.")

        if a.not_asked:
            st.subheader("Not asked yet")
            st.caption("Covered in your notes but never in the papers you gave — possible surprises.")
            st.markdown(" ".join(badge(s.topic.label, "orange") for s in a.not_asked[:15]))

        if a.gaps:
            st.subheader("Past questions your notes don't cover")
            st.caption("Study these from the textbook or ask your teacher — the notes are missing them.")
            for g in a.gaps:
                missing = f" — notes never mention: *{', '.join(g.missing_terms)}*" if g.missing_terms else ""
                year = f"({g.question.year}) " if g.question.year else ""
                st.markdown(f"- {year}{g.question.text}{missing}")
