# Exam Prep Assistant

An NLP + Generative AI chatbot for exam preparation, built with **LangChain** and **Streamlit**.

Upload your course PDFs and it will:

1. **Chat with your notes.** Ask anything and get an answer taken from *your* PDFs, with page references (RAG: Retrieval-Augmented Generation).
2. **Find the main topics** of any PDF, using classic NLP key phrases (no API key needed) or an AI topic summary.
3. **Predict exam questions.** Upload previous year question papers (PYQs). It matches every past question to a topic in your notes, finds which topics repeat, which are due and which were never asked, then generates likely questions with a reason for each. It also lists past questions your notes don't cover at all.

---

## Quick start

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac / Linux

# 2. install
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

The app opens at http://localhost:8501. Only your own computer can open it. To demo from a phone on the same Wi-Fi, remove `address = "localhost"` from `.streamlit/config.toml`, and don't keep a key in `.env` while you do.

### Get a free API key

Pick one provider in the sidebar and paste its key, or put it in a `.env` file (copy `.env.example`). A key in `.env` stays on the server and is never sent to the browser. `.env` is in `.gitignore`, so it won't be pushed to GitHub. If a key ever leaks, delete it in the provider's console and make a new one.

| Provider | Free? | Get a key |
|---|---|---|
| **Groq** (recommended) | Yes, very fast | https://console.groq.com/keys |
| Google Gemini | Free tier | https://aistudio.google.com/apikey |
| OpenAI | Paid | https://platform.openai.com/api-keys |
| Hugging Face local | Free, offline | No key. Downloads a small model (~3 GB) and runs on your laptop. Slow, and answers are weaker. |

If Groq's free tier gives a rate-limit error, wait a minute or switch the model name to `openai/gpt-oss-20b` or `qwen/qwen3.8-27b`. Groq retires models over time; the current list is at https://console.groq.com/docs/models.

### Try it with the sample papers

`samples/` has three **made-up** question papers (2023, 2024, 2025) in three different layouts. They are not real exam papers; they exist only for testing and demos. Upload your notes plus these to see the predictor working, then swap in your real PYQs.

To test without the UI:

```bash
python scripts/smoke_test.py --notes "path/to/notes_folder" --pyq samples
python scripts/smoke_test.py --notes "path/to/notes_folder" --pyq samples --provider "Groq (free API, fast)"
```

---

## How it works

```
 PDFs ──► PyPDFLoader ──► clean text ──► RecursiveCharacterTextSplitter ──► chunks
                                                                              │
                              HuggingFace embeddings (all-MiniLM-L6-v2) ◄─────┘
                                              │
                                        FAISS vector store
                                              │
   ┌──────────────────────────────────────────┼──────────────────────────────────┐
   │ CHAT                                      │ TOPICS                           │ PREDICTOR
   │ question + history                        │ RAKE-style phrases               │ PYQ PDFs ──► regex question
   │   → condense prompt (memory)              │   → TF-IDF weight                │   extraction (Q1, (a), marks, year)
   │   → MMR retriever (top 4 chunks)          │   → embedding similarity         │   → embed questions
   │   → answer prompt | LLM | parser          │   → MMR for diversity            │   → match to note topics (cosine)
   │   → answer + [1][2] sources               │ + AI summary (Pydantic parser)   │   → score topics → LLM predicts
   └───────────────────────────────────────────┴──────────────────────────────────┘
```

### Where each class topic is used

| Topic from class | Where it is in the code |
|---|---|
| NLP pipeline: data acquisition from PDF | `core/documents.py` (PyPDFLoader) |
| Text preprocessing: lowercasing, HTML/URL/punctuation removal, tokenization, stop words, stemming, lemmatization | `core/preprocess.py` (same steps as the class notebook). Shown live in the app under *Main topics → NLP pipeline view* |
| Heuristic approach: regular expressions | `core/pyq.py` finds question numbers, sub-parts, marks and years |
| Text representation: BoW / n-grams / TF-IDF | `core/topics.py` (TfidfVectorizer over 1–3 word phrases) |
| Word embeddings → sentence embeddings | `core/vectorstore.py` (sentence-transformers, the modern successor to Word2Vec) |
| LangChain Models (chat + embedding) | `core/llm.py`: one interface for Groq, Gemini, OpenAI and Hugging Face |
| Prompt templates | `ChatPromptTemplate` in `chat.py`, `topics.py`, `predictor.py` |
| Chains (LCEL `prompt \| llm \| parser`) | everywhere, e.g. `StudyChatbot.answer_chain` |
| Indexes: loader → splitter → vector store → retriever | `documents.py` + `vectorstore.py` + `chat.py` |
| Memory (ConversationBufferWindowMemory idea) | `StudyChatbot.window()` keeps the last 4 exchanges |
| Output parsers: Pydantic | `DocumentTopics`, `PredictedPaper` + `invoke_structured()` with auto-repair |
| Transformers / attention | the embedding model and every LLM used are Transformers |

### How the predictor scores topics

For every topic in your notes:

```
score = 0.45 × (years it was asked / years of papers given)
      + 0.20 × recency          (1.0 if asked in the latest paper, ×0.6 per year older)
      + 0.15 × share of marks
      + 0.20 × notes coverage   (how much of your notes is about it)
```

Topics never asked get `0.35 × notes coverage`, so large topics that haven't come yet show up as surprise candidates. A past question counts as **not covered by your notes** when its embedding is far from every notes chunk, or when its keywords (after preprocessing) never appear in the notes.

The LLM then gets this evidence plus notes excerpts and writes new questions in the same style and marks as the real papers. **It is a revision-priority tool, not a guarantee.**

---

## Project structure

```
app.py                 Streamlit UI (3 tabs)
core/
  config.py            settings + provider list
  documents.py         PDF loading, cleaning, chunking
  preprocess.py        classic NLP preprocessing (NLTK)
  vectorstore.py       embeddings + FAISS
  llm.py               provider switch + structured output helper
  chat.py              RAG chatbot with memory
  topics.py            key phrases + AI topic summary
  pyq.py               question-paper parser
  predictor.py         notes ↔ PYQ analysis + prediction
scripts/smoke_test.py  run the whole pipeline from the terminal
samples/               made-up PYQ papers for testing
```

## Limitations and ideas to extend

- **Scanned PDFs** (photos of pages) have no text layer. The app warns you. Add OCR with `pytesseract` to support them.
- The index is rebuilt every session. Save it with `FAISS.save_local()` to skip re-processing.
- Question extraction is rule-based. Unusual paper layouts may need fixing in the editable table on the predictor tab.
- Free-tier APIs limit tokens per minute, so prompts are capped at about 9–11k characters (see `core/config.py`).
