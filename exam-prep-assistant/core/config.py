"""Central settings for the Exam Prep Assistant."""

# Embedding model (runs locally, free, ~90 MB download on first run)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting (LangChain RecursiveCharacterTextSplitter)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4            # chunks sent to the LLM per question
FETCH_K = 20         # candidates considered by MMR before picking TOP_K
MEMORY_TURNS = 4     # ConversationBufferWindowMemory-style: last N question/answer pairs

# Free-tier LLM APIs have small tokens-per-minute limits, so we cap how much
# document text goes into one prompt (~4 chars per token).
DIGEST_CHAR_BUDGET = 9000
PREDICT_CHAR_BUDGET = 11000

# Supported LLM providers. "model" is only a default; any model name the
# provider offers can be typed into the sidebar.
PROVIDERS = {
    "Groq (free API, fast)": {
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "key_url": "https://console.groq.com/keys",
    },
    "Google Gemini (free tier)": {
        "key_env": "GOOGLE_API_KEY",
        "model": "gemini-2.5-flash",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "OpenAI": {
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "Hugging Face local (offline, no key)": {
        "key_env": None,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "key_url": None,
    },
}
