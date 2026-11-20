EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 4
FETCH_K = 20
MEMORY_TURNS = 4

DIGEST_CHAR_BUDGET = 9000
PREDICT_CHAR_BUDGET = 11000

PROVIDERS = {
    "Groq (free API, fast)": {
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "vision_model": "qwen/qwen3.8-27b",
        "key_url": "https://console.groq.com/keys",
    },
    "Google Gemini (free tier)": {
        "key_env": "GOOGLE_API_KEY",
        "model": "gemini-2.5-flash",
        "vision_model": "gemini-2.5-flash",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "OpenAI": {
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "vision_model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "Hugging Face local (offline, no key)": {
        "key_env": None,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "vision_model": None,
        "key_url": None,
    },
}
