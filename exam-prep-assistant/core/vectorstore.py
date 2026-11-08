"""Embeddings + FAISS vector store (the "Indexes" component of LangChain)."""

from functools import lru_cache

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # normalize_embeddings=True -> dot product == cosine similarity
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into an (n, d) array of unit vectors."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.asarray(get_embeddings().embed_documents(texts), dtype=np.float32)
