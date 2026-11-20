import os
import re
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNK_OVERLAP, CHUNK_SIZE

def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n[•●▪◦\-–]\s*\n", "\n• ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(name: str, data: bytes) -> list[Document]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    docs = []
    for page in pages:
        docs.append(Document(
            page_content=clean_text(page.page_content),
            metadata={"source": name, "page": page.metadata.get("page", 0) + 1},
        ))
    return docs


def load_pdfs(files: list[tuple[str, bytes]]) -> tuple[list[Document], list[str]]:
    all_pages, warnings = [], []
    for name, data in files:
        try:
            pages = load_pdf(name, data)
        except Exception as exc:
            warnings.append(f"Could not read {name}: {exc}")
            continue
        empty = [p.metadata["page"] for p in pages if len(p.page_content) < 20]
        if pages and len(empty) == len(pages):
            warnings.append(f"{name} has no selectable text (scanned images?). "
                            "Run it through OCR first.")
        elif len(empty) > len(pages) / 2:
            warnings.append(f"{name}: {len(empty)} of {len(pages)} pages have almost no text "
                            "(images or diagrams only).")
        all_pages.extend(p for p in pages if p.page_content)
    return all_pages, warnings


def split_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def group_by_source(docs: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for d in docs:
        grouped.setdefault(d.metadata["source"], []).append(d)
    return grouped


def make_digest(pages: list[Document], budget: int) -> str:
    if not pages:
        return ""
    per_page = max(200, min(1500, budget // len(pages)))
    parts, used = [], 0
    for p in pages:
        piece = p.page_content[:per_page].strip()
        if not piece:
            continue
        block = f"[page {p.metadata['page']}]\n{piece}"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
