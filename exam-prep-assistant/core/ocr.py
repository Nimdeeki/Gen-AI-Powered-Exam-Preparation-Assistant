import base64
import hashlib
import io
import re

from langchain_core.messages import HumanMessage
from pypdf import PdfReader

from core.config import PROVIDERS
from core.llm import call_with_retry, get_llm

OCR_PROMPT = ("Transcribe all the text on this page exactly as printed, line by line. "
              "Keep question numbers, sub-parts and marks. Output only the text.")
MIN_TEXT_CHARS = 200
MAX_SIDE = 1600

_cache: dict[str, str] = {}


def scanned_pages(data: bytes, min_chars: int = MIN_TEXT_CHARS) -> list[tuple[int, bytes]]:
    reader = PdfReader(io.BytesIO(data))
    found = []
    for number, page in enumerate(reader.pages, start=1):
        if len((page.extract_text() or "").strip()) >= min_chars:
            continue
        try:
            images = list(page.images)
            if not images:
                continue
            img = max(images, key=lambda i: len(i.data)).image.convert("RGB")
        except Exception:
            continue
        if page.rotation:
            img = img.rotate(-page.rotation, expand=True)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        found.append((number, buf.getvalue()))
    return found


def vision_llm(provider: str, api_key: str | None):
    info = PROVIDERS[provider]
    model = info.get("vision_model")
    if not model or (info["key_env"] and not api_key):
        return None, None
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0, max_retries=0,
                        max_tokens=1500, reasoning_format="hidden"), model
    return get_llm(provider, model, api_key, temperature=0), model


def _as_text(content) -> str:
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def _read_one(llm, image: bytes, on_wait=None) -> str:
    message = [HumanMessage(content=[
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
    ])]
    return call_with_retry(lambda: _as_text(llm.invoke(message).content), on_wait)


def read_images(llm, model: str, images: list[bytes], on_page=None, on_wait=None) -> list[str | Exception]:
    results: list[str | Exception] = []
    for n, image in enumerate(images, start=1):
        key = f"{model}:{hashlib.sha1(image).hexdigest()}"
        if key not in _cache:
            if on_page:
                on_page(n, len(images))
            try:
                _cache[key] = _read_one(llm, image, on_wait)
            except Exception as exc:
                results.append(exc)
                continue
        results.append(_cache[key])
    return results
