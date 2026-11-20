"""One function that returns a LangChain chat model for any provider.

This is LangChain's main selling point from class: the rest of the app calls
`llm.invoke(...)` the same way whether the model is Groq, Gemini, OpenAI or a
local Hugging Face model.
"""

import json
import re
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


def get_llm(provider: str, model: str, api_key: str | None = None,
            temperature: float = 0.3) -> BaseChatModel:
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=temperature, max_retries=3)
    if provider.startswith("Google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key,
                                      temperature=temperature, max_retries=3)
    if provider.startswith("OpenAI"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_retries=3)
    if provider.startswith("Hugging Face"):
        return _local_hf_llm(model)
    raise ValueError(f"Unknown provider: {provider}")


@lru_cache(maxsize=1)
def _local_hf_llm(model_id: str) -> BaseChatModel:
    """Runs a small open model on your own laptop. Slow, but needs no API key."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    pipe = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 1024, "do_sample": False,
                         "return_full_text": False},
    )
    return ChatHuggingFace(llm=pipe)


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in ```json fences or prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def invoke_structured(llm: BaseChatModel, prompt: ChatPromptTemplate,
                      schema: type[BaseModel], inputs: dict) -> BaseModel:
    """prompt | llm | PydanticOutputParser, with one automatic repair attempt.

    Small or free models sometimes return slightly broken JSON; we send the
    error back once and ask the model to fix it before giving up.
    """
    parser = PydanticOutputParser(pydantic_object=schema)
    inputs = {**inputs, "format_instructions": parser.get_format_instructions()}
    raw = (prompt | llm).invoke(inputs).content
    try:
        return schema.model_validate(json.loads(_extract_json(raw)))
    except Exception as first_error:
        repair = ChatPromptTemplate.from_messages([
            ("system", "You fix malformed JSON. Reply with valid JSON only, no commentary."),
            ("human", "This output failed to parse.\nError: {error}\n\n"
                      "Required format:\n{format_instructions}\n\nOutput to fix:\n{raw}"),
        ])
        fixed = (repair | llm).invoke({"error": str(first_error)[:500], "raw": raw,
                                       "format_instructions": parser.get_format_instructions()}).content
        try:
            return schema.model_validate(json.loads(_extract_json(fixed)))
        except Exception as exc:
            raise ValueError(f"The model did not return valid structured output.\n\n{raw}") from exc
