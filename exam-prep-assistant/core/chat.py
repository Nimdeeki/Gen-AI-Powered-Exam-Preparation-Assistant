"""RAG chatbot: retrieve the most relevant note chunks, then let the LLM answer from them."""

from collections.abc import Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.config import FETCH_K, MEMORY_TURNS, TOP_K

# Turns a follow-up like "explain it with an example" into a standalone question,
# so the retriever knows what "it" refers to.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as a standalone question using the chat "
               "history for context. Reply with the question only. If it is already "
               "standalone, repeat it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly study assistant for a university student preparing for exams.\n"
     "Answer using ONLY the notes excerpts below. Cite the excerpts you used like [1] or [2].\n"
     "If the notes do not contain the answer, say so clearly, then give a short general "
     "answer marked 'Not from your notes:'.\n"
     "For questions about the uploaded documents as a whole (e.g. 'what topics do my notes "
     "cover?'), answer from the document list below, covering every document, and mention "
     "that the Main topics tab gives a detailed breakdown.\n"
     "Write exam-ready answers: clear headings or bullet points, key terms in bold, and "
     "an example where it helps.\n\n"
     "Documents the student has uploaded:\n{catalog}\n\n"
     "Notes excerpts:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


class StudyChatbot:
    def __init__(self, llm: BaseChatModel, vectorstore: FAISS, catalog: str = ""):
        self.llm = llm
        self.catalog = catalog or "(not listed)"
        # MMR = Maximal Marginal Relevance: relevant chunks that are not near-duplicates
        self.retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    @staticmethod
    def window(history: list[BaseMessage]) -> list[BaseMessage]:
        """Keep only the last N exchanges (like ConversationBufferWindowMemory)."""
        return history[-2 * MEMORY_TURNS:]

    def retrieve(self, question: str, history: list[BaseMessage]) -> tuple[str, list[Document]]:
        history = self.window(history)
        standalone = question
        if history:
            standalone = (CONDENSE_PROMPT | self.llm | StrOutputParser()).invoke(
                {"history": history, "question": question}).strip() or question
        return standalone, self.retriever.invoke(standalone)

    def _inputs(self, question, history, docs):
        return {"question": question, "history": self.window(history),
                "context": format_context(docs), "catalog": self.catalog}

    def ask(self, question: str, history: list[BaseMessage]) -> dict:
        standalone, docs = self.retrieve(question, history)
        answer = self.answer_chain.invoke(self._inputs(question, history, docs))
        return {"answer": answer, "sources": docs, "standalone_question": standalone}

    def stream(self, question: str, history: list[BaseMessage],
               docs: list[Document]) -> Iterator[str]:
        """Stream the answer token by token (call `retrieve` first to get `docs`)."""
        return self.answer_chain.stream(self._inputs(question, history, docs))
