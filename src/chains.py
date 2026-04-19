from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever

SYSTEM_PROMPT = (
    "Tu es un assistant RAG fiable. "
    "Reponds en francais de maniere claire et concise. "
    "Utilise uniquement le contexte fourni. "
    "Si l'information n'est pas dans le contexte, dis-le explicitement."
)


def generate_rag_answer(
    question: str,
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    chat_history: list[dict[str, str]],
) -> tuple[str, list[str]]:
    docs = retriever.invoke(question)
    context_blocks = [doc.page_content for doc in docs]
    context_text = "\n\n".join(context_blocks)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in chat_history[-8:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(
        HumanMessage(
            content=(
                "Question utilisateur:\n"
                f"{question}\n\n"
                "Contexte documentaire:\n"
                f"{context_text}"
            )
        )
    )

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return answer, context_blocks
