import logging
import os

import openai

from .env import load_env
from .models import QueryResult, RetrievedChunk
from .prompts import SYSTEM_PROMPT, build_rag_prompt
from .retrieve import format_context, retrieve

load_env()

logger = logging.getLogger(__name__)

_openai_client = None


def get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def ask(
    question: str,
    top_k: int = 5,
    paper_filter: str | None = None,
) -> QueryResult:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    chunks = retrieve(question, top_k=top_k, paper_filter=paper_filter)
    context = format_context(chunks)
    user_prompt = build_rag_prompt(question, context)

    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content

    return QueryResult(
        question=question,
        answer=answer,
        sources=chunks,
        model_used=model,
    )


def stream_ask(
    question: str,
    top_k: int = 5,
    paper_filter: str | None = None,
    color: str = "",
    reset: str = "",
) -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    logger.info("stream_ask: question=%r, model=%s, top_k=%d", question, model, top_k)

    chunks = retrieve(question, top_k=top_k, paper_filter=paper_filter)
    context = format_context(chunks)
    user_prompt = build_rag_prompt(question, context)

    client = get_openai_client()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    print(color, end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print(reset)


def pretty_print_result(result: QueryResult) -> None:
    print(result.answer)
    print()
    for i, source in enumerate(result.sources, start=1):
        print(f"  [{i}] {source.paper_id} | {source.section} | score={source.relevance_score:.3f}")
