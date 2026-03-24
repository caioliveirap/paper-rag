from flask import Flask, Response, jsonify, request, stream_with_context

from .ask import ask, get_openai_client, stream_ask
from .env import load_env
from .prompts import SYSTEM_PROMPT, build_rag_prompt
from .retrieve import format_context, retrieve

load_env()

app = Flask(__name__)


def _parse_body() -> tuple[str | None, int, str | None]:
    body = request.get_json(silent=True) or {}
    question = body.get("question")
    top_k = int(body.get("top_k", 5))
    paper_filter = body.get("paper_filter")
    return question, top_k, paper_filter


@app.post("/ask")
def ask_endpoint():
    question, top_k, paper_filter = _parse_body()
    if not question:
        return jsonify({"error": "question is required"}), 400

    result = ask(question, top_k=top_k, paper_filter=paper_filter)
    return jsonify(result.model_dump())


@app.post("/ask/stream")
def ask_stream_endpoint():
    question, top_k, paper_filter = _parse_body()
    if not question:
        return jsonify({"error": "question is required"}), 400

    model_name = __import__("os").environ.get("LLM_MODEL", "gpt-4o-mini")
    chunks = retrieve(question, top_k=top_k, paper_filter=paper_filter)
    context = format_context(chunks)
    user_prompt = build_rag_prompt(question, context)
    client = get_openai_client()

    def generate():
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return Response(stream_with_context(generate()), content_type="text/plain")
