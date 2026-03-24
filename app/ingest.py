import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import chromadb
import fitz
import openai
import requests

from .env import load_env
from .chunking import chunk_paper
from .models import Chunk, Paper

load_env()

logger = logging.getLogger(__name__)

_chroma_client = None
_openai_client = None


def get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        chroma_host = os.environ.get("CHROMA_HOST", "")
        if chroma_host:
            chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
            _chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            chroma_path = os.environ.get("CHROMA_PATH", "data/processed/chroma")
            _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client.get_or_create_collection("papers")


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])

    return embeddings


def store_chunks(chunks: list[Chunk], embeddings: list[list[float]]):
    collection = get_chroma_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "paper_id": c.paper_id,
            "section": c.section,
            "page_numbers": json.dumps(c.page_numbers),
            "token_count": c.token_count,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def save_paper_metadata(paper: Paper):
    metadata_dir = Path("data/processed/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{paper.paper_id}.json"
    with open(metadata_path, "w") as f:
        f.write(paper.model_dump_json(indent=2))


def is_already_ingested(paper_id: str) -> bool:
    metadata_path = Path("data/processed/metadata") / f"{paper_id}.json"
    return metadata_path.exists()


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def describe_figure(image_bytes: bytes, paper_id: str, page: int, fig_index: int) -> str:
    client = _get_openai_client()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": "Describe this academic figure in detail. Include all data, trends, labels, axes, values, and scientific conclusions visible. Be thorough.",
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def extract_figure_chunks(pdf_path: Path, paper_id: str, start_chunk_index: int) -> list[Chunk]:
    doc = fitz.open(str(pdf_path))
    figure_chunks = []
    fig_num = 0

    for page_num, page in enumerate(doc):
        images = page.get_images()
        for xref, *_ in images:
            pix = fitz.Pixmap(doc, xref)

            if pix.width < 100 or pix.height < 100:
                continue

            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            fig_num += 1
            logger.info(
                "Found figure %d on page %d of %s (size %dx%d)",
                fig_num,
                page_num + 1,
                pdf_path.name,
                pix.width,
                pix.height,
            )

            image_bytes = pix.tobytes("png")
            description = describe_figure(image_bytes, paper_id, page_num, fig_num)

            chunk_index = start_chunk_index + len(figure_chunks)
            chunk_id = f"{paper_id}_{chunk_index}"
            chunk = Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=f"[Figure {fig_num}, page {page_num + 1}] {description}",
                section=f"Figure {fig_num} (page {page_num + 1})",
                page_numbers=[page_num + 1],
                token_count=len(description.split()),
                chunk_index=chunk_index,
            )
            figure_chunks.append(chunk)

    return figure_chunks


def ingest_pdf(pdf_path: Path) -> Paper:
    paper_id = pdf_path.stem

    if is_already_ingested(paper_id):
        logger.info("Paper %s already ingested, skipping.", paper_id)
        metadata_path = Path("data/processed/metadata") / f"{paper_id}.json"
        with open(metadata_path) as f:
            return Paper.model_validate_json(f.read())

    logger.info("Ingesting PDF: %s", pdf_path.name)
    raw_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_paper(raw_text, paper_id)

    all_chunks = list(text_chunks)

    extract_figures = os.environ.get("EXTRACT_FIGURES", "false").lower() == "true"
    if extract_figures:
        figure_chunks = extract_figure_chunks(pdf_path, paper_id, start_chunk_index=len(text_chunks))
        all_chunks.extend(figure_chunks)

    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)
    store_chunks(all_chunks, embeddings)

    paper = Paper(
        paper_id=paper_id,
        filename=pdf_path.name,
        title=paper_id,
        authors=[],
        abstract="",
        total_chunks=len(all_chunks),
        ingested_at=datetime.utcnow(),
    )
    save_paper_metadata(paper)
    logger.info("Ingested %s: %d chunks.", paper_id, len(all_chunks))
    return paper


def ingest_directory(pdf_dir: Path) -> list[Paper]:
    papers = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        paper = ingest_pdf(pdf_path)
        papers.append(paper)
    return papers


def ingest_text(
    raw_text: str,
    paper_id: str,
    title: str,
    authors: list[str] = [],
    abstract: str = "",
) -> Paper:
    if is_already_ingested(paper_id):
        logger.info("Paper %s already ingested, skipping.", paper_id)
        metadata_path = Path("data/processed/metadata") / f"{paper_id}.json"
        with open(metadata_path) as f:
            return Paper.model_validate_json(f.read())

    chunks = chunk_paper(raw_text, paper_id)
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    store_chunks(chunks, embeddings)

    paper = Paper(
        paper_id=paper_id,
        filename=f"{paper_id}.txt",
        title=title,
        authors=authors,
        abstract=abstract,
        total_chunks=len(chunks),
        ingested_at=datetime.utcnow(),
    )
    save_paper_metadata(paper)
    logger.info("Ingested text %s: %d chunks.", paper_id, len(chunks))
    return paper
