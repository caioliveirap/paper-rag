import re
import tiktoken

from .env import load_env
from .models import Chunk

load_env()

_encoding = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def detect_sections(text: str) -> list[tuple[str, str]]:
    # Match numbered headers like "1. Introduction", "1.1 Background", or ALL-CAPS headers
    pattern = re.compile(
        r"(?m)^(?:(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]+)|([A-Z]{3,}[A-Z\s]{0,50}))$"
    )
    matches = list(pattern.finditer(text))

    if not matches:
        return [("full_text", text)]

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections if sections else [("full_text", text)]


def split_section(
    section_name: str,
    section_text: str,
    paper_id: str,
    start_index: int,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    tokens = _encoding.encode(section_text)

    if len(tokens) <= max_tokens:
        chunk_id = f"{paper_id}_{start_index}"
        return [
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=section_text,
                section=section_name,
                token_count=len(tokens),
                chunk_index=start_index,
            )
        ]

    chunks = []
    chunk_index = start_index
    pos = 0

    while pos < len(tokens):
        end = min(pos + max_tokens, len(tokens))
        window_tokens = tokens[pos:end]
        chunk_text = _encoding.decode(window_tokens)
        chunk_id = f"{paper_id}_{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=chunk_text,
                section=section_name,
                token_count=len(window_tokens),
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

        if end == len(tokens):
            break
        pos = end - overlap_tokens

    return chunks


def chunk_paper(raw_text: str, paper_id: str) -> list[Chunk]:
    sections = detect_sections(raw_text)

    all_chunks: list[Chunk] = []
    current_index = 0

    for section_name, section_text in sections:
        section_chunks = split_section(section_name, section_text, paper_id, current_index)
        current_index += len(section_chunks)
        all_chunks.extend(section_chunks)

    # Filter chunks with fewer than 50 tokens
    all_chunks = [c for c in all_chunks if c.token_count >= 50]

    # Reassign sequential chunk_index
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i
        chunk.chunk_id = f"{chunk.paper_id}_{i}"

    return all_chunks
