from pathlib import Path

from app.logging_config import setup_logging
from app.ingest import ingest_directory

setup_logging()

papers = ingest_directory(Path("data/pdfs/"))

print(f"Done. {len(papers)} paper(s) ingested.")
for paper in papers:
    print(f"  - {paper.paper_id} ({paper.total_chunks} chunks)")
