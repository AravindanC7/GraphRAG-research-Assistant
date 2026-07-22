"""Load research papers (PDF) into a simple in-memory representation."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class LoadedPaper:
    id: str       # stable id derived from filename (improve with DOI later)
    title: str    # best-effort; filename for now
    path: str
    text: str


def load_pdf(path: Path) -> LoadedPaper:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    return LoadedPaper(
        id=path.stem,
        title=path.stem.replace("_", " ").replace("-", " "),
        path=str(path),
        text=text,
    )


def load_papers(papers_dir: str) -> list[LoadedPaper]:
    pdfs = sorted(Path(papers_dir).glob("*.pdf"))
    return [load_pdf(p) for p in pdfs]
