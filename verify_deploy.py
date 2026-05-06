"""Offline sanity checks for the Streamlit RAG deployment package."""

from __future__ import annotations

import zipfile
from pathlib import Path

from multi_agent import build_corpus, format_context, lexical_retrieve


def main() -> None:
    work_dir = Path("artifacts")
    work_dir.mkdir(parents=True, exist_ok=True)
    data_zip = work_dir / "verify_sample_data.zip"
    with zipfile.ZipFile(data_zip, "w") as archive:
        archive.writestr("sample.pdf", b"not a real pdf")

    corpus, summary = build_corpus(data_zip, max_docs=1, max_pages=1)
    assert corpus, "Corpus should contain a fallback chunk for unreadable PDFs"
    hits = lexical_retrieve(corpus, "sample pdf extraction", top_k=3)
    assert hits, "Retriever should return evidence chunks"
    context = format_context(hits)
    assert "sample.pdf" in context, "Formatted context should include source labels"
    print("Deployment sanity check passed.")
    print(summary)


if __name__ == "__main__":
    main()
