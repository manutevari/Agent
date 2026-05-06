"""Tiny offline sanity check."""

from pathlib import Path

from multi_agent import build_corpus, retrieve


def main() -> None:
    sample = Path("artifacts/sample.txt")
    sample.parent.mkdir(exist_ok=True)
    sample.write_text("Alpha beta table value 42. This is a tiny test document.", encoding="utf-8")
    corpus, note = build_corpus(sample)
    hits = retrieve(corpus, "value 42", 3)
    assert corpus and hits and "42" in hits[0]["text"]
    print("OK")
    print(note)


if __name__ == "__main__":
    main()
