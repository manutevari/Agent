"""Multi-agent prototype: planner -> executor -> verifier.

The implementation intentionally demonstrates common Python tools used in
agentic applications: decorators, async coroutines, exception handling, context
managers, generators, concurrency, parallelism, pickling, file handling, regex,
Pandas, NumPy, Matplotlib, Seaborn, Plotly, and provider adapters for Grok,
Gemini, and OpenAI/ChatGPT-style APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import pickle
import re
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - allows stdlib-only data inspection
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency check happens at runtime
    OpenAI = None  # type: ignore[assignment]

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - dependency check happens at runtime
    genai = None  # type: ignore[assignment]


ProviderName = Literal["local", "grok", "gemini", "openai", "huggingface"]


class AgentState(TypedDict, total=False):
    """Shared LangGraph state."""

    goal: str
    tasks: List[str]
    results: List[Dict[str, Any]]
    critique: str
    approved: bool
    score: int
    iteration: int
    max_iterations: int
    metrics: List[Dict[str, Any]]
    data_zip: str
    corpus: List[Dict[str, Any]]
    corpus_summary: str


@dataclass
class LLMResponse:
    """Normalized response from any provider."""

    text: str
    choices: List[str] = field(default_factory=list)
    provider: str = "unknown"
    model: str = "unknown"
    latency_s: float = 0.0


@dataclass
class CorpusChunk:
    """Small document chunk retrieved from the main PDF zip corpus."""

    source: str
    chunk_id: int
    text: str
    section: str = "Document"
    page_start: int = 1
    page_end: int = 1
    kind: str = "text"
    numbers: List[str] = field(default_factory=list)
    score: float = 0.0


def timed(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records function duration without changing its result."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    print(f"{label} finished in {time.perf_counter() - start:.2f}s")

            return async_wrapper

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                print(f"{label} finished in {time.perf_counter() - start:.2f}s")

        return wrapper

    return decorator


def retry(max_attempts: int = 3, delay_s: float = 0.5) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator with exception handling for transient provider errors."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - demo handles provider adapters uniformly
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    await asyncio.sleep(delay_s * attempt)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts") from last_error

        return wrapper

    return decorator


@contextmanager
def managed_artifact_dir(path: Path) -> Generator[Path, None, None]:
    """Context manager for artifact file handling."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        marker = path / ".last_run"
        marker.write_text(str(time.time()), encoding="utf-8")


def chunked(items: Iterable[str], size: int) -> Generator[List[str], None, None]:
    """Generator that yields task batches."""

    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


SECTION_RE = re.compile(
    r"^(abstract|introduction|results?|discussion|methods?|materials and methods|"
    r"conclusion|references|supplementary|figure\s+\d+|table\s+\d+|"
    r"\d+(?:\.\d+)*\s+[A-Z][^\n]{3,100})$",
    flags=re.IGNORECASE,
)


def extract_numbers(text: str) -> List[str]:
    """Extract numeric and structural tokens for high-precision retrieval."""

    number_re = r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s?(?:A|Å|nm|um|µm|kDa|Da|%|deg|°|M|mM|uM|µM|pH|K|C))?"
    return [item.strip() for item in re.findall(number_re, text)]


def is_section_heading(line: str) -> bool:
    """Detect section titles without relying on fixed character offsets."""

    stripped = re.sub(r"\s+", " ", line).strip()
    if not stripped or len(stripped) > 120:
        return False
    if SECTION_RE.match(stripped):
        return True
    words = stripped.split()
    return 1 <= len(words) <= 10 and stripped[:1].isupper() and sum(ch.isdigit() for ch in stripped) <= 4 and not stripped.endswith(".")


def sentence_chunks(text: str, max_words: int = 220, overlap_words: int = 35) -> Generator[str, None, None]:
    """Generator that chunks within sections and sentences, never by arbitrary character count."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if current and current_words + len(words) > max_words:
            yield " ".join(current).strip()
            tail_words = " ".join(current).split()[-overlap_words:]
            current = [" ".join(tail_words)] if tail_words else []
            current_words = len(tail_words)
        current.append(sentence)
        current_words += len(words)

    if current:
        yield " ".join(current).strip()


def pdf_names_from_zip(data_zip: Path) -> List[str]:
    """List PDF members from a data archive."""

    with zipfile.ZipFile(data_zip) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".pdf")]


def pdf_sources_from_path(data_path: Path) -> List[str]:
    """List PDFs from a zip file or a single uploaded PDF."""

    if data_path.suffix.lower() == ".pdf":
        return [data_path.name]
    if data_path.suffix.lower() == ".zip":
        return pdf_names_from_zip(data_path)
    raise ValueError("Upload a .zip containing PDFs or a single .pdf file.")


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".png", ".jpg", ".jpeg", ".webp"}


def supported_sources_from_path(data_path: Path) -> List[str]:
    """List supported document members from a file or zip archive."""

    suffix = data_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(data_path) as archive:
            return [name for name in archive.namelist() if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS and not name.endswith("/")]
    if suffix in SUPPORTED_EXTENSIONS:
        return [data_path.name]
    raise ValueError(f"Unsupported file type: {suffix}. Upload PDF, image, text, Excel/CSV/JSON, or ZIP.")


def read_source_bytes(data_path: Path, source: str) -> bytes:
    """Read a source from a direct file or a zip member."""

    if data_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(data_path) as archive:
            return archive.read(source)
    return data_path.read_bytes()


def read_text_bytes(raw: bytes, source: str) -> str:
    """Read text-like files with a forgiving decoder."""

    suffix = Path(source).suffix.lower()
    if suffix == ".json":
        try:
            obj = json.loads(raw.decode("utf-8"))
            return json.dumps(obj, indent=2, ensure_ascii=True)
        except Exception:
            pass
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def read_tabular_bytes(raw: bytes, source: str) -> str:
    """Read CSV/TSV/Excel as markdown-like text while preserving rows and columns."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas and openpyxl to read spreadsheet files.") from exc

    suffix = Path(source).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(BytesIO(raw), sep=sep)
        return frame.to_markdown(index=False)

    sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
    blocks = []
    for sheet_name, frame in sheets.items():
        blocks.append(f"Sheet: {sheet_name}\n{frame.to_markdown(index=False)}")
    return "\n\n".join(blocks)


def read_image_bytes(raw: bytes, source: str) -> str:
    """Extract image text when OCR is available, otherwise keep image metadata."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pillow to inspect image files.") from exc

    image = Image.open(BytesIO(raw))
    meta = f"Image file {source}; format={image.format}; size={image.size[0]}x{image.size[1]}; mode={image.mode}."
    try:
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
        return f"{meta}\nOCR text:\n{text}" if text else f"{meta}\nNo OCR text detected."
    except Exception:
        return f"{meta}\nOCR is not configured in this environment."


def read_pdf_pages_from_bytes(raw: bytes, max_pages: int = 8) -> List[Dict[str, Any]]:
    """Extract page text and tables while preserving page boundaries."""

    pages: List[Dict[str, Any]] = []
    try:
        import pdfplumber

        try:
            with pdfplumber.open(BytesIO(raw)) as pdf:
                for page_number, page in enumerate(pdf.pages[:max_pages], start=1):
                    text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                    table_blocks = []
                    for table_idx, table in enumerate(page.extract_tables() or [], start=1):
                        rows = [" | ".join("" if cell is None else str(cell).strip() for cell in row) for row in table if row]
                        if rows:
                            table_blocks.append({"table_id": table_idx, "text": "\n".join(rows)})
                    pages.append({"page": page_number, "text": text, "tables": table_blocks})
            return pages
        except Exception:
            pages = []
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency check happens at runtime
        raise RuntimeError("Install pypdf or pdfplumber to read PDF text.") from exc

    reader = PdfReader(BytesIO(raw))
    for page_number, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": page_number, "text": text, "tables": []})
    return pages


def read_pdf_pages(data_path: Path, source: str, max_pages: int = 8) -> List[Dict[str, Any]]:
    """Read one PDF from a zip or from a direct PDF path."""

    if data_path.suffix.lower() == ".pdf":
        raw = data_path.read_bytes()
    else:
        with zipfile.ZipFile(data_path) as archive:
            raw = archive.read(source)
    return read_pdf_pages_from_bytes(raw, max_pages=max_pages)


def read_generic_pages(data_path: Path, source: str, max_pages: int = 8) -> List[Dict[str, Any]]:
    """Read any supported source into page-like records."""

    suffix = Path(source).suffix.lower()
    raw = read_source_bytes(data_path, source)
    if suffix == ".pdf":
        return read_pdf_pages_from_bytes(raw, max_pages=max_pages)
    if suffix in {".txt", ".md", ".json"}:
        text = read_text_bytes(raw, source)
        return [{"page": 1, "text": text, "tables": []}]
    if suffix in {".csv", ".tsv", ".xlsx", ".xls"}:
        text = read_tabular_bytes(raw, source)
        return [{"page": 1, "text": text, "tables": [{"table_id": 1, "text": text}]}]
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        text = read_image_bytes(raw, source)
        return [{"page": 1, "text": text, "tables": []}]
    return [{"page": 1, "text": f"Unsupported member skipped: {source}", "tables": []}]


def read_pdf_text_from_zip(data_zip: Path, member_name: str, max_pages: int = 8) -> str:
    """Backward-compatible helper that reads PDF text from a zip archive."""

    return "\n".join(page["text"] for page in read_pdf_pages(data_zip, member_name, max_pages=max_pages))


def section_chunks_from_pages(source: str, pages: List[Dict[str, Any]]) -> Generator[CorpusChunk, None, None]:
    """Create chunks that preserve source, page, section, and table context."""

    chunk_id = 0
    current_section = "Document"
    section_buffer: List[Tuple[int, str]] = []

    def flush_buffer() -> Generator[CorpusChunk, None, None]:
        nonlocal chunk_id, section_buffer
        if not section_buffer:
            return
        page_start = section_buffer[0][0]
        page_end = section_buffer[-1][0]
        text = "\n".join(line for _, line in section_buffer).strip()
        for chunk in sentence_chunks(text, max_words=260, overlap_words=45):
            if chunk:
                yield CorpusChunk(
                    source=source,
                    chunk_id=chunk_id,
                    text=chunk,
                    section=current_section,
                    page_start=page_start,
                    page_end=page_end,
                    kind="text",
                    numbers=extract_numbers(chunk),
                )
                chunk_id += 1
        section_buffer = []

    for page in pages:
        page_number = int(page.get("page", 1))
        for raw_line in str(page.get("text", "")).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if is_section_heading(line):
                yield from flush_buffer()
                current_section = line
            else:
                section_buffer.append((page_number, line))

        for table in page.get("tables", []):
            yield from flush_buffer()
            table_text = str(table.get("text", "")).strip()
            if table_text:
                table_label = f"{current_section} / Table {table.get('table_id', '?')}"
                yield CorpusChunk(
                    source=source,
                    chunk_id=chunk_id,
                    text=table_text,
                    section=table_label,
                    page_start=page_number,
                    page_end=page_number,
                    kind="table",
                    numbers=extract_numbers(table_text),
                )
                chunk_id += 1

    yield from flush_buffer()


def build_corpus(data_zip: Path, max_docs: int = 16, max_pages: int = 8) -> Tuple[List[Dict[str, Any]], str]:
    """Build a section-aware corpus from one uploaded file or archive."""

    if not data_zip.exists():
        raise FileNotFoundError(f"Main data file not found: {data_zip}")

    chunks: List[CorpusChunk] = []
    source_names = supported_sources_from_path(data_zip)
    selected = source_names[:max_docs]
    keyword_counter: Counter[str] = Counter()
    table_count = 0

    for source in selected:
        try:
            pages = read_generic_pages(data_zip, source, max_pages=max_pages)
        except Exception as exc:
            pages = [{"page": 1, "text": f"Document extraction failed for {source}: {exc}", "tables": []}]
        doc_chunks = list(section_chunks_from_pages(source, pages))
        if not doc_chunks:
            doc_chunks = [
                CorpusChunk(
                    source=source,
                    chunk_id=0,
                    text=f"No extractable text found in the indexed pages for {source}. Increase pages per PDF or use a text-searchable PDF.",
                    section="Extraction note",
                    page_start=1,
                    page_end=max_pages,
                    kind="note",
                    numbers=extract_numbers(source),
                )
            ]
        table_count += sum(1 for chunk in doc_chunks if chunk.kind == "table")
        chunks.extend(doc_chunks)
        text = "\n".join(chunk.text for chunk in doc_chunks)
        words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)]
        keyword_counter.update(words)

    top_terms = [term for term, _ in keyword_counter.most_common(12)]
    summary = (
        f"Main data file: {data_zip}. "
        f"Supported files: {len(source_names)}; indexed files: {len(selected)}; chunks: {len(chunks)}; "
        f"table chunks: {table_count}; "
        f"frequent terms: {', '.join(top_terms[:10])}."
    )
    return [chunk.__dict__ for chunk in chunks], summary


def build_corpus_from_paths(data_paths: List[Path], max_docs: int = 16, max_pages: int = 8) -> Tuple[List[Dict[str, Any]], str]:
    """Build one aggregate corpus from multiple uploaded files."""

    all_chunks: List[Dict[str, Any]] = []
    summaries = []
    per_file_limit = max(1, max_docs)
    for path in data_paths:
        chunks, summary = build_corpus(path, max_docs=per_file_limit, max_pages=max_pages)
        all_chunks.extend(chunks)
        summaries.append(summary)
    keyword_counter: Counter[str] = Counter()
    for chunk in all_chunks:
        keyword_counter.update(re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", str(chunk.get("text", "")).lower()))
    top_terms = ", ".join(term for term, _ in keyword_counter.most_common(10))
    summary = f"Uploaded groups: {len(data_paths)}; aggregate chunks: {len(all_chunks)}; frequent terms: {top_terms}.\n" + "\n".join(summaries)
    return all_chunks, summary


def lexical_retrieve(corpus: List[Dict[str, Any]], query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """Hybrid lexical, numeric, section, and adjacency retrieval over corpus chunks."""

    query_terms = Counter(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower()))
    query_numbers = set(extract_numbers(query))
    if not query_terms:
        return corpus[:top_k]

    scored: List[Dict[str, Any]] = []
    by_key = {(item.get("source"), item.get("chunk_id")): item for item in corpus}
    for idx, item in enumerate(corpus):
        text = str(item.get("text", ""))
        section = str(item.get("section", ""))
        source = str(item.get("source", ""))
        haystack = f"{source} {section} {text}".lower()
        text_terms = Counter(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", haystack))
        overlap = sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items())
        phrase_bonus = 4 if query.lower() in haystack else 0
        source_bonus = 2 if any(term in source.lower() for term in query_terms) else 0
        section_bonus = 3 if any(term in section.lower() for term in query_terms) else 0
        numeric_bonus = 5 * len(query_numbers.intersection(set(item.get("numbers", []))))
        table_bonus = 2 if item.get("kind") == "table" and (query_numbers or "table" in query.lower()) else 0
        score = float(overlap + phrase_bonus + source_bonus + section_bonus + numeric_bonus + table_bonus)
        if score > 0:
            updated = dict(item)
            updated["score"] = score
            scored.append(updated)

    scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    expanded: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any]] = set()
    for item in scored[:top_k]:
        source = item.get("source")
        chunk_id = item.get("chunk_id")
        for neighbor_id in (chunk_id - 1, chunk_id, chunk_id + 1) if isinstance(chunk_id, int) else (chunk_id,):
            candidate = by_key.get((source, neighbor_id))
            if not candidate:
                continue
            key = (candidate.get("source"), candidate.get("chunk_id"))
            if key in seen:
                continue
            candidate = dict(candidate)
            candidate["score"] = max(float(candidate.get("score", 0.0)), float(item.get("score", 0.0)) - abs(neighbor_id - chunk_id))
            expanded.append(candidate)
            seen.add(key)
    expanded.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return expanded[: max(top_k, 1)] or corpus[:top_k]


def llamaindex_retrieve(corpus: List[Dict[str, Any]], query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """Retrieve with LlamaIndex BM25 over the preserved section/table chunks."""

    try:
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError:
        return lexical_retrieve(corpus, query, top_k=top_k)

    nodes = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in corpus:
        node_id = f"{item.get('source')}::{item.get('chunk_id')}"
        metadata = {
            "source": item.get("source"),
            "chunk_id": item.get("chunk_id"),
            "section": item.get("section"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "kind": item.get("kind"),
            "numbers": ", ".join(item.get("numbers", [])),
        }
        node = TextNode(id_=node_id, text=str(item.get("text", "")), metadata=metadata)
        nodes.append(node)
        by_id[node_id] = item

    if not nodes:
        return []

    try:
        retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
        retrieved = retriever.retrieve(query)
    except Exception:
        return lexical_retrieve(corpus, query, top_k=top_k)

    results: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any]] = set()
    for hit in retrieved:
        node = hit.node
        original = dict(by_id.get(node.node_id, {}))
        if not original:
            original = dict(node.metadata)
            original["text"] = node.text
        original["score"] = float(hit.score or 0.0)
        key = (original.get("source"), original.get("chunk_id"))
        if key not in seen:
            results.append(original)
            seen.add(key)

    numeric_hits = lexical_retrieve(corpus, query, top_k=top_k)
    for hit in numeric_hits:
        key = (hit.get("source"), hit.get("chunk_id"))
        if key not in seen:
            results.append(hit)
            seen.add(key)

    results.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return results[:top_k] or lexical_retrieve(corpus, query, top_k=top_k)


def corpus_as_langchain_documents(corpus: List[Dict[str, Any]]) -> List[Any]:
    """Represent preserved corpus chunks as LangChain Documents."""

    try:
        from langchain_core.documents import Document
    except ImportError:
        return []

    documents = []
    for item in corpus:
        documents.append(
            Document(
                page_content=str(item.get("text", "")),
                metadata={
                    "source": item.get("source"),
                    "chunk_id": item.get("chunk_id"),
                    "section": item.get("section"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "kind": item.get("kind"),
                    "numbers": item.get("numbers", []),
                    "score": item.get("score", 0.0),
                },
            )
        )
    return documents


def build_chat_prompt(history_text: str, question: str, context: str) -> str:
    """Build the chatbot prompt through LangChain when available."""

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        return (
            f"Conversation history:\n{history_text or 'No prior messages.'}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Return a direct, grounded answer with citations. Use bullet points or a compact table when useful."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHATBOT_SYSTEM),
            (
                "human",
                "Conversation history:\n{history}\n\n"
                "Question:\n{question}\n\n"
                "Retrieved context:\n{context}\n\n"
                "Return a direct, grounded answer with citations. Use bullet points or a compact table when useful.",
            ),
        ]
    )
    return prompt.format_messages(history=history_text or "No prior messages.", question=question, context=context)[1].content


def format_context(chunks: List[Dict[str, Any]], max_chars: int = 5000) -> str:
    """Format retrieved chunks for model prompts."""

    blocks = []
    for item in chunks:
        text = str(item.get("text", "")).strip()
        source = item.get("source", "unknown")
        chunk_id = item.get("chunk_id", 0)
        section = item.get("section", "Document")
        page_start = item.get("page_start", "?")
        page_end = item.get("page_end", page_start)
        kind = item.get("kind", "text")
        blocks.append(f"[{source} :: pp. {page_start}-{page_end} :: {section} :: {kind} :: chunk {chunk_id}]\n{text}")
    return "\n\n".join(blocks)[:max_chars]


def extract_json(text: str, fallback: Any) -> Any:
    """Use regular expressions to recover JSON from model responses."""

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    start_chars = {"[": "]", "{": "}"}
    for start_char, end_char in start_chars.items():
        start = candidate.find(start_char)
        end = candidate.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
            break
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return fallback


class ChatAdapter:
    """Provider adapter for sending prompts, parameters, and multiple responses."""

    def __init__(self, provider: ProviderName, temperature: float = 0.2, max_tokens: int = 900, responses: int = 1):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.responses = max(1, responses)
        self.model = "extractive-local"
        self.client = None

        if provider == "local":
            return

        if provider in {"grok", "openai", "huggingface"}:
            if OpenAI is None:
                raise RuntimeError("Install openai: pip install openai")
            if provider == "grok":
                api_key = os.getenv("GROK_API_KEY")
                base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
                self.model = os.getenv("GROK_MODEL", "grok-2-latest")
            elif provider == "huggingface":
                api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
                self.model = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL")
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            if not api_key:
                raise RuntimeError(f"Missing API key for provider: {provider}")
            self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        elif provider == "gemini":
            if genai is None:
                raise RuntimeError("Install google-generativeai: pip install google-generativeai")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY for Gemini")
            genai.configure(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(max_attempts=3)
    async def complete(self, system_prompt: str, human_prompt: str) -> LLMResponse:
        """Coroutine for sending a prompt and retrieving generated text."""

        start = time.perf_counter()

        if self.provider == "local":
            return LLMResponse(
                text=local_extractive_response(human_prompt),
                choices=[],
                provider="local",
                model=self.model,
                latency_s=time.perf_counter() - start,
            )

        if self.provider in {"grok", "openai", "huggingface"}:
            raw_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": human_prompt}]

            def call_openai_compatible() -> Any:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=raw_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.responses,
                )

            response = await asyncio.to_thread(call_openai_compatible)
            choices = [choice.message.content or "" for choice in response.choices]
            return LLMResponse(
                text=choices[0] if choices else "",
                choices=choices,
                provider=self.provider,
                model=self.model,
                latency_s=time.perf_counter() - start,
            )

        prompt = f"{system_prompt}\n\nUser request:\n{human_prompt}"

        def call_gemini() -> Any:
            return self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "candidate_count": self.responses,
                },
            )

        response = await asyncio.to_thread(call_gemini)
        choices = [getattr(candidate.content.parts[0], "text", "") for candidate in getattr(response, "candidates", [])]
        text = getattr(response, "text", "") or (choices[0] if choices else "")
        return LLMResponse(text=text, choices=choices or [text], provider=self.provider, model=self.model, latency_s=time.perf_counter() - start)


PLANNER_SYSTEM = (
    "You are a planning agent. Break the user's goal into at most 5 concrete, actionable tasks. "
    "Use the supplied PDF corpus summary as the main data context. "
    "Return only a valid JSON array of strings."
)

EXECUTOR_SYSTEM = (
    "You are an execution agent. Produce the requested deliverable for one task. "
    "Use the retrieved PDF excerpts as primary evidence. "
    "Return JSON: {\"task\": string, \"output\": string, \"sources\": [string]}."
)

VERIFIER_SYSTEM = (
    "You are a verifier. Evaluate results against the goal. "
    "Return JSON: {\"score\": int, \"critique\": string, \"approved\": bool}."
)

CHATBOT_SYSTEM = (
    "You are a highly accurate context-aware RAG chatbot for scientific PDFs. "
    "Answer only from the supplied context. Preserve numerical values, units, protein/domain names, "
    "mutations, page references, table context, and section names. If evidence is insufficient, say what is missing. "
    "For structural or numerical analysis, compare values explicitly and cite the source chunk labels."
)


def local_extractive_response(prompt: str, max_chars: int = 2200) -> str:
    """API-key-free fallback that returns grounded evidence from retrieved context."""

    if "Return JSON only" in prompt and "Goal:" in prompt and "Corpus summary:" in prompt:
        goal_match = re.search(r"Goal:\n(?P<goal>.*?)(?:\n\nCorpus summary:|\Z)", prompt, flags=re.DOTALL)
        goal = (goal_match.group("goal").strip() if goal_match else "Answer the user query")
        return json.dumps(
            [
                f"Retrieve section, table, and numeric evidence for: {goal}",
                f"Draft a grounded answer with citations for: {goal}",
                "Check the answer against retrieved evidence and identify missing context.",
            ]
        )
    if "Approve only if" in prompt and "Results:" in prompt:
        return json.dumps({"score": 75, "critique": "Local verifier approved based on retrieved evidence availability.", "approved": True})
    if "Return a compact JSON object" in prompt and "Task:" in prompt:
        task_match = re.search(r"Task:\n(?P<task>.*?)(?:\n\nRetrieved excerpts|\Z)", prompt, flags=re.DOTALL)
        task = task_match.group("task").strip() if task_match else "task"
        evidence = local_extractive_response(prompt.replace("Return a compact JSON object", ""), max_chars=max_chars)
        return json.dumps({"task": task, "output": evidence})

    context_match = re.search(r"Retrieved context:\n(?P<context>.*?)(?:\n\nReturn a direct|\Z)", prompt, flags=re.DOTALL)
    if not context_match:
        context_match = re.search(r"Retrieved excerpts from the main data zip:\n(?P<context>.*?)(?:\n\nReturn|\Z)", prompt, flags=re.DOTALL)
    context = context_match.group("context").strip() if context_match else prompt
    blocks = [block.strip() for block in re.split(r"\n\s*\n", context) if block.strip()]
    evidence = []
    for block in blocks[:5]:
        lines = block.splitlines()
        label = lines[0] if lines else "[source unavailable]"
        body = " ".join(lines[1:]).strip()
        evidence.append(f"- {label}: {body[:450]}")
    if not evidence:
        return "I could not find retrievable evidence in the uploaded documents."
    joined = "\n".join(evidence)
    return (
        "API-key-free local evidence mode is active. I cannot synthesize beyond the retrieved text, "
        "but these are the most relevant grounded excerpts:\n\n"
        f"{joined[:max_chars]}"
    )


@timed("planner")
async def planner_agent(state: AgentState, adapter: ChatAdapter) -> AgentState:
    planner_prompt = (
        f"Goal:\n{state['goal']}\n\n"
        f"Corpus summary:\n{state.get('corpus_summary', 'No corpus summary available.')}\n\n"
        "Return JSON only."
    )
    response = await adapter.complete(PLANNER_SYSTEM, planner_prompt)
    fallback = [line.strip("- ").strip() for line in response.text.splitlines() if line.strip()][:5]
    tasks = extract_json(response.text, fallback)
    if not isinstance(tasks, list):
        tasks = fallback
    state["tasks"] = [str(task) for task in tasks[:5]]
    state.setdefault("metrics", []).append({"agent": "planner", "latency_s": response.latency_s, "items": len(state["tasks"])})
    return state


@timed("executor")
async def executor_agent(state: AgentState, adapter: ChatAdapter, batch_size: int = 3, use_llamaindex: bool = True) -> AgentState:
    results: List[Dict[str, Any]] = []
    task_batches = list(chunked(state.get("tasks", []), batch_size))

    async def run_task(task: str) -> Dict[str, Any]:
        retrieve = llamaindex_retrieve if use_llamaindex else lexical_retrieve
        retrieved = retrieve(state.get("corpus", []), task, top_k=4)
        context = format_context(retrieved)
        prompt = (
            f"Task:\n{task}\n\n"
            f"Retrieved excerpts from the main data zip:\n{context}\n\n"
            "Return a compact JSON object grounded in these excerpts."
        )
        response = await adapter.complete(EXECUTOR_SYSTEM, prompt)
        parsed = extract_json(response.text, {"task": task, "output": response.text})
        if not isinstance(parsed, dict):
            parsed = {"task": task, "output": str(parsed)}
        parsed.setdefault("task", task)
        parsed.setdefault("sources", sorted({str(item.get("source", "unknown")) for item in retrieved}))
        parsed["source_chunks"] = retrieved
        parsed["provider"] = response.provider
        parsed["model"] = response.model
        parsed["latency_s"] = round(response.latency_s, 3)
        return parsed

    for batch in task_batches:
        batch_results = await asyncio.gather(*(run_task(task) for task in batch), return_exceptions=True)
        for item in batch_results:
            if isinstance(item, Exception):
                results.append({"task": "unknown", "output": f"Execution error: {item}"})
                continue
            results.append(item)

    state["results"] = results
    state.setdefault("metrics", []).append({"agent": "executor", "latency_s": sum(r.get("latency_s", 0) for r in results), "items": len(results)})
    return state


@timed("verifier")
async def verifier_agent(state: AgentState, adapter: ChatAdapter) -> AgentState:
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Main data: {state.get('data_zip', '')}\n"
        f"Corpus summary: {state.get('corpus_summary', '')}\n"
        f"Results: {json.dumps(state.get('results', []), ensure_ascii=True)}\n\n"
        "Approve only if the result satisfies the goal and cites sources from the main PDF data. Return JSON only."
    )
    response = await adapter.complete(VERIFIER_SYSTEM, prompt)
    parsed = extract_json(response.text, {"score": 50, "critique": "Could not parse verifier output.", "approved": False})
    if not isinstance(parsed, dict):
        parsed = {"score": 50, "critique": str(parsed), "approved": False}
    score = int(parsed.get("score", 0))
    approved = bool(parsed.get("approved", score >= 70))
    state["score"] = score
    state["critique"] = str(parsed.get("critique", ""))
    state["approved"] = approved
    state.setdefault("metrics", []).append({"agent": "verifier", "latency_s": response.latency_s, "items": score})
    return state


async def answer_rag_chat(
    question: str,
    corpus: List[Dict[str, Any]],
    provider: Optional[ProviderName] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 8,
    use_llamaindex: bool = True,
    allow_external_knowledge: bool = False,
) -> Dict[str, Any]:
    """Answer one chatbot query with retrieved section/table-aware context."""

    load_dotenv()
    selected_provider = provider or os.getenv("LLM_PROVIDER", "grok").lower()
    if selected_provider not in {"local", "grok", "gemini", "openai", "huggingface"}:
        raise ValueError("LLM_PROVIDER must be one of: local, grok, gemini, openai, huggingface")

    retrieve = llamaindex_retrieve if use_llamaindex else lexical_retrieve
    retrieved = retrieve(corpus, question, top_k=top_k)
    langchain_documents = corpus_as_langchain_documents(retrieved)
    context = format_context(retrieved, max_chars=12000)
    history_text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in (history or [])[-8:])
    policy = (
        "Use only the uploaded document context. If the answer is not present in the uploaded documents, say so."
        if not allow_external_knowledge
        else "Use uploaded document context first. You may add clearly labeled open-source/general knowledge only when needed."
    )
    human_prompt = build_chat_prompt(history_text=history_text, question=question, context=f"{policy}\n\n{context}")
    adapter = ChatAdapter(provider=selected_provider, temperature=0.1, max_tokens=1600)
    response = await adapter.complete(CHATBOT_SYSTEM, human_prompt)
    return {
        "answer": response.text,
        "sources": retrieved,
        "langchain_document_count": len(langchain_documents),
        "provider": response.provider,
        "model": response.model,
        "latency_s": response.latency_s,
    }


def should_continue(state: AgentState) -> str:
    """Conditional statement for LangGraph routing."""

    if state.get("approved"):
        return "approved"
    if int(state.get("iteration", 1)) >= int(state.get("max_iterations", 3)):
        return "approved"
    return "retry"


def build_graph(adapter: ChatAdapter, max_iterations: int) -> Any:
    """Build explicit LangGraph node/edge wiring."""

    from langgraph.graph import END, START, StateGraph

    async def planner_node(state: AgentState) -> AgentState:
        state["iteration"] = int(state.get("iteration", 0)) + 1
        state["max_iterations"] = max_iterations  # type: ignore[typeddict-unknown-key]
        if state.get("tasks"):
            return state
        return await planner_agent(state, adapter)

    async def executor_node(state: AgentState) -> AgentState:
        return await executor_agent(state, adapter)

    async def verifier_node(state: AgentState) -> AgentState:
        return await verifier_agent(state, adapter)

    async def refine_node(state: AgentState) -> AgentState:
        critique = state.get("critique", "No critique supplied.")
        state["tasks"] = [f"Refine previous results to address this verifier critique: {critique}"]
        return state

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("refine", refine_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges("verifier", should_continue, {"approved": END, "retry": "refine"})
    graph.add_edge("refine", "executor")
    return graph.compile()


def make_metrics_frame(state: AgentState) -> pd.DataFrame:
    """Create typed analytical data with Pandas and NumPy operations."""

    import numpy as np
    import pandas as pd

    rows = state.get("metrics", [])
    frame = pd.DataFrame(rows or [{"agent": "none", "latency_s": 0.0, "items": 0}])
    frame["latency_s"] = pd.to_numeric(frame["latency_s"], errors="coerce").fillna(0.0)
    frame["items"] = pd.to_numeric(frame["items"], errors="coerce").fillna(0).astype(int)
    frame["latency_rank"] = np.arange(1, len(frame) + 1)
    frame["efficiency"] = np.where(frame["latency_s"] > 0, frame["items"] / frame["latency_s"], 0.0)
    return frame


def save_visualizations(frame: pd.DataFrame, artifact_dir: Path) -> None:
    """Save Matplotlib, Seaborn, and Plotly outputs in parallel."""

    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns

    def matplotlib_plot() -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(frame["agent"], frame["latency_s"], marker="o")
        ax.set_title("Agent Latency")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Seconds")
        fig.tight_layout()
        fig.savefig(artifact_dir / "metrics_matplotlib.png")
        plt.close(fig)

    def seaborn_plot() -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=frame, x="agent", y="items", ax=ax)
        ax.set_title("Agent Items")
        fig.tight_layout()
        fig.savefig(artifact_dir / "metrics_seaborn.png")
        plt.close(fig)

    def plotly_plot() -> None:
        plot_frame = frame.copy()
        plot_frame["marker_size"] = plot_frame["efficiency"].clip(lower=1.0)
        fig = px.scatter(plot_frame, x="latency_s", y="items", color="agent", size="marker_size", title="Latency vs Items")
        fig.write_html(artifact_dir / "metrics_plotly.html")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(func) for func in (matplotlib_plot, seaborn_plot, plotly_plot)]
        for future in futures:
            future.result()


def persist_state(state: AgentState, artifact_dir: Path, cache_file: Path) -> None:
    """Write JSON, CSV, and pickle artifacts."""

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    final_json = artifact_dir / "final_state.json"
    final_json.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    try:
        frame = make_metrics_frame(state)
        frame.to_csv(artifact_dir / "metrics.csv", index=False)
        with cache_file.open("wb") as fh:
            pickle.dump({"state": state, "metrics": frame}, fh)
        save_visualizations(frame, artifact_dir)
    except ImportError as exc:
        metrics_path = artifact_dir / "metrics.csv"
        rows = state.get("metrics", [])
        metrics_path.write_text("agent,latency_s,items\n" + "\n".join(f"{row.get('agent')},{row.get('latency_s')},{row.get('items')}" for row in rows), encoding="utf-8")
        with cache_file.open("wb") as fh:
            pickle.dump({"state": state, "metrics": rows, "visualization_error": str(exc)}, fh)


async def run_multi_agent(
    goal: str,
    max_iterations: int = 3,
    provider: Optional[ProviderName] = None,
    data_zip: Optional[Path] = None,
    max_docs: int = 16,
    max_pages: int = 8,
) -> AgentState:
    """Run the multi-agent graph."""

    load_dotenv()
    selected_provider = provider or os.getenv("LLM_PROVIDER", "grok").lower()
    if selected_provider not in {"local", "grok", "gemini", "openai", "huggingface"}:
        raise ValueError("LLM_PROVIDER must be one of: local, grok, gemini, openai, huggingface")

    env_data_path = os.getenv("DATA_ZIP") or os.getenv("DATA_PATH")
    if data_zip is None and not env_data_path:
        raise ValueError("Provide --data-zip or set DATA_ZIP/DATA_PATH to a supported document or archive.")
    selected_zip = data_zip or Path(env_data_path or "")
    corpus, corpus_summary = await asyncio.to_thread(build_corpus, selected_zip, max_docs, max_pages)
    adapter = ChatAdapter(provider=selected_provider)  # type: ignore[arg-type]
    app = build_graph(adapter, max_iterations=max_iterations)
    initial_state: AgentState = {
        "goal": goal,
        "tasks": [],
        "results": [],
        "approved": False,
        "iteration": 0,
        "metrics": [{"agent": "corpus", "latency_s": 0.0, "items": len(corpus)}],
        "data_zip": str(selected_zip),
        "corpus": corpus,
        "corpus_summary": corpus_summary,
    }
    return await app.ainvoke(initial_state)


def agent_state_to_response(state: AgentState) -> Dict[str, Any]:
    """Convert planner/executor/verifier state into a chatbot-style response."""

    results = state.get("results", [])
    answer_parts = []
    sources: List[Dict[str, Any]] = []
    for idx, result in enumerate(results, start=1):
        task = result.get("task", f"Task {idx}")
        output = result.get("output", "")
        answer_parts.append(f"**{idx}. {task}**\n\n{output}")
        for source in result.get("source_chunks", []):
            if isinstance(source, dict):
                sources.append(source)

    critique = state.get("critique", "")
    approved = state.get("approved", False)
    score = state.get("score", "n/a")
    final_answer = "\n\n".join(answer_parts) if answer_parts else "No agent result was produced."
    final_answer += f"\n\n**Verifier:** approved={approved}, score={score}. {critique}"
    return {
        "answer": final_answer,
        "sources": sources,
        "provider": results[0].get("provider", "unknown") if results else "unknown",
        "model": results[0].get("model", "unknown") if results else "unknown",
        "latency_s": sum(float(result.get("latency_s", 0.0)) for result in results),
        "state": state,
    }


async def answer_with_agent_pipeline(
    question: str,
    provider: Optional[ProviderName],
    data_zip: Path,
    max_docs: int = 16,
    max_pages: int = 8,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Answer one user query through Planner -> Executor -> Verifier over the PDF corpus."""

    state = await run_multi_agent(
        goal=question,
        max_iterations=max_iterations,
        provider=provider,
        data_zip=data_zip,
        max_docs=max_docs,
        max_pages=max_pages,
    )
    return agent_state_to_response(state)


async def answer_with_agent_pipeline_from_corpus(
    question: str,
    corpus: List[Dict[str, Any]],
    corpus_summary: str,
    provider: Optional[ProviderName],
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Answer through Planner -> Executor -> Verifier using an already-built aggregate corpus."""

    load_dotenv()
    selected_provider = provider or os.getenv("LLM_PROVIDER", "local").lower()
    if selected_provider not in {"local", "grok", "gemini", "openai", "huggingface"}:
        raise ValueError("LLM_PROVIDER must be one of: local, grok, gemini, openai, huggingface")

    adapter = ChatAdapter(provider=selected_provider)  # type: ignore[arg-type]
    app = build_graph(adapter, max_iterations=max_iterations)
    initial_state: AgentState = {
        "goal": question,
        "tasks": [],
        "results": [],
        "approved": False,
        "iteration": 0,
        "metrics": [{"agent": "corpus", "latency_s": 0.0, "items": len(corpus)}],
        "data_zip": "uploaded aggregate corpus",
        "corpus": corpus,
        "corpus_summary": corpus_summary,
    }
    state = await app.ainvoke(initial_state)
    return agent_state_to_response(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph planner/executor/verifier prototype.")
    parser.add_argument("--goal", type=str, default="Research and summarize the top 3 trends in generative AI for 2025")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--provider", choices=["local", "grok", "gemini", "openai", "huggingface"], default=None)
    parser.add_argument("--data-zip", type=Path, default=Path(os.getenv("DATA_ZIP") or os.getenv("DATA_PATH") or ""))
    parser.add_argument("--max-docs", type=int, default=int(os.getenv("MAX_DOCS", "16")))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "8")))
    parser.add_argument("--inspect-data", action="store_true", help="Build corpus artifacts without calling an LLM provider.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    cache_file = Path(os.getenv("CACHE_FILE", artifact_dir / "agent_cache.pkl"))

    with managed_artifact_dir(artifact_dir) as run_dir:
        try:
            if args.inspect_data:
                corpus, corpus_summary = build_corpus(args.data_zip, args.max_docs, args.max_pages)
                final_state: AgentState = {
                    "goal": args.goal,
                    "tasks": [],
                    "results": [],
                    "critique": corpus_summary,
                    "approved": True,
                    "score": 100,
                    "iteration": 0,
                    "metrics": [{"agent": "corpus", "latency_s": 0.0, "items": len(corpus)}],
                    "data_zip": str(args.data_zip),
                    "corpus": corpus,
                    "corpus_summary": corpus_summary,
                }
            else:
                final_state = asyncio.run(
                    run_multi_agent(
                        args.goal,
                        max_iterations=args.iters,
                        provider=args.provider,
                        data_zip=args.data_zip,
                        max_docs=args.max_docs,
                        max_pages=args.max_pages,
                    )
                )
        except Exception as exc:
            error_state: AgentState = {
                "goal": args.goal,
                "tasks": [],
                "results": [{"task": "runtime", "output": f"Error: {exc}"}],
                "critique": "Runtime failed before verifier approval.",
                "approved": False,
                "score": 0,
                "iteration": 0,
                "metrics": [{"agent": "error", "latency_s": 0.0, "items": 0}],
            }
            persist_state(error_state, run_dir, cache_file)
            raise

        persist_state(final_state, run_dir, cache_file)
        print("\n=== Final state ===")
        print(json.dumps(final_state, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
