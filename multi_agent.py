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


ProviderName = Literal["grok", "gemini", "openai"]


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


def sentence_chunks(text: str, max_words: int = 220, overlap_words: int = 35) -> Generator[str, None, None]:
    """Generator that turns PDF text into overlapping semantic-ish chunks."""

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
    """List PDF members from the main data archive."""

    with zipfile.ZipFile(data_zip) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".pdf")]


def read_pdf_text_from_zip(data_zip: Path, member_name: str, max_pages: int = 8) -> str:
    """Read PDF text directly from the zip without extracting the dataset."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency check happens at runtime
        raise RuntimeError("Install pypdf to read the main PDF corpus: pip install pypdf") from exc

    with zipfile.ZipFile(data_zip) as archive:
        raw = archive.read(member_name)
    reader = PdfReader(BytesIO(raw))
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def build_corpus(data_zip: Path, max_docs: int = 16, max_pages: int = 8) -> Tuple[List[Dict[str, Any]], str]:
    """Build a lightweight lexical corpus from data_suryakant.zip."""

    if not data_zip.exists():
        raise FileNotFoundError(f"Main data zip not found: {data_zip}")

    chunks: List[CorpusChunk] = []
    pdf_names = pdf_names_from_zip(data_zip)
    selected = pdf_names[:max_docs]
    keyword_counter: Counter[str] = Counter()

    for source in selected:
        try:
            text = read_pdf_text_from_zip(data_zip, source, max_pages=max_pages)
        except RuntimeError:
            text = f"PDF text extraction dependency is unavailable. Source file: {source}"
        words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)]
        keyword_counter.update(words)
        for idx, chunk in enumerate(sentence_chunks(text)):
            if chunk:
                chunks.append(CorpusChunk(source=source, chunk_id=idx, text=chunk))

    top_terms = [term for term, _ in keyword_counter.most_common(12)]
    summary = (
        f"Main data archive: {data_zip}. "
        f"PDF files: {len(pdf_names)}; indexed files: {len(selected)}; chunks: {len(chunks)}; "
        f"frequent terms: {', '.join(top_terms[:10])}."
    )
    return [chunk.__dict__ for chunk in chunks], summary


def lexical_retrieve(corpus: List[Dict[str, Any]], query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Regular-expression token retrieval over the main data chunks."""

    query_terms = Counter(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower()))
    if not query_terms:
        return corpus[:top_k]

    scored: List[Dict[str, Any]] = []
    for item in corpus:
        text = str(item.get("text", ""))
        text_terms = Counter(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()))
        overlap = sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items())
        source_bonus = 1 if any(term in str(item.get("source", "")).lower() for term in query_terms) else 0
        score = float(overlap + source_bonus)
        if score > 0:
            updated = dict(item)
            updated["score"] = score
            scored.append(updated)

    scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return scored[:top_k] or corpus[:top_k]


def format_context(chunks: List[Dict[str, Any]], max_chars: int = 5000) -> str:
    """Format retrieved chunks for model prompts."""

    blocks = []
    for item in chunks:
        text = str(item.get("text", "")).strip()
        source = item.get("source", "unknown")
        chunk_id = item.get("chunk_id", 0)
        blocks.append(f"[{source} :: chunk {chunk_id}]\n{text}")
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

        if provider in {"grok", "openai"}:
            if OpenAI is None:
                raise RuntimeError("Install openai: pip install openai")
            if provider == "grok":
                api_key = os.getenv("GROK_API_KEY")
                base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
                self.model = os.getenv("GROK_MODEL", "grok-2-latest")
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

        if self.provider in {"grok", "openai"}:
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
async def executor_agent(state: AgentState, adapter: ChatAdapter, batch_size: int = 3) -> AgentState:
    results: List[Dict[str, Any]] = []
    task_batches = list(chunked(state.get("tasks", []), batch_size))

    async def run_task(task: str) -> Dict[str, Any]:
        retrieved = lexical_retrieve(state.get("corpus", []), task, top_k=4)
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
    if selected_provider not in {"grok", "gemini", "openai"}:
        raise ValueError("LLM_PROVIDER must be one of: grok, gemini, openai")

    selected_zip = data_zip or Path(os.getenv("DATA_ZIP", "data_suryakant.zip"))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph planner/executor/verifier prototype.")
    parser.add_argument("--goal", type=str, default="Research and summarize the top 3 trends in generative AI for 2025")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--provider", choices=["grok", "gemini", "openai"], default=None)
    parser.add_argument("--data-zip", type=Path, default=Path(os.getenv("DATA_ZIP", "C:/Users/Dell/Downloads/data_suryakant.zip")))
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
