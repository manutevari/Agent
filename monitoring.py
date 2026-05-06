"""Optional monitoring and feedback loop for the RAG chatbot.

Local JSONL logging always works. Weights & Biases and Evidently are enabled
only when their packages/configuration are available, so deployment does not
break if monitoring credentials are absent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MONITOR_DIR = Path(os.getenv("MONITOR_DIR", "artifacts/monitoring"))
EVENT_LOG = MONITOR_DIR / "rag_events.jsonl"
FEEDBACK_LOG = MONITOR_DIR / "feedback.jsonl"


def _safe_text(value: Any, max_chars: int = 4000) -> str:
    return str(value or "")[:max_chars]


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_jsonl(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def init_wandb() -> Optional[Any]:
    """Initialize W&B if configured."""

    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        if wandb.run is None:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "scientific-rag-chatbot"),
                entity=os.getenv("WANDB_ENTITY") or None,
                mode=os.getenv("WANDB_MODE", "online"),
                reinit=True,
            )
        return wandb
    except Exception:
        return None


def init_langsmith() -> Optional[Any]:
    """Initialize LangSmith client when configured."""

    if not os.getenv("LANGSMITH_API_KEY"):
        return None
    try:
        from langsmith import Client
    except ImportError:
        return None
    try:
        return Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    except Exception:
        return None


def log_langsmith_run(
    question: str,
    answer: str,
    provider: str,
    model: str,
    sources: List[Dict[str, Any]],
    retrieval_mode: str,
    latency_s: float,
) -> None:
    """Create a lightweight LangSmith run for observability/eval datasets."""

    client = init_langsmith()
    if client is None:
        return
    try:
        project_name = os.getenv("LANGSMITH_PROJECT", "scientific-rag-chatbot")
        run = client.create_run(
            name="rag_chat_answer",
            run_type="chain",
            project_name=project_name,
            inputs={"question": question, "retrieval_mode": retrieval_mode},
            outputs={"answer": answer},
            extra={
                "metadata": {
                    "provider": provider,
                    "model": model,
                    "latency_s": latency_s,
                    "source_count": len(sources),
                    "sources": [
                        {
                            "source": source.get("source"),
                            "section": source.get("section"),
                            "page_start": source.get("page_start"),
                            "page_end": source.get("page_end"),
                            "kind": source.get("kind"),
                            "score": source.get("score", 0.0),
                        }
                        for source in sources
                    ],
                }
            },
        )
        client.update_run(run.id, end_time=time.time())
    except Exception:
        pass


def log_rag_event(
    question: str,
    answer: str,
    provider: str,
    model: str,
    latency_s: float,
    sources: List[Dict[str, Any]],
    retrieval_mode: str,
) -> Dict[str, Any]:
    """Log one chatbot answer to local storage and W&B when available."""

    event = {
        "timestamp": time.time(),
        "question": _safe_text(question),
        "answer": _safe_text(answer),
        "provider": provider,
        "model": model,
        "latency_s": float(latency_s or 0.0),
        "retrieval_mode": retrieval_mode,
        "source_count": len(sources),
        "top_score": float(sources[0].get("score", 0.0)) if sources else 0.0,
        "table_sources": sum(1 for source in sources if source.get("kind") == "table"),
        "numeric_sources": sum(1 for source in sources if source.get("numbers")),
        "sources": [
            {
                "source": source.get("source"),
                "section": source.get("section"),
                "page_start": source.get("page_start"),
                "page_end": source.get("page_end"),
                "kind": source.get("kind"),
                "score": source.get("score", 0.0),
            }
            for source in sources
        ],
    }
    append_jsonl(EVENT_LOG, event)

    wandb = init_wandb()
    if wandb is not None:
        try:
            wandb.log(
                {
                    "rag/latency_s": event["latency_s"],
                    "rag/source_count": event["source_count"],
                    "rag/top_score": event["top_score"],
                    "rag/table_sources": event["table_sources"],
                    "rag/numeric_sources": event["numeric_sources"],
                    "rag/provider": provider,
                    "rag/retrieval_mode": retrieval_mode,
                }
            )
        except Exception:
            pass
    log_langsmith_run(
        question=question,
        answer=answer,
        provider=provider,
        model=model,
        sources=sources,
        retrieval_mode=retrieval_mode,
        latency_s=latency_s,
    )
    return event


def log_feedback(
    question: str,
    answer: str,
    rating: str,
    comment: str = "",
    provider: str = "",
    retrieval_mode: str = "",
) -> Dict[str, Any]:
    """Record user feedback for future tuning."""

    payload = {
        "timestamp": time.time(),
        "question": _safe_text(question),
        "answer": _safe_text(answer),
        "rating": rating,
        "comment": _safe_text(comment, max_chars=1000),
        "provider": provider,
        "retrieval_mode": retrieval_mode,
    }
    append_jsonl(FEEDBACK_LOG, payload)

    wandb = init_wandb()
    if wandb is not None:
        try:
            wandb.log({"feedback/rating": 1 if rating == "up" else 0, "feedback/comment_present": bool(comment)})
        except Exception:
            pass
    client = init_langsmith()
    if client is not None:
        try:
            dataset_name = os.getenv("LANGSMITH_FEEDBACK_DATASET", "scientific-rag-feedback")
            try:
                dataset = client.read_dataset(dataset_name=dataset_name)
            except Exception:
                dataset = client.create_dataset(dataset_name=dataset_name, description="User feedback for RAG chatbot answers.")
            client.create_example(
                inputs={"question": question},
                outputs={"answer": answer, "rating": rating, "comment": comment},
                dataset_id=dataset.id,
                metadata={"provider": provider, "retrieval_mode": retrieval_mode},
            )
        except Exception:
            pass
    return payload


def monitoring_summary() -> Dict[str, Any]:
    events = read_jsonl(EVENT_LOG)
    feedback = read_jsonl(FEEDBACK_LOG)
    thumbs_up = sum(1 for item in feedback if item.get("rating") == "up")
    thumbs_down = sum(1 for item in feedback if item.get("rating") == "down")
    avg_latency = sum(float(item.get("latency_s", 0.0)) for item in events) / max(len(events), 1)
    return {
        "events": len(events),
        "feedback": len(feedback),
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "avg_latency_s": round(avg_latency, 3),
    }


def write_evidently_report(output_path: Path | None = None) -> Path:
    """Create an Evidently-style HTML report from logged events and feedback."""

    output_path = output_path or (MONITOR_DIR / "evidently_report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = read_jsonl(EVENT_LOG, limit=1000)
    feedback = read_jsonl(FEEDBACK_LOG, limit=1000)

    try:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataSummaryPreset

        frame = pd.DataFrame(events or [{"latency_s": 0, "source_count": 0, "top_score": 0}])
        report = Report([DataSummaryPreset()])
        snapshot = report.run(frame, None)
        snapshot.save_html(str(output_path))
        return output_path
    except Exception:
        summary = monitoring_summary()
        rows = "\n".join(
            f"<tr><td>{item.get('provider')}</td><td>{item.get('retrieval_mode')}</td>"
            f"<td>{item.get('latency_s')}</td><td>{item.get('source_count')}</td>"
            f"<td>{item.get('top_score')}</td></tr>"
            for item in events[-100:]
        )
        feedback_rows = "\n".join(
            f"<tr><td>{item.get('rating')}</td><td>{_safe_text(item.get('comment'), 300)}</td></tr>"
            for item in feedback[-100:]
        )
        html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>RAG Monitoring Report</title></head>
<body>
<h1>RAG Monitoring Report</h1>
<p>Events: {summary['events']} | Feedback: {summary['feedback']} | Up: {summary['thumbs_up']} | Down: {summary['thumbs_down']} | Avg latency: {summary['avg_latency_s']}s</p>
<h2>Recent Events</h2>
<table border="1" cellpadding="4"><tr><th>Provider</th><th>Retrieval</th><th>Latency</th><th>Sources</th><th>Top score</th></tr>{rows}</table>
<h2>Recent Feedback</h2>
<table border="1" cellpadding="4"><tr><th>Rating</th><th>Comment</th></tr>{feedback_rows}</table>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")
        return output_path
