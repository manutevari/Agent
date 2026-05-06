"""Streamlit deployment entrypoint for the zip-backed multi-agent RAG prototype."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from multi_agent import build_corpus, format_context, lexical_retrieve, run_multi_agent


st.set_page_config(page_title="PDF Multi-Agent RAG", page_icon="A", layout="wide")


def apply_secret_env() -> None:
    """Copy Streamlit secrets into environment variables expected by the backend."""

    for key in (
        "LLM_PROVIDER",
        "GROK_API_KEY",
        "GROK_BASE_URL",
        "GROK_MODEL",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


def save_uploaded_zip(uploaded_file: Any) -> Path:
    """Persist the uploaded zip for this Streamlit session."""

    temp_dir = Path(tempfile.gettempdir()) / "streamlit_pdf_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_dir / uploaded_file.name
    zip_path.write_bytes(uploaded_file.getbuffer())
    return zip_path


@st.cache_data(show_spinner=False)
def cached_corpus(zip_path: str, max_docs: int, max_pages: int) -> tuple[List[Dict[str, Any]], str]:
    """Cache PDF extraction so UI reruns stay responsive."""

    return build_corpus(Path(zip_path), max_docs=max_docs, max_pages=max_pages)


def run_async(coro: Any) -> Any:
    """Run an async backend call from Streamlit."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def metrics_frame(state: Dict[str, Any]) -> pd.DataFrame:
    rows = state.get("metrics", [])
    frame = pd.DataFrame(rows or [{"agent": "none", "latency_s": 0.0, "items": 0}])
    frame["latency_s"] = pd.to_numeric(frame["latency_s"], errors="coerce").fillna(0.0)
    frame["items"] = pd.to_numeric(frame["items"], errors="coerce").fillna(0).astype(int)
    return frame


apply_secret_env()

st.title("PDF Multi-Agent RAG")
st.caption("Planner -> executor -> verifier over a ZIP corpus of PDFs.")

with st.sidebar:
    st.header("Data")
    uploaded_zip = st.file_uploader("Upload data_suryakant.zip", type=["zip"])
    default_zip = st.text_input("Local zip path", value=os.getenv("DATA_ZIP", ""))
    max_docs = st.slider("PDFs to index", min_value=1, max_value=32, value=16)
    max_pages = st.slider("Pages per PDF", min_value=1, max_value=20, value=8)

    st.header("Model")
    provider = st.selectbox("Provider", options=["grok", "gemini", "openai"], index=0)
    iterations = st.slider("Verifier iterations", min_value=1, max_value=5, value=3)

    st.header("Secrets")
    st.caption("Use Streamlit Cloud secrets for API keys. Local `.env` also works.")

zip_path: Path | None = None
if uploaded_zip is not None:
    zip_path = save_uploaded_zip(uploaded_zip)
elif default_zip:
    zip_path = Path(default_zip)

if zip_path is None:
    st.info("Upload `data_suryakant.zip` or provide a valid local path to begin.")
    st.stop()

if not zip_path.exists():
    st.error(f"ZIP file not found: {zip_path}")
    st.stop()

left, right = st.columns([1.2, 0.8])

with left:
    goal = st.text_area(
        "Research goal",
        value="Summarize the main structural biology findings across this PDF corpus.",
        height=120,
    )

with right:
    st.metric("Data file", zip_path.name)
    st.metric("Max docs", max_docs)
    st.metric("Max pages", max_pages)

with st.spinner("Building PDF corpus..."):
    try:
        corpus, corpus_summary = cached_corpus(str(zip_path), max_docs, max_pages)
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.subheader("Corpus")
st.write(corpus_summary)

preview_query = st.text_input("Preview retrieval", value=goal)
matches = lexical_retrieve(corpus, preview_query, top_k=4)

if matches:
    with st.expander("Retrieved excerpts", expanded=False):
        st.text(format_context(matches, max_chars=7000))

run_clicked = st.button("Run planner, executor, verifier", type="primary")

if run_clicked:
    with st.spinner("Running agents..."):
        try:
            final_state = run_async(
                run_multi_agent(
                    goal=goal,
                    max_iterations=iterations,
                    provider=provider,
                    data_zip=zip_path,
                    max_docs=max_docs,
                    max_pages=max_pages,
                )
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.subheader("Final Answer")
    st.write(f"Approved: `{final_state.get('approved')}`")
    st.write(f"Score: `{final_state.get('score', 'n/a')}`")
    if final_state.get("critique"):
        st.write(final_state["critique"])

    for idx, result in enumerate(final_state.get("results", []), start=1):
        with st.expander(f"Result {idx}: {result.get('task', 'task')}", expanded=True):
            st.write(result.get("output", ""))
            sources = result.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))

    metric_data = metrics_frame(final_state)
    st.subheader("Run Metrics")
    st.dataframe(metric_data, use_container_width=True)
    chart = px.bar(metric_data, x="agent", y="latency_s", color="agent", title="Agent latency")
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("Raw state"):
        st.json(final_state)
