"""Concise Streamlit RAG chatbot."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from multi_agent import (
    answer_rag_chat,
    answer_with_agent_pipeline_from_corpus,
    build_corpus_from_paths,
    format_context,
    retrieve,
)


st.set_page_config(page_title="Document RAG", layout="wide")


def save_upload(file: Any) -> Path:
    root = Path(tempfile.gettempdir()) / "simple_rag_uploads"
    root.mkdir(exist_ok=True)
    path = root / file.name
    path.write_bytes(file.getbuffer())
    return path


def export(answer: str, sources: List[Dict[str, Any]], fmt: str) -> tuple[bytes, str, str]:
    if fmt == "JSON":
        return json.dumps({"answer": answer, "sources": sources}, indent=2).encode(), "answer.json", "application/json"
    if fmt == "HTML":
        return f"<pre>{answer}</pre>".encode(), "answer.html", "text/html"
    if fmt == "CSV":
        rows = ["source,page,section,score"] + [f"{s['source']},{s['page']},{s['section']},{s.get('score',0)}" for s in sources]
        return ("\n".join(rows)).encode(), "answer.csv", "text/csv"
    if fmt == "TXT":
        return answer.encode(), "answer.txt", "text/plain"
    return (answer + "\n\n" + "\n".join(f"- {s['source']} p.{s['page']} {s['section']}" for s in sources)).encode(), "answer.md", "text/markdown"


def show_agent_turns(turns: List[Dict[str, Any]]) -> None:
    if not turns:
        return
    with st.expander("Agent conversation", expanded=True):
        for t in turns:
            st.markdown(f"**{t.get('agent', 'agent').title()}**: {t.get('message', '')}")
            if t.get("payload"):
                st.json(t["payload"])


for key in ("OPENAI_API_KEY", "GROK_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY"):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = str(st.secrets[key])


st.title("Document RAG")
st.caption("Upload documents, ask questions, answer from the uploads. Simple on purpose.")

with st.sidebar:
    files = st.file_uploader(
        "Upload files",
        type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    local_path = st.text_input("Or local path", "")
    provider = st.selectbox("Provider", ["local", "openai", "grok", "gemini", "huggingface", "openrouter"])
    mode = st.radio("Mode", ["RAG", "Planner → Executor → Verifier"])
    external = st.toggle("Allow open-source/general knowledge", value=False)
    fmt = st.selectbox("Download format", ["Markdown", "JSON", "CSV", "HTML", "TXT"])
    top_k = st.slider("Evidence chunks", 3, 15, 8)

paths = [save_upload(f) for f in files] if files else ([Path(local_path)] if local_path else [])
if not paths:
    st.info("Upload one or more files to begin.")
    st.stop()

missing = [p for p in paths if not p.exists()]
if missing:
    st.error(f"Missing file: {missing[0]}")
    st.stop()

with st.spinner("Indexing..."):
    corpus, summary = build_corpus_from_paths(paths)

st.success(summary)

query = st.text_area("Ask a question", height=100, placeholder="Ask only what should be answered from the uploaded documents.")
ask = st.button("Ask", type="primary")

preview = retrieve(corpus, query or "summary", top_k)
with st.expander("Evidence preview"):
    st.text(format_context(preview))

if ask and query.strip():
    os.environ["LLM_PROVIDER"] = provider
    with st.spinner("Answering..."):
        if mode.startswith("Planner"):
            result = st.session_state.get("_last") or None
            result = __import__("asyncio").run(answer_with_agent_pipeline_from_corpus(query, corpus, summary, provider))
        else:
            result = __import__("asyncio").run(answer_rag_chat(query, corpus, provider=provider, top_k=top_k, allow_external_knowledge=external))
    st.markdown(result["answer"])
    show_agent_turns(result.get("conversation", []))
    with st.expander("Sources", expanded=True):
        for s in result["sources"]:
            st.markdown(f"**{s['source']}** p.{s['page']} · {s['section']} · score {s.get('score', 0):.3f}")
            st.text(s["text"][:1200])
    data, name, mime = export(result["answer"], result["sources"], fmt)
    st.download_button("Download answer", data, name, mime)
