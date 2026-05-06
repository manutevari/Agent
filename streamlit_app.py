"""Streamlit deployment entrypoint for the zip-backed multi-agent RAG prototype."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from monitoring import log_feedback, log_rag_event, monitoring_summary, write_evidently_report
from multi_agent import answer_rag_chat, answer_with_agent_pipeline_from_corpus, build_corpus, build_corpus_from_paths, format_context, lexical_retrieve, llamaindex_retrieve, run_multi_agent


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
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_BASE_URL",
        "HF_MODEL",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_EVAL_DATASET",
        "LANGSMITH_FEEDBACK_DATASET",
    ):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


def provider_status(provider: str) -> tuple[bool, str]:
    """Return whether a provider has the credentials needed to run."""

    if provider == "local":
        return True, "Local evidence mode does not need an API key."
    if provider == "grok":
        return bool(os.getenv("GROK_API_KEY")), "Set GROK_API_KEY in Streamlit secrets."
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY")), "Set GOOGLE_API_KEY in Streamlit secrets."
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY")), "Set OPENAI_API_KEY in Streamlit secrets."
    if provider == "huggingface":
        return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")), "Set HF_TOKEN in Streamlit secrets."
    return False, "Unknown provider."


def save_uploaded_data(uploaded_file: Any) -> Path:
    """Persist one uploaded file for this Streamlit session."""

    temp_dir = Path(tempfile.gettempdir()) / "streamlit_pdf_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    data_path = temp_dir / uploaded_file.name
    data_path.write_bytes(uploaded_file.getbuffer())
    return data_path


@st.cache_data(show_spinner=False)
def cached_corpus(paths: tuple[str, ...], scope: str, selected_name: str, max_docs: int, max_pages: int) -> tuple[List[Dict[str, Any]], str, Path]:
    """Cache PDF extraction so UI reruns stay responsive."""

    path_objects = [Path(path) for path in paths]
    if scope == "Separate selected file":
        selected = next((path for path in path_objects if path.name == selected_name), path_objects[0])
        corpus, summary = build_corpus(selected, max_docs=max_docs, max_pages=max_pages)
        return corpus, summary, selected
    corpus, summary = build_corpus_from_paths(path_objects, max_docs=max_docs, max_pages=max_pages)
    return corpus, summary, path_objects[0]


def run_async(coro: Any) -> Any:
    """Run an async backend call from Streamlit."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


apply_secret_env()

st.title("Scientific PDF RAG Chatbot")
st.caption("Context-aware retrieval over uploaded PDFs or ZIP archives, preserving sections, tables, pages, and numerical evidence.")

with st.sidebar:
    st.header("Data")
    uploaded_data = st.file_uploader("Upload documents", type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    default_zip = st.text_input("Local file path", value=os.getenv("DATA_ZIP", ""))
    corpus_scope = st.radio("Corpus scope", options=["Aggregate all uploads", "Separate selected file"], index=0)
    max_docs = st.slider("Files or ZIP members to index", min_value=1, max_value=64, value=16)
    max_pages = st.slider("Pages per PDF", min_value=1, max_value=80, value=12)
    top_k = st.slider("Evidence chunks", min_value=3, max_value=15, value=8)
    use_llamaindex = st.toggle("Use LlamaIndex retrieval", value=True)
    allow_external_knowledge = st.toggle("Allow open-source/general knowledge", value=False)

    st.header("Model")
    provider = st.selectbox("Provider", options=["local", "grok", "gemini", "openai", "huggingface"], index=0)
    iterations = st.slider("Verifier iterations", min_value=1, max_value=5, value=3)
    response_mode = st.radio("Response mode", options=["RAG chatbot", "Planner -> Executor -> Verifier"], index=0)
    provider_ready, provider_message = provider_status(provider)
    if provider_ready:
        st.success(provider_message)
    else:
        st.warning(provider_message)

    st.header("Secrets")
    st.caption("Use Streamlit Cloud secrets for API keys. Local `.env` also works.")

    st.header("Monitoring")
    summary = monitoring_summary()
    st.metric("Logged answers", summary["events"])
    st.metric("Feedback", summary["feedback"])
    st.caption(f"Avg latency: {summary['avg_latency_s']}s")
    if os.getenv("LANGSMITH_API_KEY"):
        st.success(f"LangSmith: {os.getenv('LANGSMITH_PROJECT', 'scientific-rag-chatbot')}")
    else:
        st.caption("LangSmith tracing/evals disabled until LANGSMITH_API_KEY is set.")
    if st.button("Write Evidently report"):
        report_path = write_evidently_report()
        st.success(f"Report written: {report_path}")

data_paths: List[Path] = []
if uploaded_data:
    data_paths = [save_uploaded_data(item) for item in uploaded_data]
elif default_zip:
    data_paths = [Path(default_zip)]

if not data_paths:
    st.info("Upload ZIP/PDF/image/text/Excel files or provide a valid local path to begin.")
    st.stop()

missing_paths = [path for path in data_paths if not path.exists()]
if missing_paths:
    st.error(f"Data file not found: {missing_paths[0]}")
    st.stop()

selected_name = data_paths[0].name
if corpus_scope == "Separate selected file" and len(data_paths) > 1:
    selected_name = st.selectbox("Document group", options=[path.name for path in data_paths])

summary_cols = st.columns(4)
summary_cols[0].metric("Uploads", len(data_paths))
summary_cols[1].metric("Max docs", max_docs)
summary_cols[2].metric("Max pages", max_pages)
summary_cols[3].metric("Evidence chunks", top_k)

with st.spinner("Building PDF corpus..."):
    try:
        corpus, corpus_summary, active_data_path = cached_corpus(tuple(str(path) for path in data_paths), corpus_scope, selected_name, max_docs, max_pages)
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.subheader("Corpus")
st.write(corpus_summary)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

st.subheader("Ask A Query")
with st.form("query_form", clear_on_submit=True):
    typed_question = st.text_area(
        "Question",
        placeholder="Ask about structural differences, numerical values, tables, domains, mechanisms, or cross-paper comparisons.",
        height=110,
    )
    submitted_question = st.form_submit_button("Ask", type="primary")

preview_query = st.text_input("Evidence preview", value="glycoprotein structure fusion pH neutralizing antibody")
preview_retrieve = llamaindex_retrieve if use_llamaindex else lexical_retrieve
matches = preview_retrieve(corpus, preview_query, top_k=top_k)

if matches:
    with st.expander("Retrieved excerpts", expanded=False):
        st.text(format_context(matches, max_chars=10000))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Evidence"):
                for source in message["sources"]:
                    label = (
                        f"{source.get('source')} | pp. {source.get('page_start')}-{source.get('page_end')} | "
                        f"{source.get('section')} | {source.get('kind')}"
                    )
                    st.markdown(f"**{label}**")
                    st.text(str(source.get("text", ""))[:1800])

chat_question = st.chat_input("Ask a detailed question about structures, tables, values, domains, mechanisms, or comparisons")
question = typed_question if submitted_question and typed_question.strip() else chat_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        provider_ready, provider_message = provider_status(provider)
        if not provider_ready:
            st.error(provider_message)
            st.stop()
        with st.spinner("Retrieving section-aware evidence and answering..."):
            try:
                if response_mode == "Planner -> Executor -> Verifier":
                    response = run_async(
                        answer_with_agent_pipeline_from_corpus(
                            question=question,
                            corpus=corpus,
                            corpus_summary=corpus_summary,
                            provider=provider,
                            max_iterations=iterations,
                        )
                    )
                else:
                    response = run_async(
                        answer_rag_chat(
                            question=question,
                            corpus=corpus,
                            provider=provider,
                            history=st.session_state.messages,
                            top_k=top_k,
                            use_llamaindex=use_llamaindex,
                            allow_external_knowledge=allow_external_knowledge,
                        )
                    )
            except Exception as exc:
                st.exception(exc)
                st.stop()
        st.markdown(response["answer"])
        with st.expander("Evidence"):
            for source in response["sources"]:
                label = (
                    f"{source.get('source')} | pp. {source.get('page_start')}-{source.get('page_end')} | "
                    f"{source.get('section')} | {source.get('kind')} | score {source.get('score', 0):.1f}"
                )
                st.markdown(f"**{label}**")
                st.text(str(source.get("text", ""))[:1800])
        st.caption(f"{response['provider']} / {response['model']} in {response['latency_s']:.2f}s")
        st.caption(f"LangChain documents: {response.get('langchain_document_count', 0)}")

    retrieval_mode = "agent_planner_executor_verifier" if response_mode == "Planner -> Executor -> Verifier" else ("llamaindex_bm25+numeric" if use_llamaindex else "lexical_numeric")
    log_rag_event(
        question=question,
        answer=response["answer"],
        provider=response["provider"],
        model=response["model"],
        latency_s=response["latency_s"],
        sources=response["sources"],
        retrieval_mode=retrieval_mode,
    )
    st.session_state.last_response = {
        "question": question,
        "answer": response["answer"],
        "provider": response["provider"],
        "retrieval_mode": retrieval_mode,
    }
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})

if st.session_state.last_response:
    with st.expander("Feedback loop", expanded=False):
        st.caption("Your feedback is saved locally and sent to W&B when configured.")
        col_up, col_down = st.columns(2)
        feedback_comment = st.text_area("Feedback note", height=80, placeholder="What was correct, missing, or numerically wrong?")
        if col_up.button("Helpful"):
            log_feedback(rating="up", comment=feedback_comment, **st.session_state.last_response)
            st.success("Feedback saved.")
        if col_down.button("Needs correction"):
            log_feedback(rating="down", comment=feedback_comment, **st.session_state.last_response)
            st.warning("Feedback saved for review.")

with st.expander("Optional planner/executor/verifier run"):
    goal = st.text_area(
        "Research goal",
        value="Summarize the main structural biology findings across this PDF corpus with numerical and source-backed evidence.",
        height=100,
    )
    if st.button("Run multi-agent analysis"):
        provider_ready, provider_message = provider_status(provider)
        if not provider_ready:
            st.error(provider_message)
            st.stop()
        try:
            final_state = run_async(
                run_multi_agent(
                    goal=goal,
                    max_iterations=iterations,
                    provider=provider,
                    data_zip=active_data_path,
                    max_docs=max_docs,
                    max_pages=max_pages,
                )
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

        st.write(f"Approved: `{final_state.get('approved')}`")
        st.write(f"Score: `{final_state.get('score', 'n/a')}`")
        st.write(final_state.get("critique", ""))
        for idx, result in enumerate(final_state.get("results", []), start=1):
            with st.expander(f"Result {idx}: {result.get('task', 'task')}", expanded=True):
                st.write(result.get("output", ""))
                sources = result.get("sources", [])
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
