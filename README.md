# Multi-Agent LangGraph Prototype

Context-aware RAG chatbot plus planner -> executor -> verifier prototype built with LangChain prompt/document primitives, LangGraph orchestration, LlamaIndex retrieval, and provider adapters for Grok, Gemini, OpenAI/ChatGPT-compatible chat completions, or Hugging Face Inference Providers. The main data source is whatever supported document or archive the user uploads in the Streamlit app.

## What is included

- `multi_agent.py` - runnable backend with PDF/ZIP ingestion, LangChain `Document` objects and chat prompt templates, LangGraph agent workflow, section-aware and table-aware retrieval, LlamaIndex BM25 retrieval, numeric boosting, chatbot answering, decorators, async coroutines, exception handling, context managers, generators, concurrency, parallel chart generation, pickling, file handling, regular expressions, pandas, NumPy, Matplotlib, Seaborn, and Plotly.
- `requirements.txt` - installable Python dependencies.
- `.env.example` - provider keys and runtime settings.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python multi_agent.py --goal "Research and summarize the top 3 trends in generative AI for 2025" --iters 3
```

Set `LLM_PROVIDER` to `local`, `grok`, `gemini`, `openai`, or `huggingface`. `local` is an API-key-free evidence mode for testing retrieval. Grok, OpenAI, and Hugging Face use OpenAI-style chat completions. Gemini uses `google-generativeai`.

## Streamlit Deployment

The Streamlit entrypoint is `streamlit_app.py`. It works as a chatbot: upload ZIP, PDF, image, text/Markdown, CSV/TSV, Excel, or JSON files, ask questions, and inspect the retrieved evidence with page, section, table, and source labels.

By default, answers are restricted to uploaded documents only. The sidebar includes `Allow open-source/general knowledge`; external knowledge may be used only when that toggle is enabled.

Uploads can be answered in two scopes:

- `Aggregate all uploads` combines all uploaded files into one corpus.
- `Separate selected file` answers against only the selected uploaded file/group.

The chat sidebar includes two response modes:

- `RAG chatbot` answers directly from retrieved context.
- `Planner -> Executor -> Verifier` sends each user query through the LangGraph agent workflow over the uploaded ZIP/PDF corpus, then returns the verified response.

Local run:

```bash
pip install -r requirements.txt
python verify_deploy.py
python langsmith_eval.py
streamlit run streamlit_app.py
```

On Streamlit Cloud:

- Set the main file path to `streamlit_app.py`.
- Add API keys in Streamlit secrets using `.streamlit/secrets.toml.example` as the template.
- Upload supported documents in the app sidebar after deployment, or set `DATA_ZIP`/`DATA_PATH` only when deploying with a file already available in the runtime.

For Hugging Face, set `LLM_PROVIDER="huggingface"`, `HF_TOKEN`, and optionally `HF_MODEL` in Streamlit secrets.

The RAG pipeline does not split every 1000 characters. It preserves page boundaries, detects section headings, extracts tables with `pdfplumber`/spreadsheet readers when available, chunks by section and sentence windows, retrieves with LlamaIndex BM25, boosts exact numerical matches, and retrieves adjacent chunks for connected context.

## Monitoring And Feedback

The app logs each chatbot answer to `artifacts/monitoring/rag_events.jsonl` and feedback to `artifacts/monitoring/feedback.jsonl`.

- Weights & Biases is optional. Set `WANDB_API_KEY`, `WANDB_PROJECT`, and optionally `WANDB_ENTITY` in Streamlit secrets.
- Evidently is optional. Use the sidebar button `Write Evidently report` to create an HTML monitoring report under `artifacts/monitoring/evidently_report.html`.
- LangSmith is optional. Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` to log chatbot runs, feedback examples, and evaluation datasets.
- The feedback loop stores helpful / needs-correction ratings plus comments for later prompt, retrieval, or corpus tuning.

Run local LangSmith-ready evals with:

```bash
python langsmith_eval.py
```

Without `LANGSMITH_API_KEY`, the script still writes `artifacts/langsmith_eval_results.json`. With the key, it also syncs eval examples to LangSmith.

To inspect uploaded/local data without calling an LLM:

```bash
python multi_agent.py --inspect-data --data-zip path/to/your-documents.zip
```

## Outputs

Each run writes JSON summaries and visual artifacts into `ARTIFACT_DIR`:

- `final_state.json`
- `metrics.csv`
- `agent_cache.pkl`
- `metrics_matplotlib.png`
- `metrics_seaborn.png`
- `metrics_plotly.html`
