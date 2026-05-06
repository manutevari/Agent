# Multi-Agent LangGraph Prototype

Planner -> executor -> verifier prototype built with LangChain message types, LangGraph orchestration, and provider adapters for Grok, Gemini, or OpenAI/ChatGPT-compatible chat completions. The main data source is `data_suryakant.zip`, a PDF corpus read directly from the zip archive.

## What is included

- `multi_agent.py` - runnable prototype with PDF zip ingestion, corpus retrieval, decorators, async coroutines, exception handling, context managers, generators, concurrency, parallel chart generation, pickling, file handling, regular expressions, pandas, NumPy, Matplotlib, Seaborn, and Plotly.
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

Set `LLM_PROVIDER` to `grok`, `gemini`, or `openai`. Grok and OpenAI use OpenAI-style chat completions. Gemini uses `google-generativeai`.

## Streamlit Deployment

The Streamlit entrypoint is `streamlit_app.py`.

Local run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Streamlit Cloud:

- Set the main file path to `streamlit_app.py`.
- Add API keys in Streamlit secrets using `.streamlit/secrets.toml.example` as the template.
- Upload `data_suryakant.zip` in the app sidebar after deployment, or set `DATA_ZIP` only when deploying with the zip already available in the runtime.

To inspect the main PDF data without calling an LLM:

```bash
python multi_agent.py --inspect-data --data-zip C:/Users/Dell/Downloads/data_suryakant.zip
```

## Outputs

Each run writes JSON summaries and visual artifacts into `ARTIFACT_DIR`:

- `final_state.json`
- `metrics.csv`
- `agent_cache.pkl`
- `metrics_matplotlib.png`
- `metrics_seaborn.png`
- `metrics_plotly.html`
