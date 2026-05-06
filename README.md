# Document RAG

A small Streamlit app for asking questions over uploaded documents.

## What It Does

- Upload one or more files: ZIP, PDF, TXT, Markdown, JSON, CSV/TSV, Excel, or images.
- Ask a question.
- Retrieve relevant chunks with TF-IDF and numeric/table boosts.
- Answer from uploaded documents by default.
- Optionally allow general/open-source knowledge.
- Optionally show a simple Planner -> Executor -> Verifier conversation.
- Download answers as Markdown, JSON, CSV, HTML, or TXT.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Providers

Default provider is `local`, which needs no API key and returns evidence from the uploaded documents.

Optional OpenAI-compatible providers:

- `openai`: `OPENAI_API_KEY`
- `grok`: `GROK_API_KEY`
- `huggingface`: `HF_TOKEN`
- `openrouter`: `OPENROUTER_API_KEY`

Set provider keys in Streamlit secrets or environment variables.

## Files

- `streamlit_app.py`: UI
- `multi_agent.py`: compact RAG core
- `requirements.txt`: deployment dependencies
