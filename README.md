# Evaluation Policy Q&A Assistant

A Streamlit RAG (Retrieval-Augmented Generation) chatbot for querying Canada's *Policy on Results* (Treasury Board, 2016) in English and French.

## How it works

1. Source PDFs in `data/` are indexed into a FAISS vector store via `ingest.py`.
2. At runtime, `app/main.py` loads the FAISS index and uses [LangChain](https://github.com/langchain-ai/langchain) + [Groq](https://groq.com/) to answer questions.

### Self-healing startup

The app automatically builds the FAISS index on first launch if `data/faiss_index/` is absent.  
This means **no manual `python ingest.py` step is required** when deploying to Streamlit Cloud or any fresh environment that only has the repository files.

A spinner is shown while the index is being created (~1 minute on first run).  
Once built, the index is cached and subsequent startups load it instantly.

> **Note:** `data/faiss_index/` is listed in `.gitignore` because it is a generated artifact.  
> The source PDFs committed to `data/` are sufficient to reconstruct it.

## Local setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key
echo "GROQ_API_KEY=<your-key>" > .env

# 3. (Optional) Pre-build the index manually
python ingest.py

# 4. Run the app
streamlit run app/main.py
```

If you skip step 3, the app will build the index automatically on first launch.

## Deployment (Streamlit Cloud)

1. Fork / push this repository to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io), pointing to `app/main.py`.
3. Add `GROQ_API_KEY` as a secret in the app settings.
4. Deploy — the index is built automatically on the first cold start.

## Environment

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for the Groq LLM (required) |
