
# Career Guide Chatbot

This repository contains a small Flask-based RAG (retrieval-augmented generation)
demo that answers career-related questions using a public Q/A dataset, a
vector database (Pinecone) for retrieval, and an LLM (Gemini) for final
generation. The project is designed to be runnable locally for development and
demo purposes.

---

## 1) Project Purpose

The goal of this project is to demonstrate how to build a lightweight career
assistant using a retrieval pipeline. The assistant retrieves relevant Q/A
snippets from a curated dataset and conditions an LLM to produce concise,
actionable answers.

## 2) Dataset

- Source: `Pradeep016/career-guidance-qa-dataset` (loaded via the
   `datasets` library).
- Content: role-specific questions and answers about career paths.
- Note: The dataset itself is not included in this repo; the code loads it at
   runtime via the `datasets` package. You do not need to store the dataset in
   the repository.

## 3) Methods and Architecture

- Embeddings: The code uses a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`) by default for embeddings. The implementation supports
   optional cloud embeddings (Gemini or OpenAI) if configured via `.env`.
- Vector store: Pinecone is used for indexing and retrieval. If Pinecone is
   not configured, the app falls back to a simple lexical reranker.
- Generation model: Gemini (or compatible LLM) is used to generate the final
   answer. If no valid Gemini URL is configured, the app uses a deterministic
   local synthesizer to return useful guidance without making network calls.
- RAG pipeline: embed -> index/query -> assemble context -> LLM generate.

## 4) Results Summary

- This is a demo repository. The code is designed for exploration rather than
   high-stakes production use. When configured with real embedding and LLM
   providers, the system returns more fluent and relevant answers. Locally,
   the fallback synthesizer produces structured and helpful responses for demo
   purposes.

## 5) How to Run (Development)

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy and edit the environment file:

```powershell
copy .env.example .env
notepad .env
```

Fill the file with your API keys if available. Minimal keys:
- `PINECONE_API_KEY` (optional, required to persist vectors to Pinecone)
- `GEMINI_API_KEY` (optional; project includes a local synthesizer fallback)

4. Run the app:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## 6) Project Structure & Files

- `app.py` - Main Flask application and orchestration logic.
- `templates/` - UI templates (index.html).
- `static/` - Frontend assets (CSS, JS).
- `requirements.txt` - Python package requirements.
- `.env` / `.env.example` - Environment variables for API keys.

## 7) Technologies Used

- Web: Flask, Flask-Cors
- Embeddings: sentence-transformers (local by default)
- Vector DB: Pinecone (optional)
- LLM Generation: Gemini API 
- Data: Hugging Face `datasets` to load `Pradeep016/career-guidance-qa-dataset`

## 8) RAG Pipeline Details

1. User submits question via the web UI.
2. Question is embedded using the configured embedding method.
3. Top-k vectors are retrieved from Pinecone (or locally via lexical
    reranking).
4. Retrieved Q/A snippets are assembled into a context string.
5. The context and user question are sent to an LLM for generation (Gemini),
    or to a local synthesizer if the LLM URL isn't configured.

## 9) Usage Guide (UI)

- Type your career question in the text box and press Enter or click Send.
- The interface shows a short 'thinking' indicator while the backend retrieves
   context and (optionally) calls the LLM.
- Results appear as a structured answer in the chat area.

## 10) Notes on Deployment

- This repo does not include an automated deployment. For production
   deployment, you can containerize the app (Docker), configure secrets via
   environment variables on the host, and deploy to a platform (Heroku,
   Render, GCP Cloud Run, etc.). The README's 'Run' section above covers local
   development.

## 11) Screenshots / Visuals

- The UI is intentionally minimal. See `templates/index.html` and
   `static/style.css` for the frontend implementation.

---



