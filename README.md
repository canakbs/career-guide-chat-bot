<<<<<<< HEAD
# career-guide-chat-bot
This chat bot gives you career advices based on Gemini API and Pinecone vector database.
=======
# Career Guide Chatbot

Small Flask app that demos a career-guidance Q&A assistant built from a public dataset.

Features
- Loads `Pradeep016/career-guidance-qa-dataset` (via `datasets`).
- Local deterministic embedding fallback and lexical reranking so the app runs without external services.
- Optional Pinecone integration for vector storage & search (enabled via env vars).
- Optional Gemini (or other LLM) integration for answer generation (enabled via `GEMINI_API_KEY`).

Setup

1. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill any keys you want to use (Pinecone, Gemini):

   ```powershell
   copy .env.example .env
   notepad .env
   ```

Run

```powershell
python app.py
```

Visit http://127.0.0.1:5000 in your browser.

Notes
- If `PINECONE_API_KEY` is set and `pinecone-client` imports successfully, the app will try to initialize and upsert the dataset into the configured index on first run.
- If `GEMINI_API_KEY` is not set you'll receive a mocked response. Replace `GEMINI_API_URL` with your provider endpoint or adapt `app.py` to use an official SDK.

Security
- Keep API keys out of source control. Use environment variables or a secrets manager in production.

Next steps / improvements
- Add proper async request handling and streaming completions from the LLM.
- Replace the simple lexical reranker with embedding + cosine similarity for better results.
- Add unit tests and CI.
>>>>>>> 9abc55a (Initial commit)
