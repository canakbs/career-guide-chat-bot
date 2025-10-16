import os
import logging
import random
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Project: Career Guide Chatbot
#
# Purpose:
# - A small RAG-style (retrieval-augmented generation) demo that answers
#   career-related questions using a public Q/A dataset and an LLM (Gemini).
# - Retrieval uses Pinecone as a vector store; embeddings are produced by a
#   sentence-transformers model (local) or can be replaced by a cloud provider.
#
# Notes:
# - This file contains only the application logic. Comments explain the
#   architecture, environment variables, and how components interact.
# - For deployment, set the necessary API keys in the `.env` file. If a real
#   LLM endpoint is not configured, the code falls back to local synthesizer
#   and retrieval so the app remains usable for demos.
#
# Key components:
# - EmbeddingService: local sentence-transformers model for embeddings.
# - GeminiClient: wrapper to call the Gemini generation API (if configured).
# - Pinecone integration: optional vector database for retrieval.
# - CareerAssistant: orchestrates retrieval and generation for queries.
#
# See README.md for detailed instructions, dataset info, and architecture.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 2. ENV ---
load_dotenv()

# --- 3. CONFIG ---
class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "career-guide")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    EMBEDDING_DIM = 384
    MAX_TEXT_LEN = 300
    BATCH_SIZE = 100


# --- 4. EMBEDDING SERVICE ---
class EmbeddingService:
    def __init__(self):
        # Load a small, fast sentence-transformers model for local embeddings.
        # This choice is suitable for development and demo purposes. Replace
        # the model name or use a cloud embedding API for production-quality
        # vectors.
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("✅ Embedding modeli yüklendi.")
        except Exception as e:
            logging.error(f"❌ Model yüklenemedi: {e}")
            self.model = None

    def embed(self, text: str):
        # Create an embedding vector for the provided text. This method
        # returns a Python list of floats suitable for indexing into a
        # vector database (e.g., Pinecone). The function raises on invalid
        # inputs or if the model failed to load.
        if not text:
            raise ValueError("Boş metin için embedding oluşturulamaz.")
        if not self.model:
            raise RuntimeError("Embedding modeli yüklenmedi, model None durumda.")
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logging.error(f"Embedding oluşturulamadı: {e}")
            raise e



# --- 5. GEMINI CLIENT ---
class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"  # ✅ v1beta sürümü korundu

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):  # ✅ Model güncellendi
        self.api_key = api_key
        self.model = model_name
        self.headers = {"Content-Type": "application/json"}

    def generate(self, question: str, context: str, history: list = None):
        if not self.api_key:
            return "Mock response: LLM API key not configured."

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Konuşma geçmişini Gemini formatına çevir
        gemini_history = []
        if history:
            for turn in history:
                gemini_history.append({"role": "user", "parts": [{"text": turn["question"]}]})
                gemini_history.append({"role": "model", "parts": [{"text": turn["answer"]}]})
        
        
        # Sistem talimatı
        system_instruction = f"""
You are a helpful AI assistant specialized in career guidance.
Use the following external context for your answer if it's relevant to the user's LATEST question.
If the context seems irrelevant to the user's latest question, rely on the conversation history to provide a helpful response.

CONTEXT:
{context}
"""

        # Mevcut soruyu geçmişe ekle
        current_question_part = {"role": "user", "parts": [{"text": question}]}

        # Payload oluştur
        payload = {
            "contents": gemini_history + [current_question_part],
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            }
        }

        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            # API yanıtını güvenli şekilde çöz
            if "candidates" in data and data["candidates"]:
                content = data["candidates"][0].get("content", {})
                if "parts" in content and content["parts"]:
                    return content["parts"][0].get("text", "No response text found.")
            return "No valid response from Gemini."

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Gemini HTTP hatası: {http_err} - {r.text}")
            return f"Gemini API HTTP hatası: {r.status_code} — {r.text}"
        except Exception as e:
            logging.error(f"Gemini API hatası: {e}")
            return "Gemini API not reachable."
        

# --- 6. CAREER ASSISTANT ---
class CareerAssistant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = EmbeddingService()
        self.pinecone_index = self._setup_pinecone()
        self.gemini = GeminiClient(cfg.GEMINI_API_KEY)
        # self._load_dataset_to_pinecone() # Bu satırı deployda kapatmak iyi bir fikir olabilir

    def _setup_pinecone(self):
        # ... (Bu fonksiyon aynı kalıyor) ...
        if not self.cfg.PINECONE_API_KEY: return None
        try:
            pc = Pinecone(api_key=self.cfg.PINECONE_API_KEY)
            if self.cfg.PINECONE_INDEX not in pc.list_indexes().names():
                pc.create_index(
                    name=self.cfg.PINECONE_INDEX,
                    dimension=self.cfg.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.cfg.PINECONE_ENV),
                )
            return pc.Index(self.cfg.PINECONE_INDEX)
        except Exception as e:
            logging.error(f"Pinecone başlatılamadı: {e}")
            return None

    def _load_dataset_to_pinecone(self):
        # ... (Bu fonksiyon aynı kalıyor) ...
        if not self.pinecone_index: return
        if self.pinecone_index.describe_index_stats().get("total_vector_count", 0) > 0:
            logging.info("Index zaten dolu, veri yükleme atlandı.")
            return
        try:
            ds = load_dataset("Pradeep016/career-guidance-qa-dataset", split="train")
            examples = random.sample(list(ds), min(300, len(ds)))
            vectors = []
            for i, row in enumerate(examples):
                q = row.get("question", "").strip()
                if not q: continue
                vectors.append({
                    "id": f"vec_{i}", "values": self.embedder.embed(q),
                    "metadata": {"question": q[:300], "answer": row.get("answer", "")[:300]}
                })
            for i in range(0, len(vectors), self.cfg.BATCH_SIZE):
                self.pinecone_index.upsert(vectors=vectors[i:i + self.cfg.BATCH_SIZE])
            logging.info(f"{len(vectors)} vektör Pinecone’a yüklendi.")
        except Exception as e:
            logging.error(f"Veri yükleme hatası: {e}")

    def get_answer(self, question: str, history: list = None):
        if not question:
            return "Please enter a question."
        
        context = "No relevant context found."
        max_score = 0.0  # 🔹 Skor takibi için eklendi
        
        try:
            if self.pinecone_index:
                query_emb = self.embedder.embed(question)
                res = self.pinecone_index.query(vector=query_emb, top_k=3, include_metadata=True)
                relevant_matches = [m for m in res["matches"] if m["score"] > 0.3]

                # 🔹 Maksimum skor hesapla
                if res.get("matches"):
                    max_score = max([m["score"] for m in res["matches"]])

                if relevant_matches:
                    context = "\n".join([
                        f"Q: {m['metadata'].get('question', '')}\nA: {m['metadata'].get('answer', '')}"
                        for m in relevant_matches
                    ])
        except Exception as e:
            logging.error(f"Sorgu hatası: {e}")
            context = "Error during context retrieval."

        # 🔹 Skor düşükse bazı meslekleri yasakla
        restricted_jobs = ["UX/UI Designer", "Game Developer", "Mobile App Developer"]
        job_restriction_instruction = ""

        if max_score < 0.5:
            job_restriction_instruction = (
                f"\nIMPORTANT: Do NOT mention or base your answer on these jobs: {', '.join(restricted_jobs)}.\n"
                "Focus on other careers instead."
            )
        else:
            job_restriction_instruction = (
                f"\nYou MAY mention these jobs if relevant: {', '.join(restricted_jobs)}.\n"
            )

        # 🔹 Bu kısıtlamayı Gemini'ye ekle
        context += job_restriction_instruction

        return self.gemini.generate(question, context, history)

    def health(self):
        return { "status": "healthy", "pinecone_ready": bool(self.pinecone_index), "gemini_ready": bool(self.cfg.GEMINI_API_KEY) }

# --- 7. FLASK APP ---
app = Flask(__name__)
CORS(app)

cfg = Config()
assistant = CareerAssistant(cfg)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"answer": "Geçersiz istek."}), 400
    
    question = data["question"].strip()
    # GÜNCELLEME: Frontend'den gelen 'history' listesini al
    history = data.get("history", [])
    
    # GÜNCELLEME: CareerAssistant'a hem soruyu hem de geçmişi gönder
    answer = assistant.get_answer(question, history)
    
    return jsonify({"answer": answer})

@app.route("/health")
def health():
    return jsonify(assistant.health())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
