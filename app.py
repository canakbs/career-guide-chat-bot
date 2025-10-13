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

# --- 1. LOGGING ---
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
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("✅ Embedding modeli yüklendi.")
        except Exception as e:
            logging.error(f"❌ Model yüklenemedi: {e}")
            self.model = None

    def embed(self, text: str):
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
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model_name
        self.headers = {"Content-Type": "application/json"}

    def generate(self, question: str, context: str):
        if not self.api_key:
            return "Mock yanıt: Yapay zeka anahtarı bulunamadı."

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        prompt = f"""
You are a helpful AI assistant specialized in career guidance.
Use the following context for your answer if it's relevant.

CONTEXT:
{context}

QUESTION:
{question}
"""

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }

        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No response.")
            )

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Gemini HTTP hatası: {http_err} - {r.text}")
            # Hata mesajı daha açıklayıcı olsun:
            return f"Gemini API HTTP hatası: {r.status_code} — {r.text}"
        except Exception as e:
            logging.error(f"Gemini API hatası: {e}")
            return "Gemini API erişilemedi."


# --- 6. CAREER ASSISTANT ---
class CareerAssistant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = EmbeddingService()
        self.pinecone_index = self._setup_pinecone()
        self.gemini = GeminiClient(cfg.GEMINI_API_KEY)
        self._load_dataset_to_pinecone()

    def _setup_pinecone(self):
        if not self.cfg.PINECONE_API_KEY:
            logging.warning("Pinecone API key eksik.")
            return None
        try:
            pc = Pinecone(api_key=self.cfg.PINECONE_API_KEY)
            if self.cfg.PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
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
        if not self.pinecone_index:
            return

        stats = self.pinecone_index.describe_index_stats()
        if stats.get("total_vector_count", 0) > 0:
            logging.info("Index zaten dolu, veri yükleme atlandı.")
            return      
            
        try:
            ds = load_dataset("Pradeep016/career-guidance-qa-dataset", split="train")
            examples = random.sample(list(ds), min(300, len(ds)))

            vectors = []
            for i, row in enumerate(examples):
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                if not q:
                    continue
                emb = self.embedder.embed(q)
                vectors.append({
                    "id": f"vec_{i}",
                    "values": emb,
                    "metadata": {"question": q[:300], "answer": a[:300]},
                })

            for i in range(0, len(vectors), self.cfg.BATCH_SIZE):
                self.pinecone_index.upsert(vectors=vectors[i:i + self.cfg.BATCH_SIZE])
            logging.info(f"{len(vectors)} vektör Pinecone’a yüklendi.")
        except Exception as e:
            logging.error(f"Veri yükleme hatası: {e}")



    def get_answer(self, question: str):
        if not question:
            return "Lütfen bir soru girin."
        
        query_emb = self.embedder.embed(question)
        
        try:
            # Gelişmiş sorgu parametreleri
            res = self.pinecone_index.query(
                vector=query_emb, 
                top_k=5, 
                include_metadata=True,
                include_values=False,  # Gereksiz veri transferini azalt
                filter=None,  # İsterseniz metadata filtreleri ekleyin
            )
            
            # Skorları kontrol et
            for i, match in enumerate(res["matches"]):
                logging.info(f"Match {i+1}: Score={match['score']:.4f}, ID={match['id']}")
            
            # Skor eşiği uygula
            relevant_matches = [m for m in res["matches"] if m["score"] > 0.3]  # Eşik değeri ayarla
            
            if not relevant_matches:
                context = "No relevant context found."
                logging.warning("Yüksek skorlu eşleşme bulunamadı")
            else:
                context = "\n".join([
                    f"Q: {m['metadata'].get('question', '')}\nA: {m['metadata'].get('answer', '')}" 
                    for m in relevant_matches
                ])
                
        except Exception as e:
            logging.error(f"Sorgu hatası: {e}")
            context = "No context found."

        return self.gemini.generate(question, context)

    def health(self):
        return {
            "status": "healthy",
            "pinecone_ready": bool(self.pinecone_index),
            "gemini_ready": bool(self.cfg.GEMINI_API_KEY),
            "embedding_model": "all-MiniLM-L6-v2",
        }




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
    return jsonify({"answer": assistant.get_answer(data["question"].strip())})

@app.route("/health")
def health():
    return jsonify(assistant.health())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
