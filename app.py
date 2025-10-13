import os
import random
from click import prompt
import requests
import json
from markdown import markdown
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from flask_cors import CORS


# Load .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# ------------------ CONFIG ------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "career-guide")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIM = 1024
MAX_TEXT_LEN = 300
BATCH_SIZE = 100

print(f"🔍 Environment check:")
print(f"🔍 PINECONE_API_KEY: {'✅' if PINECONE_API_KEY else '❌'}")
print(f"🔍 GEMINI_API_KEY: {'✅' if GEMINI_API_KEY else '❌'}")

# ------------------ PINECONE INIT ------------------
pc = None
index = None

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
            )
        index = pc.Index(PINECONE_INDEX)
        print(f"✅ Pinecone index ready: {PINECONE_INDEX}")
except Exception as e:
    print(f"⚠️ Could not initialize Pinecone: {e}")

# ------------------ DATASET ------------------
dataset = load_dataset("Pradeep016/career-guidance-qa-dataset", split="train")

# ------------------ LOCAL EMBEDDING ------------------
def embed_text(text):
    text = text or ""
    values = [float((ord(c) % 100) / 100.0) for c in text[:EMBEDDING_DIM]]
    return values + [0.0] * (EMBEDDING_DIM - len(values))

# ------------------ UPSERT TO PINECONE ------------------
def upsert_to_pinecone():
    if not index:
        print("⚠️ Pinecone index not available, skipping upsert.")
        return
    
    try:
        n_samples = min(1000, len(dataset))
        sampled_indices = random.sample(range(len(dataset)), n_samples)
        
        print(f"🔄 Starting upsert of {n_samples} samples in batches of {BATCH_SIZE}...")
        
        vectors = []
        total_upserted = 0
        
        for i, idx in enumerate(sampled_indices):
            row = dataset[idx]
            q = (row.get("question") or "")[:MAX_TEXT_LEN]
            a = (row.get("answer") or "")[:MAX_TEXT_LEN]
            r = (row.get("role") or "")[:50]
            metadata = {"question": q, "answer": a, "role": r}

            vec = embed_text(q)
            vectors.append({
                "id": f"vec_{i}",
                "values": vec,
                "metadata": metadata
            })
            
            if len(vectors) >= BATCH_SIZE or i == len(sampled_indices) - 1:
                try:
                    index.upsert(vectors=vectors)
                    total_upserted += len(vectors)
                    print(f"✅ Upserted batch of {len(vectors)} vectors. Total: {total_upserted}")
                    vectors = []
                except Exception as batch_error:
                    print(f"⚠️ Batch upsert failed: {batch_error}")
                    vectors = []

        print(f"🎉 Successfully upserted {total_upserted} vectors to Pinecone.")

    except Exception as e:
        print(f"⚠️ Could not upsert to Pinecone: {e}")

# Uygulama başladığında Pinecone'a veri yükle
upsert_to_pinecone()

# ------------------ GEMINI FUNCTION ------------------
def ask_gemini(question, context):
    print(f"🔍 ask_gemini called with question: '{question}'")
    
    if not GEMINI_API_KEY:
        mock_response = (
            "Based on career guidance principles: To become an engineer, you typically need "
            "a bachelor's degree in engineering, strong math and science skills, and relevant "
            "certifications. Gain practical experience through internships and develop problem-solving abilities."
        )
        print(f"🔍 No API key, returning mock response")
        return mock_response
    
    try:
        prompt = f"""
You are a helpful AI assistant specialized in career advice and general knowledge.

Below is additional context information retrieved from a database. 
Use it only as a supporting source, not as the only truth. 
If the context is incomplete or irrelevant, use your own general knowledge to give the best possible answer.

---
📘 CONTEXT (from vector database):
{context if context.strip() else "No extra context provided."}
---

💬 USER QUESTION:
{question}

🎯 INSTRUCTIONS:
- Give a complete, comprehensive answer without being cut off
- Use bullet points or short paragraphs where helpful
- If the context does not contain relevant data, still answer using your general knowledge
- Prefer clarity and practicality over formality
- Make sure to provide a full answer that covers all aspects of the question
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048
            }
        }

        headers = {"Content-Type": "application/json"}

        print(f"🔍 Sending request to Gemini API...")
        response = requests.post(url, json=payload, headers=headers, timeout=45)
        print(f"🔍 Gemini API status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"🔍 Gemini API response received successfully")
            
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    text = parts[0]["text"].strip()
                    print(f"🔍 Gemini response length: {len(text)} characters")
                    
                    if len(text) > 1900:
                        print(f"⚠️ Response might be truncated, length: {len(text)}")
                        if not text.endswith(('.', '!', '?')) and not any(marker in text.lower() for marker in ['finally', 'in conclusion', 'to summarize']):
                            text += "\n\n*Note: This response was automatically shortened due to length limits.*"
                    
                    return text

            print(f"🔍 No valid response format in Gemini API response")
            return "I couldn't generate a proper response from the AI service."
        
        else:
            error_msg = f"Gemini API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            if response.status_code == 400:
                return "There's an issue with the AI service configuration. Please check the API settings."
            elif response.status_code == 403:
                return "API access denied. Please check your Gemini API key."
            elif response.status_code == 404:
                print(f"🔍 Model not found, trying alternative...")
                return ask_gemini_fallback(question, context)
            elif response.status_code == 429:
                return "Too many requests. Please try again later."
            else:
                return "I'm currently unable to connect to the AI service. Please try again later."
            
    except requests.exceptions.Timeout:
        print(f"❌ Gemini API timeout")
        return "The AI service took too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        print(f"❌ Gemini API connection error")
        return "Cannot connect to the AI service. Please check your internet connection."
    except Exception as e:
        print(f"❌ Error calling Gemini API: {str(e)}")
        return f"Technical error: {str(e)}"

def ask_gemini_fallback(question, context):
    """gemini-2.5-flash bulunamazsa alternatif model dene"""
    try:
        fallback_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-1.0-pro"
        ]
        
        for model in fallback_models:
            print(f"🔄 Trying fallback model: {model}")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nProvide a comprehensive answer:"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048
                }
            }
            
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    print(f"✅ Success with fallback model: {model}")
                    return text
        
        return "I apologize, but I couldn't connect to any AI service at the moment."
        
    except Exception as e:
        print(f"❌ Fallback also failed: {e}")
        return "Service temporarily unavailable. Please try again later."


# ------------------ FLASK ROUTES ------------------
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    print(f"🔍 /ask endpoint called")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"answer": "Invalid request format. Please send JSON data."}), 400
            
        question = data.get("question", "").strip()
        print(f"🔍 Received question: '{question}'")
        
        if not question:
            return jsonify({"answer": "Please enter a question."}), 400

        context = ""
        if index:
            print(f"🔍 Querying Pinecone...")
            query_vector = embed_text(question)
            results = index.query(vector=query_vector, top_k=3, include_metadata=True)
            matches = results.get("matches", [])
            print(f"🔍 Found {len(matches)} matches from Pinecone")
            
            if matches:
                context_parts = []
                for i, match in enumerate(matches):
                    answer = match.get("metadata", {}).get("answer", "")
                    if answer:
                        context_parts.append(f"- {answer}")
                context = "\n".join(context_parts)
        
        if not context:
            context = "General career guidance: focus on skills development, gain relevant experience, and seek mentorship."
            print(f"🔍 Using fallback context")

        print(f"🔍 Context length: {len(context)} characters")
        print(f"🔍 Calling ask_gemini...")
        answer = ask_gemini(question, context)
        print(f"🔍 Final answer ready: {answer[:100]}...")
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        error_msg = f"Error in /ask route: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"answer": "Sorry, I encountered an error while processing your request. Please try again."}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "pinecone_ready": index is not None,
        "gemini_ready": bool(GEMINI_API_KEY),
        "gemini_key_exists": bool(GEMINI_API_KEY)
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)