# 📘 RAG Backend — FastAPI + FAISS + BM25 + Re-Ranking + Gemini  

A production-ready Retrieval-Augmented Generation (RAG) backend that supports:  

- PDF upload  
- Text chunking  
- Embeddings using **E5-base-v2**  
- Semantic search (FAISS)  
- Keyword search (BM25)  
- Cross-encoder re-ranking  
- Context optimization  
- Gemini 2.5 Pro streaming responses  
- Docker deployment  
- Railway free hosting  

---

## 🚀 Features

### 🔍 Hybrid Retrieval  
- **Semantic Search (FAISS + E5-base-v2)**  
- **Keyword Search (BM25)**  
- **Re-ranking using CrossEncoder (MS Marco MiniLM / TinyBERT)**  

### 🧠 LLM Integration  
- Uses **Gemini 2.5 Pro**  
- Fully supports **streaming responses**  
- Optimized context to reduce hallucination  

### 📄 PDF Support  
- Automatic chunking  
- Passage embeddings  
- Stored in FAISS + BM25 indexes  

### 🐳 Docker Ready  
- Fully containerized backend  
- Deployable to Railway, Render, or Fly.io  

---

## 📁 Project Structure

```
rag-backend/
│
├── app.py                 # FastAPI main application
├── Dockerfile             # Docker build file
├── requirements.txt       # Python dependencies
├── core/
│   ├── embeddings.py      # E5-base embeddings (query + passage)
│   ├── vector_store.py    # FAISS vector DB
│   ├── bm25_store.py      # BM25 keyword index
│   ├── reranker.py        # Cross-encoder re-ranking
│   ├── rag.py             # Full RAG pipeline
│
├── routers/
│   ├── upload.py          # PDF upload + chunking
│   ├── chat.py            # Chat endpoint (streaming)
│
└── ...
```

---

## ⚙️ Installation (Local Development)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pranil22/rag-backend.git
cd rag-backend
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` file

```
GEMINI_API_KEY=your_api_key_here
ENV=development
```

### 5️⃣ Run FastAPI locally

```bash
uvicorn app:app --reload --port 8000
```

Open API docs:

👉 http://localhost:8000/docs

---

## 🐳 Running with Docker

### 1️⃣ Build the image

```bash
docker build -t rag-backend .
```

### 2️⃣ Run the container with `.env`

```bash
docker run --env-file .env -p 8000:8000 rag-backend
```

Backend is now live at:

👉 http://localhost:8000  
👉 http://localhost:8000/docs

---

## 🚀 Deploy on Railway (Free)

### 1️⃣ Push code to GitHub

```bash
git add .
git commit -m "deploy backend"
git push origin main
```

### 2️⃣ Go to Railway → **New Project → Deploy from GitHub**

Railway automatically detects the `Dockerfile`.

### 3️⃣ Add environment variables

```
GEMINI_API_KEY=your_key
ENV=production
```

### 4️⃣ After deployment, Railway gives a URL:

```
https://<project>.up.railway.app
```

### 5️⃣ Use this URL in your frontend

Example (React Vite `.env`):

```
VITE_API_BASE=https://<project>.up.railway.app
```

---

## 🔌 API Endpoints

### 📄 `/upload` — Upload PDF  
- Extracts text  
- Splits into chunks  
- Embeds each chunk  
- Saves into FAISS + BM25  
- Returns document ID  

### 💬 `/chat` — RAG Chat (Streaming)
Steps:  
1. Query embedding (E5-base-v2)  
2. FAISS semantic search  
3. BM25 keyword search  
4. Merge candidates  
5. Cross-encoder re-ranking  
6. Build optimized context  
7. Gemini 2.5 Pro streaming answer  

---

## 🔒 Environment Variables

| Variable | Description |
|---------|-------------|
| `GEMINI_API_KEY` | Gemini API key for LLM |
| `ENV` | development / production |

---

## 📦 Requirements

```
fastapi
uvicorn[standard]
python-multipart
numpy
faiss-cpu
sentence-transformers
rank-bm25
google-generativeai
langchain
langchain-community
```

---

## 🛠 Troubleshooting

### ❌ FAISS dimension mismatch  
E5-base-v2 → **768 dimensions**  
Ensure your FAISS index is initialized with:

```python
VectorStore(dim=768)
```

### ❌ Streaming not working  
Railway supports streaming — ensure frontend uses ReadableStream.

### ❌ FAISS index resets on restart  
Enable Railway **Volumes** and store:  
- `index.faiss`  
- `chunks.json`

---

## 📜 License

MIT License — free to use, modify, distribute.

---

## 🤝 Contributing

PRs and issues are welcome!

---

## 🙌 Credits

Built using:  
FastAPI • FAISS • BM25 • E5 Embeddings • CrossEncoders • Gemini API • Docker • Railway  
