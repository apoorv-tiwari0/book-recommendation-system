# 📚 Book Recommendation System (NLP + LSTM + Hybrid Retrieval)

## 🔍 Project Overview

This project implements a **large-scale book recommendation engine** using **Natural Language Processing (NLP)**.  
It generates **semantic embeddings** for over **1 million books** using an **LSTM-based model trained with self-supervised triplet loss**, and serves recommendations via a **FastAPI backend**.

The system uses a **hybrid retrieval strategy**:
1. **TF-IDF (lexical retrieval)** for candidate generation  
2. **LSTM semantic embeddings** for fine-grained ranking  

This design ensures both **topical correctness** and **semantic relevance**.

---

## 🧠 High-Level Architecture

```
User Query
   ↓
FastAPI (/recommend)
   ↓
Text Cleaning
   ↓
TF-IDF Candidate Selection (Top ~1000)
   ↓
LSTM Triplet Encoder (Query Embedding)
   ↓
Cosine Similarity (Embedding Re-ranking)
   ↓
Top-K Book Recommendations
```

---

## 🗂️ Project Structure

```
book-recommendation/
│
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── recommender.py       # Core inference logic (DO NOT break)
│   └── schemas.py           # Request/response schemas
│
├── artifacts/               # CRITICAL: production assets
│   ├── models/
│   │   └── lstm_encoder_triplet.keras
│   ├── embeddings/
│   │   ├── book_embeddings.npy
│   │   └── book_metadata.csv
│   ├── tokenizer/
│   │   └── tokenizer.pkl
│   └── tfidf/
│       ├── vectorizer.pkl
│       └── tfidf_matrix.pkl
│
├── src/                     # Training & preprocessing scripts
│   ├── preprocessing/
│   ├── model/
│   └── inference/
│
├── requirements.txt
└── README.md
```

---

## 🏗️ What Has Already Been Done (IMPORTANT)

### ✅ Data
- Used Goodreads dataset (only CSVs with descriptions)
- Cleaned, normalized, and filtered descriptions
- Final dataset size: ~1.04M books

### ✅ Tokenization
- Keras tokenizer with:
  - `max_vocab_size = 50,000`
  - `max_sequence_length = 300`
- Tokenizer is **frozen** and reused everywhere

### ✅ Model
- **LSTM-based encoder**
- Trained using **self-supervised triplet loss**
- Objective:
  - Pull embeddings of the same book together
  - Push embeddings of different books apart
- Output: **128-dimensional semantic embeddings**

### ✅ Embeddings
- All books encoded offline
- Stored as:
  ```
  artifacts/embeddings/book_embeddings.npy
  ```
- Embeddings are **L2-normalized**

### ✅ Retrieval Strategy
- **Hybrid approach** (very important):
  - TF-IDF → candidate filtering
  - LSTM embeddings → semantic ranking
- Pure embedding search was tested and found inferior without lexical grounding

### ✅ Deployment
- Served via **FastAPI**
- Single endpoint:
  ```
  POST /recommend
  ```
- Swagger UI available at:
  ```
  http://127.0.0.1:8000/docs
  ```

---

## 🚀 How to Run the API (Local)

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start the server
```bash
python -m uvicorn app.main:app
```

### 3️⃣ Open Swagger UI
```
http://127.0.0.1:8000/docs
```

### 4️⃣ Example request
```json
{
  "query": "books about artificial intelligence",
  "top_k": 10
}
```

---

## ⚠️ VERY IMPORTANT — DO NOT BREAK THESE

### ❌ Do NOT retrain unless you know what you’re doing
- Tokenizer
- TF-IDF vectorizer
- Embeddings
- LSTM model

They are **tightly coupled**.

### ❌ Do NOT delete `artifacts/`
The API **depends on these at startup**.

### ❌ Do NOT return raw Pandas / NumPy objects from FastAPI
- Always convert to native Python types
- NaNs must be converted to `None`

(This was already fixed in `recommender.py`.)

---

## 🧪 If You Need to Retrain (Advanced)

Only do this if:
- You are changing the dataset
- You are improving the model intentionally

Correct order:
1. Train LSTM with triplet loss
2. Regenerate book embeddings
3. Rebuild TF-IDF index (optional)
4. Restart API

---

## 🧠 Design Rationale (Why this works)

- **TF-IDF** ensures topical relevance
- **Triplet loss** gives real semantic separation
- **Offline embedding generation** makes inference fast
- **FastAPI** allows clean deployment and testing

This mirrors **industry-grade recommendation systems**.

---

## 📌 Known Limitations (Expected)

- Dataset is academic-heavy
- Short queries can be ambiguous
- “Romantic” may map to *romanticism* unless query is specific
- Not optimized for approximate NN (FAISS not used yet)

---

## 🔮 Possible Extensions

- FAISS for faster similarity search
- Language & year filters
- Query expansion
- Web UI
- Docker deployment
- Cloud hosting (Render / Railway / AWS)

---

## 👤 Handoff Notes

If you’re continuing this project:
- Start with `app/recommender.py`
- Use Swagger UI to test
- Treat `artifacts/` as read-only unless retraining

---

## ✅ Final Status

✔ End-to-end NLP pipeline  
✔ Large-scale semantic embeddings  
✔ Hybrid retrieval system  
✔ Production-ready API  
