# 📚 Book Recommendation System
### NLP + LSTM + Hybrid Retrieval

---

## 🔍 Project Overview

A large-scale book recommendation engine using Natural Language Processing (NLP). It generates semantic embeddings for over 1 million books using an LSTM-based model trained with self-supervised triplet loss, and serves recommendations via a FastAPI backend.

📦 **Download data, embeddings, and TF-IDF:** [Google Drive](https://drive.google.com/drive/folders/1dnTdDDlWa3BFaPdeN_fCEEDd2W-200NS?usp=sharing)

---

## 🧠 Core System — Recommendation Engine

The system uses a **hybrid retrieval strategy**:

- **TF-IDF** (lexical retrieval) → candidate generation
- **LSTM semantic embeddings** → ranking

This ensures topical correctness and semantic understanding.

### High-Level Architecture

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
│   ├── main.py               # FastAPI entry point
│   ├── recommender.py        # Recommendation logic (DO NOT MODIFY CARELESSLY)
│   └── schemas.py
│
├── artifacts/                # Production assets
│   ├── models/
│   ├── embeddings/
│   ├── tokenizer/
│   └── tfidf/
│
├── autocomplete_lstm/        # ⭐ Autocomplete module
│   ├── data/
│   │   ├── titles.txt
│   │   └── dataset.npz
│   │
│   ├── preprocessing/
│   │   ├── extract_titles.py
│   │   ├── check_tokenizer.py
│   │   └── build_dataset.py
│   │
│   ├── model/
│   │   └── train_lstm.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── tokenizer/
│   │   └── tokenizer.pkl
│   │
│   └── weights/
│       └── lstm.pth
│
├── src/
├── requirements.txt
└── README.md
```

---

## ✨ Autocomplete System

### Goal

Given a partial query like `"harry potter"`, the system provides:

- 🔍 Matching book titles
- 🔮 Next-word predictions

### Architecture

```
User Input
    ↓
1. Exact Title Matching (Prefix Search)
    ↓
2. LSTM Prediction (Fallback)
    ↓
Output: Matching titles + Predicted continuation
```

### Implementation Details

| Component | Detail |
|-----------|--------|
| **Data** | 918,930 book titles extracted to `autocomplete_lstm/data/titles.txt` |
| **Tokenizer** | Vocabulary size 50,000, built on titles only, stored at `autocomplete_lstm/tokenizer/tokenizer.pkl` |
| **Dataset** | Sliding window (5-word input → 2-word output), ~4.97M samples, stored as `dataset.npz` |
| **Model** | PyTorch LSTM: Embedding → LSTM → FC layer (predicts 2 words), GPU-enabled |

### Why Hybrid?

The LSTM model alone tends to predict generic phrases like `"of the"` or `"in the"` because it learns language frequency patterns rather than actual book titles. This is expected behavior for sequence models. The hybrid approach fixes this:

| Approach | Problem |
|----------|---------|
| Only LSTM | Predicts generic phrases |
| Only Matching | No intelligent completion |
| **Hybrid** ✅ | Best of both |

**Primary method — Exact Title Matching:** Matches user input directly against titles and returns real book names.

**Secondary method — LSTM Prediction:** Used as a fallback when no strong title matches are found.

---

## 🚀 API Reference

### Start the Server

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python -m uvicorn app.main:app

# 3. Open Swagger UI
# http://127.0.0.1:8000/docs
```

### `POST /recommend`

Returns top-K book recommendations for a given query.

### `POST /autocomplete`

**Request:**
```json
{
  "query": "harry potter"
}
```

**Response:**
```json
{
  "query": "harry potter",
  "matches": [
    "harry potter and the goblet of fire",
    "harry potter and the chamber of secrets"
  ],
  "prediction": "and the"
}
```

---

## 🧪 Autocomplete Training Pipeline

If rebuilding the autocomplete system from scratch:

```bash
# Step 1: Extract titles
python autocomplete_lstm/preprocessing/extract_titles.py

# Step 2: Build dataset
python autocomplete_lstm/preprocessing/build_dataset.py

# Step 3: Train model
python autocomplete_lstm/model/train_lstm.py
```

---

## ⚠️ Important Notes

### Do NOT Modify
- `artifacts/` — tokenizer, TF-IDF, and embeddings
- Do not retrain the recommendation model unless explicitly required

### Known Limitations

**Recommendation System:**
- Dataset is academically weighted
- Short queries can be vague

**Autocomplete System:**
- LSTM predictions are generic by design
- Exact title matching is the primary method

---

## 🔮 Future Improvements

- FAISS for fast similarity search
- Better autocomplete ranking
- Transformer-based autocomplete
- Web UI
- Cloud deployment

---

## 👤 Contribution Summary

**Existing System:**
- Hybrid recommendation engine (TF-IDF + LSTM embeddings)
- FastAPI deployment

**New Work:**
- Autocomplete system (titles-based)
- PyTorch LSTM model
- Dataset generation pipeline
- Hybrid autocomplete (matching + prediction)
- Swagger integration

---

## ✅ Status

| Feature | Status |
|---------|--------|
| Large-scale NLP recommendation system | ✔ Complete |
| Hybrid retrieval (TF-IDF + embeddings) | ✔ Complete |
| Autocomplete system | ✔ Complete |
| API-ready with Swagger UI | ✔ Complete |
| GPU-supported training (PyTorch) | ✔ Complete |