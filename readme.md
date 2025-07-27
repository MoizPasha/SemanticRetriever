# Semantic Retriever

A lightweight, general-purpose retrieval system for matching natural language queries to a corpus of pre-embedded text entries using [Sentence Transformers](https://www.sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss).

---

## 🔍 Use Cases

- General Q&A systems
- Chatbot memory lookup
- Semantic document or snippet search
- Knowledge base search
- Fast retrieval for Retrieval-Augmented Generation (RAG)

---

## 🚀 Features

- 🔎 Fast dense vector search with FAISS
- 🧠 Transformer-based sentence embeddings (`intfloat/e5-base-v2`)
- ✅ Embedding normalization for cosine similarity
- 💬 Easily extendable to any domain (FAQs, docs, KBs, etc.)
- 💡 Model-agnostic architecture — swap in your own sentence encoder

---

## 📦 Requirements

- Python 3.7+
- `sentence-transformers`
- `faiss-cpu`
- `numpy`

```bash
pip install sentence-transformers faiss-cpu numpy
py embedder.py
py retriever.py
```