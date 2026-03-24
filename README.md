# Mini RAG Assistant — Construction Marketplace AI

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline
for answering questions using internal company documents instead of general model knowledge.

The system retrieves relevant document chunks using embeddings + vector search,
and generates grounded answers using an LLM.

---

## Objective

Build a chatbot that:

- Retrieves relevant document chunks
- Generates answers only from retrieved context
- Shows retrieved context for transparency
- Runs locally with a custom frontend

---

## Tech Stack

- Python
- sentence-transformers (embeddings)
- FAISS (vector search)
- Transformers (LLM)
- Streamlit (frontend UI)

---

## Project Structure

```
rag-assistant/
├── data/                    # PDF documents
├── embed.py                 # Embedding functions
├── vector_store.py          # FAISS vector store
├── llm.py                   # LLM generation
├── rag.py                   # Main RAG pipeline
├── app.py                   # Streamlit frontend
└── README.md
```

---

## Setup

1. **Install dependencies:**

```bash
pip install streamlit sentence-transformers faiss-cpu
```

2. **Add PDF documents** to the `data/` folder.

---

## Usage

### Run the RAG pipeline

```bash
python rag.py
```

This will:
- Load documents
- Chunk them
- Build FAISS index
- Answer a sample question

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## How It Works

### 1. Loading & Chunking

`utils.py` loads PDFs and splits them into overlapping chunks.

### 2. Embedding

`embed.py` uses `sentence-transformers` to convert chunks into vectors.

### 3. Vector Store

`vector_store.py` uses FAISS for fast similarity search.

### 4. Retrieval

When a question is asked:
- Embed the question
- Search FAISS for top-k chunks

### 5. Generation

`llm.py` uses Flan-T5 to generate an answer based **only** on retrieved chunks.

---

## Customization

- Change embedding model in `embed.py`
- Change LLM in `llm.py`
- Adjust chunk size in `utils.py`
- Modify FAISS parameters in `vector_store.py`

---

## License

MIT

---

## Contact

[Priyanshu Patel]
[Patelpriyanshu77777@gmail.com]
