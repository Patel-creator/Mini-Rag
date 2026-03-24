# Mini RAG Assistant — Dual-Model Architecture

This project implements a simple, sophisticated Retrieval-Augmented Generation (RAG) pipeline
for answering questions using internal company documents instead of general model knowledge.

The system retrieves relevant document chunks using PyPDF2 + FAISS vector search,
and generates grounded answers using an interactive Streamlit Chat Interface.

---

## Objective

Build a chatbot that:

- Retrieves relevant document chunks from text and PDF files
- Generates answers strictly from retrieved context
- Features an interactive, memory-aware Chat UI
- Supports **Dual-Model Evaluation**: Directly compares local HuggingFace models against local Ollama models.

---

## Tech Stack

- **Python**
- **sentence-transformers** (Embeddings)
- **FAISS** (Vector Search via CPU)
- **Transformers** (HuggingFace Flan-T5)
- **Ollama** (Phi-3 local API)
- **Streamlit** (Interactive Chat UI)
- **PyPDF2** (PDF Parsing)

---

## Project Structure

```
rag-assistant/
├── data/                    # PDF and Text documents
├── embed.py                 # Embedding functions
├── vector_store.py          # FAISS vector store database
├── utils.py                 # Loading & chunking logic 
├── llm.py                   # HuggingFace LLM generation
├── ollama_llm.py            # Ollama API generation 
├── rag.py                   # CLI-based RAG pipeline
├── app.py                   # Streamlit Chat Interface Frontend
├── evaluation.py            # Automated multi-question model evaluator
└── README.md
```

---

## Setup

1. **Install Python dependencies:**

```bash
pip install streamlit sentence-transformers faiss-cpu numpy torch transformers PyPDF2
```

2. **Install & Run Ollama:**
Download Ollama and pull the `phi3` model:
```bash
ollama run phi3
```

3. **Add documents** (PDFs, Markdown, or TXT) to the `data/` folder.

---

## Usage

### 1. Interactive Chat Application

Launch the Streamlit interactive chat UI:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### 2. Model Evaluation Suite

Run the automated evaluation pipeline to compare HuggingFace and Ollama responses side-by-side using predefined questions:

```bash
python evaluation.py
```

### 3. CLI Pipeline

Test the basic terminal-only interactive pipeline:

```bash
python rag.py
```

---

## How It Works

1. **Loading & Chunking:** `utils.py` loads PDFs and text files and splits them into overlapping chunks.
2. **Embedding:** `embed.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert chunks into vectors.
3. **Vector Store:** `vector_store.py` uses FAISS for lightning-fast similarity search to find the Top-K chunks.
4. **Generation:** 
    - `llm.py` uses Flan-T5 sequentially.
    - `ollama_llm.py` connects to background Ollama processes for cutting-edge local inference.

---

## Customization

- Change the HuggingFace model in `llm.py`.
- Change the Ollama target model in `ollama_llm.py`.
- Adjust the text chunking size and overlap thresholds in `utils.py`.

---

## License

MIT

---

## Contact

[Priyanshu Patel]
[Patelpriyanshu77777@gmail.com]
