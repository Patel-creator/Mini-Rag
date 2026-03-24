from utils import load_documents, chunk_text
from embed import embed_texts, embed_query
from vector_store import VectorStore

from llm import generate_answer
from ollama_llm import generate_answer_ollama


docs = load_documents("data")

chunks = []

for d in docs:
    chunks.extend(chunk_text(d))


embeddings = embed_texts(chunks)

store = VectorStore(dim=embeddings.shape[1])

store.add(embeddings, chunks)


questions = [

    "What is quality assurance?",
    "What is escrow payment?",
    "What are package prices?",
    "What is maintenance program?",
    "Do you provide real-time tracking?",
    "What is stage based payment?",
    "What is Pinnacle package price?",
    "What happens if delay occurs?",
    "What is wallet amount?",
    "What does Indecimal promise?"

]


for q in questions:

    print("\n========================")
    print("Question:", q)

    q_emb = embed_query(q)

    results = store.search(q_emb, k=3)

    context = "\n\n".join(results)

    print("\nContext:\n", context)


    print("\nFlanT5 Answer:")
    print(generate_answer(context, q))


    print("\nOllama Answer:")
    print(generate_answer_ollama(context, q))