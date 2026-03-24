from utils import load_documents, chunk_text
from embed import embed_texts, embed_query
from vector_store import VectorStore
from llm import generate_answer


docs = load_documents("data")

chunks = []

for d in docs:
    chunks.extend(chunk_text(d))


embeddings = embed_texts(chunks)


store = VectorStore(dim=embeddings.shape[1])

store.add(embeddings, chunks)


query = input("Ask: ")


q_emb = embed_query(query)

results = store.search(q_emb, k=3)


context = "\n\n".join(results)


print("\nRetrieved Context:\n")
print(context)


answer = generate_answer(context, query)


print("\nFinal Answer:\n")
print(answer)