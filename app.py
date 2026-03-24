import streamlit as st

from utils import load_documents, chunk_text
from embed import embed_texts, embed_query
from vector_store import VectorStore
from llm import generate_answer


st.title("Mini RAG Assistant")


@st.cache_resource
def setup_rag():

    docs = load_documents("data")

    chunks = []

    for d in docs:
        chunks.extend(chunk_text(d))

    embeddings = embed_texts(chunks)

    store = VectorStore(dim=embeddings.shape[1])

    store.add(embeddings, chunks)

    return store, chunks


store, chunks = setup_rag()


query = st.text_input("Ask a question")


if query:

    q_emb = embed_query(query)

    results = store.search(q_emb, k=3)

    context = "\n\n".join(results)

    st.subheader("Retrieved Context")

    st.write(context)

    answer = generate_answer(context, query)

    st.subheader("Final Answer")

    st.write(answer)