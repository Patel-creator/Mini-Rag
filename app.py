import streamlit as st

from utils import load_documents, chunk_text
from embed import embed_texts, embed_query
from vector_store import VectorStore
from llm import generate_answer
# from ollama_llm import generate_answer_ollama


st.title("Mini RAG Assistant")
k = st.slider("Top-K chunks", 1, 5, 3)

@st.cache_resource
def setup_rag():

    docs = load_documents("data")

    chunks = []

    for d in docs:
        chunks.extend(chunk_text(d))

    embeddings = embed_texts(chunks)

    store = VectorStore(dim=embeddings.shape[1])

    store.add(embeddings, chunks)

    return store


store = setup_rag()


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.chat_message("user").write(msg["content"])

    else:

        st.chat_message("assistant").write(msg["content"])

        with st.expander("Retrieved Context"):
            st.write(msg["context"])


if query := st.chat_input("Ask something"):

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            q_emb = embed_query(query)
        
            results = store.search(q_emb, k=k)
        
            context = "\n\n".join(results)
        
            answer = generate_answer(context, query)
            
            st.write(answer)
            with st.expander("Retrieved Context"):
                st.write(context)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "context": context,
        }
    )