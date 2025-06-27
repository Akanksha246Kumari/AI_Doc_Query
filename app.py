import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from new_utils import process_document, get_text_chunks_with_meta
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import CTransformers
from langchain.schema import Document

load_dotenv()

UPLOAD_DIR = "data/uploads"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
RAG_THRESHOLD = 0.7  # 70%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_reranker_model():
    return CrossEncoder(CROSS_ENCODER_MODEL)

@st.cache_resource
def get_llm(model_path):
    return CTransformers(
        model=model_path,
        model_type="mistral",
        config={'max_new_tokens': 256, 'temperature': 0.01, 'context_length':4096}
    )

def clamp_score(score):
    return float(max(0.0, min(1.0, score)))

def main():
    st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
    st.title("AI-Powered Offline Document Query System")

    if "vs" not in st.session_state:
        st.session_state.vs = None
        st.session_state.chat_history = []

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        chunks = []
        for u in uploaded_files:
            path = os.path.join(UPLOAD_DIR, u.name)
            with open(path, "wb") as f:
                f.write(u.getbuffer())
            st.write(f"Processing `{u.name}` …")
            raw = process_document(path)
            chunks.extend(get_text_chunks_with_meta(raw, u.name))

        docs = [
            Document(page_content=c["text"],
                     metadata={"filename": c["filename"], "chunk_id": c["chunk_id"]})
            for c in chunks
        ]

        st.session_state.vs = FAISS.from_documents(
            docs, get_embedding_model(), distance_strategy=DistanceStrategy.COSINE
        )
        st.success("Document processed and indexed successfully!")

    query = st.chat_input("Ask a question…")
    if query and st.session_state.vs:
        with st.spinner("Retrieving…"):
            results = st.session_state.vs.similarity_search_with_score(query, k=5)
            reranker = get_reranker_model()
            pairs = [[query, doc.page_content] for doc, _ in results]
            rerank_scores = reranker.predict(pairs)

            scored = []
            for (doc, dist), raw in zip(results, rerank_scores):
                sim = 1 - dist
                rerank_sim = sigmoid(raw) if not np.isnan(raw) else 0.0
                combined = 0.3 * sim + 0.7 * rerank_sim
                scored.append((doc, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_doc = scored[0][0]
            top_score = clamp_score(scored[0][1])

            all_chunks = [doc.page_content for doc, _ in scored]
            concatenated_chunks = "\n\n".join(all_chunks)

            final_answer = get_llm(LOCAL_LLM_MODEL_PATH)(
                f"You are a helpful assistant. Use only the context provided below to answer the user's question. "
                f"Do not invent new questions or answers. Only respond to the user’s question using a clear, friendly tone. "
                f"Structure the answer with bullet points or short paragraphs for easy reading.\n\n"
                f"Context:\n{top_doc.page_content}\n\n"
                f"Question:\n{query}\n\n"
                f"Answer:"
            ).strip()


            st.session_state.chat_history.append({
                "query": query,
                "answer": final_answer,
                "score": top_score,
                "source_chunks": concatenated_chunks,
                "filename": top_doc.metadata.get("filename", "unknown")
            })

    for e in st.session_state.chat_history:
        st.markdown("**User:**")
        st.markdown(e["query"])
        st.markdown("**Assistant:**")
        st.markdown(e["answer"])

        if e["score"] >= RAG_THRESHOLD:
            with st.expander(f"View Source Chunks ({e['filename']})"):
                st.text_area("", e["source_chunks"], height=250, disabled=True)
            st.markdown(f"Relevance Score: {e['score']*100:.1f}%")
        else:
            st.markdown("->Answer generated from model’s general knowledge")

        st.markdown("---")

if __name__ == "__main__":
    main()