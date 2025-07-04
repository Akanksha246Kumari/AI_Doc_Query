import os
import streamlit as st
import numpy as np
import uuid
from dotenv import load_dotenv
from utils import process_document, get_text_chunks_with_meta
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import CTransformers
from langchain.schema import Document

load_dotenv()

UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
EMBEDDING_MODEL_PATH = "model/bge-base-en-v1.5"
CROSS_ENCODER_MODEL_PATH = "model/cross-encoder_ms-marco-MiniLM-L-12-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def min_max_norm(arr):
    if arr is None or len(arr) == 0:
        return []
    arr = np.array(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.ones_like(arr)
    return (arr - min_val) / (max_val - min_val)


def clamp_score(score):
    return float(max(0.0, min(1.0, score)))


def format_as_bullet_points(text):
    lines = text.split("\n")
    bullets = [f"- {line.strip()}" for line in lines if line.strip()]
    return "\n".join(bullets)


def main():
    st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
    st.title("AI-Powered Offline Document Query System")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    from utils import ensure_models_downloaded, get_embedding_model, get_reranker_model, get_llm
    ensure_models_downloaded()

    if "vs" not in st.session_state:
        st.session_state.vs = None
        st.session_state.chat_history = []

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        chunks = []
        for u in uploaded_files:
            path = os.path.join(UPLOAD_DIR, u.name)
            with open(path, "wb") as f:
                f.write(u.getbuffer())
            st.write(f"Processing {u.name} …")
            raw = process_document(path)
            chunks.extend(get_text_chunks_with_meta(raw, u.name))

        docs = [
            Document(page_content=c["text"], metadata={"filename": c["filename"], "chunk_id": c["chunk_id"]})
            for c in chunks
        ]

        st.session_state.vs = FAISS.from_documents(
            docs, get_embedding_model(), distance_strategy=DistanceStrategy.COSINE
        )
        st.session_state.vs.save_local(VECTOR_STORE_DIR)
        st.success("✅ Document processed and indexed successfully!")

    elif st.session_state.vs is None:
        try:
            st.session_state.vs = FAISS.load_local(
                VECTOR_STORE_DIR,
                embeddings=get_embedding_model(),
                distance_strategy=DistanceStrategy.COSINE
            )
            st.success("✅ Loaded existing vector store from disk!")
        except Exception:
            st.info("📂 No documents uploaded yet. Please upload to start.")

    query = st.chat_input("Ask a question…")
    if query and st.session_state.vs:
        with st.spinner("🔍 Retrieving relevant answer…"):
            results = st.session_state.vs.similarity_search_with_score(query, k=10)
            reranker = get_reranker_model()
            pairs = [[query, doc.page_content] for doc, _ in results]
            rerank_scores_raw = reranker.predict(pairs)
            sim_scores_raw = [1 - dist for _, dist in results]

            sim_scores_norm = min_max_norm(sim_scores_raw)
            rerank_scores_norm = min_max_norm(rerank_scores_raw)

            scored = []
            for i, (doc, _) in enumerate(results):
                combined = 0.5 * sim_scores_norm[i] + 0.5 * rerank_scores_norm[i]
                scored.append((doc, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_5_chunks = [(doc, clamp_score(score)) for doc, score in scored[:5] if score >= 0.3]

            all_chunks = [
                (doc.page_content, score, doc.metadata.get("filename", "unknown"), doc.metadata.get("chunk_id", "na"))
                for doc, score in top_5_chunks
            ]
            rag_context = "\n\n".join([doc.page_content for doc, _ in top_5_chunks])
            rag_context_bullets = format_as_bullet_points(rag_context)

            llm_answer = get_llm(LOCAL_LLM_MODEL_PATH)(
                f"You are a helpful assistant. Without using any document context, answer the following question in bullet points:\n\n"
                f"Question:\n{query}\n\nAnswer:"
            ).strip()

            rag_answer = get_llm(LOCAL_LLM_MODEL_PATH)(
                f"You are a precise assistant. Based ONLY on the context provided, directly answer the user’s question."
                f" Format the answer in bullet points. Do NOT generate new questions or rephrase the input. If the answer is not found, say 'Not found in context.'\n\n"
                f"Context:\n{rag_context_bullets}\n\nQuestion:\n{query}\n\nAnswer:"
            ).strip()

            final_decision = get_llm(LOCAL_LLM_MODEL_PATH)(
                f"You are an intelligent assistant. Choose the better answer to the user's question based on accuracy and context.\n\n"
                f"Question:\n{query}\n\n"
                f"Answer 1 (from document context):\n{rag_answer}\n\n"
                f"Answer 2 (from general model knowledge):\n{llm_answer}\n\n"
                f"Only reply with 'Answer 1' or 'Answer 2'."
            ).strip()

            show_rag_chunks = final_decision.lower().startswith("answer 1")
            final_answer = rag_answer if show_rag_chunks else llm_answer

            st.session_state.chat_history.append({
                "query": query,
                "answer": final_answer,
                "source_chunks": all_chunks if show_rag_chunks else [],
                "used_rag": show_rag_chunks
            })

    for e in st.session_state.chat_history:
        st.markdown("**User:**")
        st.markdown(e["query"])
        st.markdown("**Assistant:**")
        st.markdown(e["answer"])

        if e.get("used_rag"):
            for i, (chunk, score, filename, chunk_id) in enumerate(e["source_chunks"], 1):
                unique_key = f"chunk_{filename}_{chunk_id}_{i}_{uuid.uuid4().hex[:6]}"
                with st.expander(f"Source Chunk {i} ({filename})"):
                    st.text_area("Chunk", chunk, height=250, disabled=True, key=unique_key)
                    st.markdown(f"**Relevance Score: {score * 100:.1f}%**")
        else:
            st.markdown("→ Answer generated from model’s general knowledge")

        st.markdown("---")


if __name__ == "__main__":
    main()

