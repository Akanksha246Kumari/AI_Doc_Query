import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from utils import process_document, get_text_chunks
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import CTransformers

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
UPLOAD_DIR = "data/uploads"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Helper Functions ---
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
        config={'max_new_tokens': 256, 'temperature': 0.01, 'context_length': 4096}
    )

# --- Chat Style App ---
def main():
    st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
    st.title("AI-Powered Offline Document Query System")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vs" not in st.session_state:
        st.session_state.vs = None

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Extracting and indexing document..."):
            text = process_document(file_path)
            if not text:
                st.error("Could not extract text from document.")
                return

            chunks = get_text_chunks(text)
            embeddings = get_embedding_model()
            vs = FAISS.from_texts(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
            st.session_state.vs = vs
            st.success("Document processed and indexed successfully!")

    query = st.chat_input("Ask a question...")

    if query and st.session_state.vs:
        with st.spinner("Generating answer..."):
            embedding_model = st.session_state.embeddings if "embeddings" in st.session_state else get_embedding_model()
            reranker = get_reranker_model()
            llm = get_llm(LOCAL_LLM_MODEL_PATH)

            docs_and_scores = st.session_state.vs.similarity_search_with_score(query, k=5)
            pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
            rerank_scores = reranker.predict(pairs)

            combined = []
            for (doc, dist), rerank_raw in zip(docs_and_scores, rerank_scores):
                orig_sim = 1.0 - dist
                try:
                    rerank_sim = sigmoid(rerank_raw)
                    if np.isnan(rerank_sim): rerank_sim = 0.0
                except: rerank_sim = 0.0
                score = orig_sim * 0.3 + rerank_sim * 0.7
                combined.append((doc, score))
            combined = sorted(combined, key=lambda x: x[1], reverse=True)

            top_context = combined[0][0].page_content if combined else ""
            rag_prompt = f"Answer the question using only the context below. Be brief.\nContext:\n{top_context}\n\nQuestion: {query}\nAnswer:"
            rag_answer = llm(rag_prompt).strip()

            llm_prompt = f"Answer this question truthfully.\nQuestion: {query}\nAnswer:"
            llm_answer = llm(llm_prompt).strip()

            best_answer = rag_answer if combined[0][1] > 0.6 else llm_answer
            source_info = ("rag" if best_answer == rag_answer else "llm")

            st.session_state.chat_history.append({
                "query": query,
                "best": best_answer,
                "source": source_info,
                "context": top_context if source_info == "rag" else None
            })

    # --- Display chat history with text labels ---
    for entry in st.session_state.chat_history:
        # User message
        st.markdown("**User:**")
        st.markdown(f"{entry['query']}")
        
        # Assistant message
        st.markdown("**Assistant:**")
        st.markdown(entry["best"])
        if entry["source"] == "rag":
            with st.expander("View Source Chunk"):
                st.text_area("", entry["context"], height=200, disabled=True, label_visibility="collapsed")
        elif entry["source"] == "llm":
            st.caption("Answer generated from model's general knowledge")
        st.markdown("\n")

if __name__ == "__main__":
    main()
