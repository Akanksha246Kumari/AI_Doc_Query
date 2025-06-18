import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from utils import process_document, get_text_chunks, clean_text
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import CTransformers

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
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

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Document Query", layout="wide")
    st.title("AI-Powered Offline Document Query System")

    # --- Document Upload and Processing ---
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            text = process_document(file_path)
            if not text or not text.strip() or "Tesseract not found" in text:
                st.error(
                    f"Failed to extract text. Reason: {text if text else 'No text found.'}"
                )
                return

            chunks = get_text_chunks(text)
            if not chunks:
                st.error("Could not generate text chunks. The document may be empty.")
                return

            # Build vector store using Cosine Similarity
            embeddings = get_embedding_model()
            vs = FAISS.from_texts(
                chunks, embeddings, distance_strategy=DistanceStrategy.COSINE
            )
            st.session_state.vs = vs
            st.session_state.embeddings = embeddings
            st.success("Document indexed successfully!")

    # ---- Query Processing ----
    query = st.text_input("Enter your query:")
    if st.button("Search") and query:
        if "vs" not in st.session_state:
            st.error("Please upload and process a document first.")
            return

        with st.spinner("Retrieving answers ..."):
            try:
                # 1. Similarity search
                docs_and_scores = st.session_state.vs.similarity_search_with_score(query, k=5)

                # 2. Rerank
                reranker = get_reranker_model()
                pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
                rerank_scores = reranker.predict(pairs)

                # 3. Weighted combination
                combined = []
                for (doc, dist), rerank_raw in zip(docs_and_scores, rerank_scores):
                    orig_sim = 1.0 - dist
                    try:
                        rerank_sim = sigmoid(rerank_raw)
                        if np.isnan(rerank_sim):
                            rerank_sim = 0.0
                    except (OverflowError, ValueError):
                        rerank_sim = 0.0
                    
                    score = orig_sim * 0.3 + rerank_sim * 0.7
                    combined.append((doc, score))
                results = sorted(combined, key=lambda x: x[1], reverse=True)

                # 4. Decide if any chunk is relevant enough
                RELEVANCE_THRESHOLD = 0.60
                llm = get_llm(LOCAL_LLM_MODEL_PATH)

                is_irrelevant = not results or np.isnan(results[0][1]) or results[0][1] < RELEVANCE_THRESHOLD

                if is_irrelevant:
                    # If no chunk is relevant, answer from model's knowledge directly
                    st.subheader("Answer")
                    prompt = f"Directly answer the question in a single, short sentence. Do not add any other explanation.\nQuestion: {query}\nAnswer:"
                    answer = llm(prompt).strip()
                    st.write(f"1. {answer}")
                    st.caption("Answer from model knowledge (no relevant document context)")
                    return

                # 5. Display results for relevant queries
                st.header(f"Top {len(results)} Results")
                for i, (doc, score) in enumerate(results, 1):
                    with st.container(border=True):
                        prompt = (
                            """Answer ONLY from this context (1-2 sentences):\n"""
                            f"{doc.page_content}\n\nQuestion: {query}\nAnswer:"
                        )
                        answer = llm(prompt).strip()
                        st.write(f"{i}. {answer}")

                        with st.expander("📄 Source Chunk"):
                            st.text_area("", value=doc.page_content, height=200, disabled=True)

            except Exception as exc:
                st.error(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
