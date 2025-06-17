import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from utils import process_document, get_text_chunks
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain.llms import CTransformers

# Configuration
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
load_dotenv()

# Path setup
UPLOAD_DIR = os.path.join("data", "uploads")
VECTOR_STORE_DIR = os.path.join("data", "vector_store")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

@st.cache_resource
def get_llm(model_path):
    if not os.path.exists(model_path):
        st.error(f"LLM model not found at '{model_path}'.")
        return None
    return CTransformers(
        model=model_path,
        model_type='mistral',
        config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 4096}
    )

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def get_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu', max_length=512)

def main():
    st.set_page_config(page_title="Document Query", layout="wide")
    st.title("AI-Powered Offline Document Query System")

    # Document processing
    uploaded = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded:
        with st.spinner("Processing document chunks (500 tokens)..."):
            path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            
            text = process_document(path)
            chunks = get_text_chunks(text)
            emb = get_embedding_model()
            vs = FAISS.from_texts(chunks, emb)
            st.session_state.vs = vs
            st.session_state.embeddings = emb

    # RAG Query Processing
    query = st.text_input("Enter your query:")
    if st.button("Search") and query:
        if 'vs' not in st.session_state:
            st.error("Please upload a document first")
            return
            
        with st.spinner("Processing..."):
            try:
                # 1. Get query embedding
                query_embedding = st.session_state.embeddings.embed_query(query)
                
                # 2. Similarity search with scores
                docs_and_scores = st.session_state.vs.similarity_search_with_score(query, k=5)
                
                # 3. Rerank with CrossEncoder
                reranker = get_reranker_model()
                pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
                scores = reranker.predict(pairs)
                
                # 4. Combine and sort results
                combined = [(doc, orig_score*0.3 + rerank_score*0.7) 
                           for (doc, orig_score), rerank_score in zip(docs_and_scores, scores)]
                results = sorted(combined, key=lambda x: x[1], reverse=True)
                
                # 5. Generate and display answers
                llm = get_llm(LOCAL_LLM_MODEL_PATH)
                st.header(f"Top {len(results)} Results")
                
                for i, (doc, score) in enumerate(results, 1):
                    with st.container(border=True):
                        # Generate concise RAG answer
                        prompt = f"""Answer ONLY from this context (1-2 sentences):
                        {doc.page_content}
                        
                        Question: {query}
                        Answer:"""
                        answer = llm(prompt).strip()
                        
                        # Display result
                        st.write(f"{i}. {answer}")
                        #st.caption(f"Relevance score: {score:.2f}")
                        
                        # Show source chunk
                        with st.expander(f"📄 Source Chunk"):
                            st.text_area("", value=doc.page_content, height=200, disabled=True)
                            
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
