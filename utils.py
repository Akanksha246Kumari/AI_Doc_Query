
import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from langchain_community.llms import CTransformers
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
EMBEDDING_MODEL_PATH = "model/bge-base-en-v1.5"
CROSS_ENCODER_MODEL_PATH = "model/cross-encoder_ms-marco-MiniLM-L-12-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# ✅ Score normalization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clamp_score(score):
    return float(max(0.0, min(1.0, score)))

# ✅ Model downloader
def ensure_models_downloaded():
    # ✅ BGE Embedding Model
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        st.info("⏬ Downloading BGE Embedding Model...")
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        model.save(EMBEDDING_MODEL_PATH)

    # ✅ Cross Encoder Reranker Model
    if not os.path.exists(CROSS_ENCODER_MODEL_PATH):
        st.info("⏬ Downloading Cross Encoder Reranker...")
        model = AutoModel.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", local_files_only=False)
        
        os.makedirs(CROSS_ENCODER_MODEL_PATH, exist_ok=True)
        model.save_pretrained(CROSS_ENCODER_MODEL_PATH)
        tokenizer.save_pretrained(CROSS_ENCODER_MODEL_PATH)


# ✅ Streamlit cached models
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"local_files_only": True}
    )

@st.cache_resource
def get_reranker_model():
    return CrossEncoder(CROSS_ENCODER_MODEL_PATH, local_files_only=True)

@st.cache_resource
def get_llm(model_path):
    return CTransformers(
        model=model_path,
        model_type="mistral",
        config={'max_new_tokens': 256, 'temperature': 0.01, 'context_length': 4096}
    )

# ✅ Text formatting for LLM response
def format_as_bullet_points(text):
    lines = text.split("\n")
    bullets = [f"- {line.strip()}" for line in lines if line.strip()]
    return "\n".join(bullets)

# ✅ OCR + PDF handling
def process_document(path):
    ext = path.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file format")

    return text

# ✅ Text chunking with metadata using recursive splitter
def get_text_chunks_with_meta(text, filename):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "filename": filename,
            "chunk_id": f"{filename}_chunk_{i+1}"
        }
        for i, chunk in enumerate(chunks)
    ]