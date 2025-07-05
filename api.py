import os
import json
import numpy as np
import warnings
import logging
from dotenv import load_dotenv
from utils import (
    ensure_models_downloaded,
    get_embedding_model,
    get_reranker_model,
    get_llm,
    process_document,
    get_text_chunks_with_meta
)
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ---- Config & Suppressions ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
for name in ["streamlit", "transformers", "urllib3", "langchain"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

load_dotenv()

UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
EMBEDDING_MODEL_PATH = "model/bge-base-en-v1.5"
CROSS_ENCODER_MODEL_PATH = "model/cross-encoder_ms-marco-MiniLM-L-12-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


# ---- Utility Functions ----
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


# ---- Core Wrapper ----
def ask_question_from_pdf(pdf_path, query):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # ✅ Download model if not already in /model/
    ensure_models_downloaded()

    # ✅ Extract and chunk text
    raw = process_document(pdf_path)
    filename = os.path.basename(pdf_path)
    chunks = get_text_chunks_with_meta(raw, filename)

    docs = [
        Document(page_content=c["text"], metadata={"filename": c["filename"], "chunk_id": c["chunk_id"]})
        for c in chunks
    ]

    # ✅ Build or load vectorstore
    vs = FAISS.from_documents(docs, get_embedding_model(), distance_strategy=DistanceStrategy.COSINE)
    vs.save_local(VECTOR_STORE_DIR)

    # ✅ Similarity + Rerank
    results = vs.similarity_search_with_score(query, k=10)
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
        {
            "chunk": doc.page_content,
            "score": score,
            "filename": doc.metadata.get("filename", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "na")
        }
        for doc, score in top_5_chunks
    ]
    rag_context = "\n\n".join([doc.page_content for doc, _ in top_5_chunks])
    rag_context_bullets = format_as_bullet_points(rag_context)

    llm = get_llm(LOCAL_LLM_MODEL_PATH)

    llm_answer = llm(
        f"You are a helpful assistant. Without using any document context, answer the following question in bullet points:\n\n"
        f"Question:\n{query}\n\nAnswer:"
    ).strip()

    rag_answer = llm(
        f"You are a precise assistant. Based ONLY on the context provided, directly answer the user’s question."
        f" Format the answer in bullet points. Do NOT generate new questions or rephrase the input. If the answer is not found, say 'Not found in context.'\n\n"
        f"Context:\n{rag_context_bullets}\n\nQuestion:\n{query}\n\nAnswer:"
    ).strip()

    final_decision = llm(
        f"You are an intelligent assistant. Choose the better answer to the user's question based on accuracy and context.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer 1 (from document context):\n{rag_answer}\n\n"
        f"Answer 2 (from general model knowledge):\n{llm_answer}\n\n"
        f"Only reply with 'Answer 1' or 'Answer 2'."
    ).strip()

    show_rag_chunks = final_decision.lower().startswith("answer 1")
    final_answer = rag_answer if show_rag_chunks else llm_answer

    return {
        "query": query,
        "answer": final_answer,
        "source_chunks": all_chunks if show_rag_chunks else [],
        "used_rag": show_rag_chunks
    }


# ---- For CLI test only ----
if __name__ == "__main__":
    pdf_path = "/Users/akankshakumari/Desktop/Doc-query/data/uploads/Shreeya_Srinath_Feedback.pdf"  # Change as needed
    query = "What is this document about?"
    result = ask_question_from_pdf(pdf_path, query)

    print(f"\nUSER:\n{result['query']}\n")
    print(f"ASSISTANT:\n{result['answer']}\n")

    if result.get("used_rag") and result.get("source_chunks"):
        print("SOURCE CHUNKS:\n")
        for i, chunk_info in enumerate(result["source_chunks"], 1):
            print(f"Chunk {i} ({chunk_info['filename']}):")
            print(chunk_info["chunk"])
            print(f"Relevance Score: {chunk_info['score'] * 100:.1f}%\n")
    else:
        print("→ Answer generated from model’s general knowledge.\n")
