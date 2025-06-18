"""
Simple API Wrapper for Document Query System

This module provides a single function that takes a query and returns formatted results.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils import process_document, get_text_chunks, clean_text
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import CTransformers

# --- Constants ---
UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Initialize models (cached)
def init_models():
    """Initialize and cache all required models."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    reranker = CrossEncoder(CROSS_ENCODER_MODEL)
    llm = CTransformers(
        model=LOCAL_LLM_MODEL_PATH,
        model_type="mistral",
        config={'max_new_tokens': 256, 'temperature': 0.01, 'context_length': 4096}
    )
    return embeddings, reranker, llm

# Initialize models once
_embeddings, _reranker, _llm = init_models()

def query_document(query: str, doc_path: str) -> Dict:
    """
    Process a query against a document and return formatted results matching Streamlit UI.
    
    Args:
        query: User's query string
        doc_path: Path to the document file
        
    Returns:
        Dict containing:
            - 'status': 'success' or 'error'
            - 'output': Formatted string matching Streamlit UI
            - 'is_knowledge_based': Boolean indicating if answer came from model knowledge
    """
    try:
        # Process document and build index
        text = process_document(doc_path)
        if not text or not text.strip() or "Tesseract not found" in text:
            raise ValueError(f"Failed to extract text. Reason: {text if text else 'No text found.'}")

        chunks = get_text_chunks(text)
        if not chunks:
            raise ValueError("Could not generate text chunks. The document may be empty.")

        # Build vector store using Cosine Similarity
        vs = FAISS.from_texts(
            chunks, _embeddings, distance_strategy=DistanceStrategy.COSINE
        )

        # Similarity search
        docs_and_scores = vs.similarity_search_with_score(query, k=5)

        # Rerank
        pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
        rerank_scores = _reranker.predict(pairs)

        # Weighted combination
        combined = []
        for (doc, dist), rerank_raw in zip(docs_and_scores, rerank_scores):
            orig_sim = 1.0 - dist
            try:
                rerank_sim = 1 / (1 + np.exp(-rerank_raw))
                if np.isnan(rerank_sim):
                    rerank_sim = 0.0
            except (OverflowError, ValueError):
                rerank_sim = 0.0
            
            score = orig_sim * 0.3 + rerank_sim * 0.7
            combined.append((doc, score))
        results = sorted(combined, key=lambda x: x[1], reverse=True)

        # Decide if any chunk is relevant enough
        RELEVANCE_THRESHOLD = 0.60
        is_irrelevant = not results or np.isnan(results[0][1]) or results[0][1] < RELEVANCE_THRESHOLD

        if is_irrelevant:
            # If no chunk is relevant, answer from model's knowledge directly
            prompt = f"Directly answer the question in a single, short sentence. Do not add any other explanation.\nQuestion: {query}\nAnswer:"
            answer = _llm(prompt).strip()
            return {
                'status': 'success',
                'output': f"1. {answer}\n\nAnswer from model knowledge (no relevant document context)",
                'is_knowledge_based': True
            }

        # Generate RAG-based answers
        output = []
        for i, (doc, score) in enumerate(results, 1):
            prompt = (
                """Answer ONLY from this context (1-2 sentences):\n"""
                f"{doc.page_content}\n\nQuestion: {query}\nAnswer:"
            )
            answer = _llm(prompt).strip()
            output.append(f"{i}. {answer}\n")
            output.append("Source Chunk:\n")
            output.append(doc.page_content)
            output.append("\n\n")

        return {
            'status': 'success',
            'output': "".join(output),
            'is_knowledge_based': False
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'output': f"Error: {str(e)}",
            'is_knowledge_based': False
        }

# Example usage:
if __name__ == "__main__":
    # Example query
    doc_path = "data/uploads/1706.03762v7.pdf"  # Example document path
    query = "What is positional encoding?"
    
    result = query_document(query, doc_path)
    
    print("\nQuery Results:")
    print(f"Status: {result['status']}")
    print(f"Is Knowledge Based: {result['is_knowledge_based']}")
    print("\nAnswers:")
    for i, answer in enumerate(result['answers'], 1):
        print(f"{i}. {answer}")
        
    if not result['is_knowledge_based']:
        print("\nSource Chunks:")
        for i, chunk in enumerate(result['source_chunks'], 1):
            print(f"\nSource Chunk {i}:")
            print(chunk)
