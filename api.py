import os
import re
import logging
import numpy as np
from typing import Dict
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain.schema import Document
from new_utils import process_document, get_text_chunks_with_meta

# Suppress logs
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

# Constants
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_PATH      = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

SELECTION_THRESHOLD = 0.6  # choose RAG when score ≥ 60%
DISPLAY_THRESHOLD   = 0.7  # show chunk/relevance when score ≥ 70%

class DocumentQueryWrapper:
    def __init__(self):
        self.embedding = None
        self.reranker  = None
        self.llm       = None
        self.vs        = None
        self._init_models()

    def _init_models(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sentence_transformers import CrossEncoder
        from langchain_community.llms import CTransformers

        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.reranker  = CrossEncoder(CROSS_ENCODER_MODEL)
        self.llm       = CTransformers(
            model=LLM_MODEL_PATH,
            model_type="mistral",
            config={'max_new_tokens':256,'temperature':0.01,'context_length':4096}
        )

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def load_document(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        text = process_document(path)
        chunks = get_text_chunks_with_meta(text, Path(path).name)
        docs = [Document(page_content=c["text"], metadata={"filename": c["filename"], "chunk_id": c["chunk_id"]}) for c in chunks]
        self.vs = FAISS.from_documents(
            docs, self.embedding, distance_strategy=DistanceStrategy.COSINE
        )
        return True

    def query_document(self, query: str) -> Dict:
        if self.vs is None:
            return {
                "user":     query,
                "answer":   "No document loaded.",
                "source":   "llm",
                "context":  None,
                "score":    0.0,
                "filename": ""
            }

        hits = self.vs.similarity_search_with_score(query, k=5)
        pairs = [[query, doc.page_content] for doc,_ in hits]
        rerank_raw = self.reranker.predict(pairs)

        scored = []
        for (doc, dist), raw in zip(hits, rerank_raw):
            sim        = 1.0 - dist
            rerank_sim = self._sigmoid(raw) if not np.isnan(raw) else 0.0
            combined   = 0.3 * sim + 0.7 * rerank_sim
            scored.append((doc, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_doc = scored[0][0]
        top_score = max(0.0, min(1.0, scored[0][1]))
        context = "\n\n".join([doc.page_content for doc, _ in scored])

        prompt = (
            f"You are a helpful assistant. Use only the context provided below to answer the user's question. "
            f"Do not invent new questions or answers. Only respond to the user’s question using a clear, friendly tone. "
            f"Structure the answer with bullet points or short paragraphs for easy reading.\n\n"
            f"Context:\n{top_doc.page_content}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:"
        )

        if top_score >= SELECTION_THRESHOLD:
            final = self.llm(prompt).strip()
            source = "rag"
        else:
            final = self.llm(f"Answer truthfully:\nQ: {query}\nA:").strip()
            source = "llm"
            context = None

        return {
            "user":     query,
            "answer":   final,
            "source":   source,
            "context":  context,
            "score":    top_score,
            "filename": top_doc.metadata.get("filename", "")
        }

def format_chat_message(resp: Dict) -> str:
    lines = [
        f"**User:**\n{resp['user']}\n\n",
        f"**Assistant:**\n{resp['answer']}\n"
    ]

    if resp["source"] == "rag" and resp["score"] >= DISPLAY_THRESHOLD:
        pct = f"{resp['score']*100:.1f}%"
        lines += [
            "\n<details><summary>View Source chunk (" + resp["filename"] + ")</summary>\n",
            "```\n" + resp["context"] + "\n```\n",
            "</details>\n",
            f"\nRelevance: {pct}\n"
        ]
    else:
        lines.append("\n*Answer generated from model’s general knowledge*\n")

    return "".join(lines)

if __name__ == "__main__":
    DOC_PATH = "data/uploads/1706.03762v7.pdf"
    QUERY    = "what is self attention?"

    wrapper = DocumentQueryWrapper()
    if not wrapper.load_document(DOC_PATH):
        print(f"Error: failed to load {DOC_PATH}")
        exit(1)

    resp   = wrapper.query_document(QUERY)
    output = format_chat_message(resp)
    print(output)
