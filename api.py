"""
Document Query API Wrapper

This module provides a simple interface for querying documents with natural language.
It replicates the exact output format of the Streamlit app for consistency.

Usage:
    # Initialize the wrapper
    from api import DocumentQueryWrapper
    
    # Create an instance
    query_wrapper = DocumentQueryWrapper()
    
    # Query a document
    result = query_wrapper.query_document(
        query="Your question here",
        document_path="path/to/document.pdf"  # or use document_content and filename
    )
    
    # Get the formatted output
    print(result)
"""

import os
import re
import logging
import numpy as np
import fitz  # PyMuPDF
from typing import Dict, Optional, Union, List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy

# Configure logging to suppress all output
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

# Constants
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class DocumentQueryWrapper:
    """Main class for querying documents with natural language."""
    
    def __init__(self):
        """Initialize the wrapper with lazy-loaded models."""
        self.embedding_model = None
        self.reranker = None
        self.llm = None
        self.vector_store = None
        self._initialize_models()
        
    def _initialize_models(self) -> bool:
        """
        Lazy load all required models.
        
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from sentence_transformers import CrossEncoder
            from langchain_community.llms import CTransformers
            
            if self.embedding_model is None:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
            if self.reranker is None:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                
            if self.llm is None:
                self.llm = CTransformers(
                    model="model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    model_type="mistral",
                    config={'max_new_tokens': 256, 'temperature': 0.01}
                )
            
            return True
            
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
            
    def _sigmoid(self, x):
        """Sigmoid function for score normalization."""
        return 1 / (1 + np.exp(-x))
        
    def _extract_text_from_pdf(self, file_path):
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
            
    def _extract_text_from_image(self, file_path):
        """Extract text from image using Tesseract OCR."""
        try:
            from PIL import Image
            import pytesseract
            return pytesseract.image_to_string(Image.open(file_path))
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _process_document(self, file_path):
        """Process a document and extract its text content."""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        text_content = ""
        if file_extension == ".pdf":
            text_content = self._extract_text_from_pdf(file_path)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            text_content = self._extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return self._clean_text(text_content) if text_content else ""
    
    def _get_text_chunks(self, text):
        """Split text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        return text_splitter.split_text(text)
    
    def load_document(self, file_path: str) -> bool:
        """
        Load and process a document for querying.
        
        Args:
            file_path: Path to the document file (PDF or image)
            
        Returns:
            bool: True if document was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            logger.info(f"Processing document: {file_path}")
            text = self._process_document(file_path)
            if not text:
                logger.error("No text could be extracted from the document")
                return False
                
            chunks = self._get_text_chunks(text)
            self.vector_store = FAISS.from_texts(
                chunks, 
                self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE
            )
            logger.info(f"Document processed successfully. Created {len(chunks)} chunks.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return False
    
    def query_document(self, query: str) -> Dict[str, str]:
        """
        Query the loaded document and get a response in the same format as the Streamlit app.
        
        Args:
            query: The question to ask about the document
            
        Returns:
            Dict: {
                'user': user_query,
                'answer': generated_answer,
                'source': 'rag' or 'llm',
                'context': source_context (if source is 'rag')
            }
        """
        if not self.vector_store:
            return {
                'user': query,
                'answer': 'No document is loaded. Please load a document first.',
                'source': 'llm',
                'context': None
            }
            
        try:
            # Get similar documents
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=5)
            
            # Rerank the results
            pairs = [[query, doc.page_content] for doc, _ in docs_and_scores]
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine similarity scores with reranker scores
            combined = []
            for (doc, dist), rerank_raw in zip(docs_and_scores, rerank_scores):
                orig_sim = 1.0 - dist
                try:
                    rerank_sim = self._sigmoid(rerank_raw)
                    if np.isnan(rerank_sim): 
                        rerank_sim = 0.0
                except:
                    rerank_sim = 0.0
                score = orig_sim * 0.3 + rerank_sim * 0.7
                combined.append((doc, score))
            
            # Sort by combined score
            combined = sorted(combined, key=lambda x: x[1], reverse=True)
            
            if not combined:
                return {
                    'user': query,
                    'answer': 'No relevant content found in document',
                    'source': 'llm',
                    'context': None
                }
            
            # Get top context for RAG answer
            top_context = combined[0][0].page_content if combined else ""
            
            # Generate RAG answer
            rag_prompt = f"""Answer the question using only the context below. Be brief.
Context:
{top_context}

Question: {query}
Answer:"""
            rag_answer = self.llm(rag_prompt).strip()
            
            # Generate LLM-only answer (fallback)
            llm_prompt = f"Answer this question truthfully.\nQuestion: {query}\nAnswer:"
            llm_answer = self.llm(llm_prompt).strip()
            
            # Determine which answer to use based on score threshold
            if combined and combined[0][1] > 0.6:  # Use RAG answer if score is above threshold
                best_answer = rag_answer
                source_info = "rag"
                context = top_context
            else:  # Fall back to LLM answer
                best_answer = llm_answer
                source_info = "llm"
                context = None
            
            return {
                'user': query,
                'answer': best_answer,
                'source': source_info,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error querying document: {e}")
            return {
                'user': query,
                'answer': f'An error occurred while processing your query: {str(e)}',
                'source': 'llm',
                'context': None
            }

def format_chat_message(response: Dict) -> str:
    """
    Format the response in the same way as the Streamlit app.
    
    Args:
        response: Dictionary containing user, answer, source, and context
        
    Returns:
        str: Formatted chat message
    """
    formatted = [
        f"**User:**\n{response['user']}\n\n",
        f"**Assistant:**\n{response['answer']}\n"
    ]
    
    if response['source'] == 'rag':
        formatted.append("\n<details><summary>View Source Chunk</summary>")
        formatted.append(f"\n```\n{response['context']}\n```")
        formatted.append("\n</details>")
    elif response['source'] == 'llm' and not response['context']:
        formatted.append("\n*Answer generated from model's general knowledge*")
    
    return "".join(formatted)

def suppress_output():
    """Suppress all output during processing."""
    import os
    import sys
    from contextlib import contextmanager, redirect_stdout, redirect_stderr
    
    # Set environment variable to handle OpenMP initialization
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    @contextmanager
    def suppress_stdout_stderr():
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                # Also suppress C/C++ output
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
                # Suppress warnings
                import warnings
                warnings.filterwarnings('ignore')
                # Suppress FAISS logs
                import faiss
                faiss.omp_set_num_threads(1)
                
                yield
    
    return suppress_stdout_stderr()

def main():
    """Run the document query with predefined document and question."""
    # Predefined values
    DOCUMENT_PATH = "data/uploads/1706.03762v7.pdf"
    QUESTION = "What is positional encoding?"
    
    # Store the final output
    output = []
    
    # Suppress all output during processing
    with suppress_output():
        try:
            # Initialize the wrapper
            wrapper = DocumentQueryWrapper()
            
            # Load the document
            if not wrapper.load_document(DOCUMENT_PATH):
                print("Error: Failed to load document")
                return
            
            # Process the query
            response = wrapper.query_document(QUESTION)
            
            # Prepare the output
            output.append(f"Question: {QUESTION}")
            output.append(f"Answer: {response['answer']}")
            
            # Add source chunk if available
            if response.get('context'):
                output.append("\nSource Chunk:")
                output.append(response['context'])
            
        except Exception as e:
            output.append(f"Error: {e}")
    
    # Print the final output
    print("\n".join(output))

if __name__ == "__main__":
    main()