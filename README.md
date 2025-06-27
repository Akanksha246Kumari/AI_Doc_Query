# AI-Enabled Offline Document Query System

This project implements an offline document query system using Retrieval Augmented Generation (RAG), CrossEncoder-based reranking, local embeddings, OCR, and a local Large Language Model (LLM) for structured, conversational answers.

## ✨ Features
- Multi-file upload support (PDF, PNG, JPG, JPEG)
- OCR for image-based documents using Tesseract
- Text extraction from PDFs using PyMuPDF
- Local embedding generation (HuggingFace)
- FAISS vector store for document chunks
- Semantic retrieval + reranking (CrossEncoder)
- Final answer generation using local LLM (Mistral-7B)
- Display all matched chunks + highlight source chunk
- Fully offline operation

## 🛠 System Requirements
- Python 3.8+
- RAM: 16GB minimum (32GB recommended)
- Disk: 10GB+ free
- OS: Linux/macOS/Windows

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Akanksha246Kumari/AI_Doc_Query.git
cd AI_Doc_Query
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

#### macOS (Homebrew)
```bash
brew install tesseract
```

#### Windows
Download and install Tesseract OCR from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

### 3. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Language Model
```bash
mkdir -p model
cd model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
cd ..
```

### 5. Run the Application
```bash
streamlit run app.py
```
Then open: http://localhost:8501

## 🏗 Project Structure
```
AI_Doc_Query/
├── app.py              # Streamlit UI for document Q&A
├── api.py              # CLI + backend access via Python
├── utils.py            # Text extraction, cleaning, and chunking
├── requirements.txt    # Python dependencies
├── .gitignore          # Files to ignore
└── model/              # Local GGUF LLM
    └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## 🤖 API Usage
```python
from api import DocumentQueryWrapper
wrapper = DocumentQueryWrapper()
wrapper.load_document("your_file.pdf")
response = wrapper.query_document("your question")
print(response['answer'])
```

## 🖥 CLI Usage
```bash
python3 api.py
```

## 🔧 Configuration
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

## ✅ Testing
```bash
python -c "import streamlit, pytesseract, fitz, PIL, langchain, faiss, ctransformers; print('✓ All dependencies installed')"
```

## 🐛 Troubleshooting
- **OCR not working**: Check Tesseract installation.
- **Model not found**: Place `.gguf` file in `model/`.
- **Out of memory**: Restart system or reduce active load.
