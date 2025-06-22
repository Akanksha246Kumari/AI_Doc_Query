# AI-Enabled Offline Document Query System

This project implements an offline document query system using Retrieval Augmented Generation (RAG), a reranker, file upload capabilities, and Optical Character Recognition (OCR).

## Features
- File upload (PDF, PNG, JPG)
- OCR for image-based documents
- Text extraction from PDFs
- Local embedding generation
- Local vector store for document chunks
- Retrieval of relevant document chunks
- Reranking of retrieved chunks for improved relevance
- Answer generation using a local Large Language Model (LLM)
- Offline operation

## 🛠 System Requirements

- Python 3.8+
- RAM: Minimum 16GB (32GB recommended)
- Storage: 10GB+ free space
- OS: Linux/macOS/Windows 

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Akanksha246Kumari/AI_Doc_Query.git
cd Doc_Query
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

#### macOS (using Homebrew)
```bash
brew install tesseract
```

#### Windows
1. Download Tesseract installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your system PATH

### 3. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Language Model

```bash
# Create model directory
mkdir -p model

# Download Mistral 7B model (4.37GB)
cd model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
cd ..
```

### 5. Run the Application

```bash
streamlit run app.py
```

Access the web interface at: http://localhost:8501

## 🏗 Project Structure

```
Doc_Query/
├── app.py              # Main Streamlit application
├── api.py             # Programmatic API and command-line interface
├── utils.py           # Core utilities (OCR, text processing)
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore rules
└── model/             # Local LLM storage (not in git)
    └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## 🤖 Usage

### Web Interface

1. **Upload Documents**:
   - Click "Upload Document"
   - Select PDF, PNG, JPG, or JPEG files
   - Wait for processing to complete

2. **Query Documents**:
   - Enter your question in the search box
   - Click "Search"
   - View results with source context

### API Usage

The application also provides a simple Python API for programmatic access:

```python
from api import DocumentQueryWrapper

# Initialize the wrapper
wrapper = DocumentQueryWrapper()

# Load a document
wrapper.load_document("path/to/your/document.pdf")  # or .png, .jpg, .jpeg

# Query the document
response = wrapper.query_document("Your question here")

# The response contains:
# - answer: The generated answer
# - source: 'rag' if from document, 'llm' if from model knowledge
# - context: The source chunk if available
print(f"Question: {response['user']}")
print(f"Answer: {response['answer']}")
if response.get('context'):
    print(f"\nSource Chunk:\n{response['context']}")
```

### Command Line Usage

You can also use the provided `api.py` script directly:

```bash
# Using default document and question
python3 api.py

# Or specify your own document and question
python3 api.py --document path/to/your/document.pdf --question "Your question here"
```

The script will output the question, answer, and source chunk (if available) in a clean format.

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration (optional):
```env
# Add any environment-specific settings here
```

### Model Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

## 🧪 Testing

### Verify Installation
```bash
python -c "
import streamlit, pytesseract, fitz, PIL, langchain, faiss, ctransformers;
print('✓ All dependencies installed successfully')
"
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   - Verify the model is in `model/` directory
   - Check file permissions

2. **Tesseract OCR Errors**
   - Ensure Tesseract is installed and in PATH
   - On Windows, verify the installation path is correct

3. **Memory Issues**
   - Close other memory-intensive applications
   - Consider using a machine with more RAM


## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone https://github.com/Akanksha246Kumari/AI_Doc_Query.git
    cd Doc-query
    ```

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Download a GGUF Language Model:**
    This application uses a local LLM for answer generation. You'll need to download a GGUF-compatible model.

2.  **Local LLM (GGUF Model)**

This project uses a local Large Language Model (LLM) in GGUF format for answer generation. For best performance, a model in the 7-billion-parameter range is recommended. You will need to download a model and place it in the project's root directory.

- **Recommended Model**: `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf`
  - This is the instruct-tuned version of Mistral 7B, offering a great balance of performance and resource requirements.
- **Download Link**: [**Download from Hugging Face (TheBloke)**](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)

**Instructions:**
1. Click the download link above to get the model file (it's about 4.37 GB).
2. Place the downloaded `.gguf` file into the newly created `model/` directory.
3. The application (`app2.py`) is now configured to look for this specific model file name by default.
    - Also, ensure the `model_type` in the `CTransformers` initialization in `app.py` (e.g., `'llama'`, `'mistral'`) matches your chosen model architecture.

## Running the Application

1.  **Ensure your virtual environment is activated:**
    ```bash
    # macOS/Linux
    source .venv/bin/activate
    
    # Windows
    .venv\Scripts\activate
    ```

2.  **Run the Streamlit App:**
    Navigate to the project's root directory (where `app.py` is located) and run:
    ```bash
    streamlit run app.py
    ```

3.  **Access the Application:**
    Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  Use the file uploader to select a document (PDF, PNG, JPG, JPEG).
2.  The system will process the document, extract text, create chunks, and build a vector store.
3.  Once processed, you can type your query into the text input box and press Enter.
4.  The system will retrieve relevant chunks, rerank them, and generate an answer using the local LLM.
