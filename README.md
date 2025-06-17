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

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
- **Tesseract OCR Engine**: This is required for extracting text from images.
    - Installation instructions: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    - Make sure to add Tesseract to your system's PATH or update the path in `utils.py` if necessary.

## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository-url>
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
