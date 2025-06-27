import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def extract_text_from_image(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except pytesseract.TesseractNotFoundError:
        return "OCR_ERROR: Tesseract not found."
    except Exception as e:
        print(f"OCR error on {image_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
        doc.close()
    except Exception as e:
        print(f"PDF extraction error on {pdf_path}: {e}")
    return text

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        raw = extract_text_from_image(file_path)
    else:
        print(f"Unsupported extension: {ext}")
        return ""
    if "OCR_ERROR" in raw:
        return raw
    return clean_text(raw)

def get_text_chunks_with_meta(text, filename, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_text(text)
    return [
        {"text": chunk, "filename": filename, "chunk_id": idx}
        for idx, chunk in enumerate(splits)
    ]
