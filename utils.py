import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# --- Configuration for Tesseract (if needed) ---
# Ensure Tesseract is installed and in your PATH.
# If not, you might need to specify its location, e.g.:
# For Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS (if installed via Homebrew):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' 
# Or for Linux:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def extract_text_from_image(image_path):
    """
    Extracts text from an image file using Tesseract OCR.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text, or an empty string if an error occurs.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not found in your PATH.")
        print("Please install Tesseract OCR and make sure it's added to your system's PATH.")
        print("For installation instructions, visit: https://tesseract-ocr.github.io/tessdoc/Installation.html")
        return "OCR_ERROR: Tesseract not found."
    except Exception as e:
        print(f"Error during OCR for image {image_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text, or an empty string if an error occurs.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") # "text" for plain text
        doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def process_document(file_path):
    """
    Processes a document (PDF, image) and extracts its text content.
    Args:
        file_path (str): Path to the document.
    Returns:
        str: Extracted text content, or None if the file type is unsupported or an error occurs.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    text_content = ""
    print(f"Processing document: {file_path} with extension {file_extension}")
    if file_extension == ".pdf":
        text_content = extract_text_from_pdf(file_path)
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        text_content = extract_text_from_image(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return None
        
    if text_content.strip() == "OCR_ERROR: Tesseract not found.":
        # Propagate Tesseract error to UI if possible, or handle appropriately
        return "Tesseract not found. Please ensure it is installed and in your PATH."
    elif not text_content.strip():
        print(f"Warning: No text extracted from {file_path}. The document might be empty or scanned without selectable text (for PDFs).")

    return text_content

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == '__main__':
    # This section is for testing utils.py directly.
    # You would need to create dummy files or point to existing ones.
    # Example:
    # UPLOAD_DIR_TEST = "test_docs" 
    # os.makedirs(UPLOAD_DIR_TEST, exist_ok=True)
    # test_pdf_path = os.path.join(UPLOAD_DIR_TEST, "sample.pdf") 
    # test_image_path = os.path.join(UPLOAD_DIR_TEST, "sample.png")

    # # Create a dummy PDF for testing (requires reportlab)
    # try:
    #     from reportlab.pdfgen import canvas
    #     c = canvas.Canvas(test_pdf_path)
    #     c.drawString(100, 750, "This is a test PDF document for an offline RAG system.")
    #     c.drawString(100, 730, "It contains multiple sentences and paragraphs to test text extraction and chunking.")
    #     c.save()
    #     print(f"Created dummy PDF: {test_pdf_path}")
    # except ImportError:
    #     print("reportlab not installed, skipping dummy PDF creation.")

    # # Create a dummy PNG for testing (requires Pillow)
    # try:
    #     from PIL import Image, ImageDraw, ImageFont
    #     img = Image.new('RGB', (600, 100), color = (255, 255, 255))
    #     d = ImageDraw.Draw(img)
    #     try:
    #         font = ImageFont.truetype("arial.ttf", 20)
    #     except IOError:
    #         font = ImageFont.load_default()
    #     d.text((10,10), "Test OCR text from a sample image.", fill=(0,0,0), font=font)
    #     img.save(test_image_path)
    #     print(f"Created dummy PNG: {test_image_path}")
    # except ImportError:
    #     print("Pillow not installed, skipping dummy image creation.")
    # except pytesseract.TesseractNotFoundError: # Catch this early for utils testing
    #     print("Tesseract not found during dummy file creation. OCR will fail.")


    # if os.path.exists(test_pdf_path):
    #     print(f"\n--- Testing PDF Extraction ({test_pdf_path}) ---")
    #     pdf_text = process_document(test_pdf_path)
    #     if pdf_text and "Tesseract not found" not in pdf_text:
    #         print(f"Extracted Text (first 300 chars):\n{pdf_text[:300]}")
    #         chunks = get_text_chunks(pdf_text)
    #         print(f"\nNumber of chunks: {len(chunks)}")
    #         if chunks:
    #             print(f"First chunk (first 100 chars):\n{chunks[0][:100]}")
    #     else:
    #         print(f"PDF processing result: {pdf_text}")


    # if os.path.exists(test_image_path):
    #     print(f"\n--- Testing Image OCR ({test_image_path}) ---")
    #     image_text = process_document(test_image_path)
    #     if image_text and "Tesseract not found" not in image_text:
    #         print(f"Extracted Text (first 300 chars):\n{image_text[:300]}")
    #         chunks = get_text_chunks(image_text)
    #         print(f"\nNumber of chunks: {len(chunks)}")
    #         if chunks:
    #             print(f"First chunk (first 100 chars):\n{chunks[0][:100]}")
    #     else:
    #         print(f"Image processing result: {image_text}")
    pass
