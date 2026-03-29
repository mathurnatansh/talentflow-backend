import fitz  # PyMuPDF
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text from a given PDF file in bytes using PyMuPDF.
    """
    text = ""
    try:
        # Open PDF from memory
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""
