# src/processing.py
"""Functions for processing .msg files into extractable text."""

import streamlit as st
from io import BytesIO

def extract_content_from_msg(msg_bytes: bytes) -> tuple[str | None, str | None]:
    """Extracts subject and body (prefer text) from .msg file bytes."""
    try:
        import extract_msg
    except ImportError:
        st.error("extract_msg library not installed. Cannot process .msg files.")
        return None, None

    try:
        # Create a file-like object from bytes
        msg_file = BytesIO(msg_bytes)
        msg = extract_msg.Message(msg_file)

        subject = msg.subject
        # Prioritize plain text body, fall back to HTML body if text is empty/None
        body = msg.body
        if not body:
            body = msg.html_body # May contain HTML tags

        # Basic cleaning if HTML was extracted (optional, can be improved)
        if body and ("<html" in body.lower() or "<body" in body.lower()):
             try:
                 from bs4 import BeautifulSoup
                 soup = BeautifulSoup(body, 'html.parser')
                 body = soup.get_text(separator='\n', strip=True)
             except ImportError:
                 st.warning("BeautifulSoup4 library not installed. HTML body might not be cleaned properly.")
             except Exception as e:
                 st.warning(f"Could not parse HTML body: {e}")


        return subject, body

    except Exception as e:
        st.error(f"Error processing .msg file content: {e}")
        return None, None

def generate_pdf_from_msg_content(subject: str | None, body: str | None) -> bytes | None:
    """Generates a simple PDF from extracted subject and body."""
    if not subject and not body:
        st.warning("No subject or body content to generate PDF from.")
        return None

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        st.error("reportlab library not installed. Cannot generate intermediate PDF.")
        return None

    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add Subject
        if subject:
            subject_style = styles['h2']
            subject_style.alignment = TA_LEFT
            story.append(Paragraph(f"Subject: {subject}", subject_style))
            story.append(Spacer(1, 12)) # Add space after subject

        # Add Body
        if body:
            body_style = styles['BodyText']
            # Handle potential multiple lines in the body
            body_paragraphs = body.split('\n')
            for para in body_paragraphs:
                 # Skip empty lines resulting from split
                 if para.strip():
                     story.append(Paragraph(para, body_style))
                     # Add a small space between paragraphs for readability, optional
                     story.append(Spacer(1, 6))

        if not story:
             st.warning("PDF generation skipped: No content added.")
             return None

        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    except Exception as e:
        st.error(f"Error generating intermediate PDF: {e}")
        return None


def extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extracts text content from PDF bytes using PyMuPDF."""
    if not pdf_bytes:
        return None
    try:
        import fitz  # PyMuPDF
    except ImportError:
        st.error("PyMuPDF library not installed. Cannot extract text from PDF.")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n" # Add newline between pages
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None


# --- Main Processing Function ---

def process_msg_content_to_text(msg_bytes: bytes) -> str | None:
    """
    Processes .msg file bytes through the pipeline:
    .msg -> Extract Content -> Generate PDF -> Extract Text from PDF
    """
    st.write("Processing .msg file...") # Provide feedback

    # 1. Extract content from .msg
    subject, body = extract_content_from_msg(msg_bytes)
    if body is None: # Body is the primary content needed
        st.warning("Could not extract body content from .msg file.")
        # Optionally return subject if only that is needed, or None
        return None

    # 2. Generate intermediate PDF (if needed - this step ensures cleaner text extraction)
    # If direct text extraction from msg body is reliable, this can be skipped.
    # However, PDF step often normalizes formatting.
    pdf_bytes = generate_pdf_from_msg_content(subject, body)
    if not pdf_bytes:
        st.warning("Could not generate intermediate PDF from .msg content.")
        # Fallback: try returning the extracted body directly? Needs cleaning.
        # from .utils import clean_text
        # return clean_text(body) if body else None
        return None # Fail if PDF step fails

    # 3. Extract text from the generated PDF
    extracted_text = extract_text_from_pdf(pdf_bytes)
    if not extracted_text:
        st.warning("Could not extract text from the intermediate PDF.")
        return None

    st.write("...Processing complete.")
    return extracted_text

