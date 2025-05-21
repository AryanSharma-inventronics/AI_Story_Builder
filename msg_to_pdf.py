# convert_msg_to_pdf.py
"""
Reads .msg files from a source directory ('data/raw_files'),
extracts content (subject, body), generates a simple PDF for each,
and saves the PDF to a destination directory ('data/processed_files').
Handles cases where plain text or HTML body might be missing.
"""

import os
import sys
import traceback
from io import BytesIO
import re

# --- Dependencies ---
try:
    import extract_msg
except ImportError:
    print("Error: 'extract_msg' library not found. Please install it: pip install extract_msg")
    sys.exit(1)

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
except ImportError:
    print("Error: 'reportlab' library not found. Please install it: pip install reportlab")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: 'beautifulsoup4' not found (pip install beautifulsoup4). HTML body cleaning will be basic.")


# --- Configuration ---
SOURCE_DIR = "./data/raw_files"
DEST_DIR = "./data/processed_files"


def clean_html_basic(html_content: str) -> str:
    """Basic HTML tag removal if BeautifulSoup is not available."""
    import re
    # Remove tags
    text = re.sub('<[^>]+>', ' ', html_content)
    # Replace common HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_and_convert(msg_filepath: str, pdf_filepath: str):
    """Extracts content from a .msg file and saves it as a PDF."""
    print(f"  Processing: {os.path.basename(msg_filepath)}")
    subject = None
    body = None
    body_text = None
    body_html = None

    try:
        # Extract content using extract_msg
        msg = extract_msg.Message(msg_filepath)
        subject = msg.subject

        # --- Safely check for body content ---
        # Check for plain text body first
        if hasattr(msg, 'body') and msg.body:
            body_text = msg.body
            body = body_text # Use plain text if available
            # print("    Using plain text body.")
        # If no plain text body, check for HTML body
        elif hasattr(msg, 'html_body') and msg.html_body:
            body_html = msg.html_body
            # print("    Using HTML body.")
            if HAS_BS4:
                try:
                    soup = BeautifulSoup(body_html, 'html.parser')
                    body_parts = []
                    for element in soup.find_all(['p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']): # Added more block tags
                        text = element.get_text(strip=True)
                        if text:
                            # Add list item markers if needed
                            if element.name == 'li':
                                body_parts.append(f"* {text}") # Simple bullet
                            else:
                                body_parts.append(text)
                        # Add newline after block elements or breaks for structure
                        if element.name in ['p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                             body_parts.append("\n")
                    body = "\n".join(body_parts).strip()
                    body = re.sub(r'\n{3,}', '\n\n', body) # Collapse multiple newlines
                except Exception as parse_e:
                    print(f"    Warning: BeautifulSoup parsing failed ({parse_e}), using basic HTML cleaning.")
                    body = clean_html_basic(body_html)
            else:
                 body = clean_html_basic(body_html)
        else:
            # If neither body type has content
            print("    Warning: No usable body content (plain text or HTML) found.")
            body = "" # Use empty body

        # --- Handle Subject ---
        if not subject:
            print("    Warning: No subject found.")
            subject = "No Subject"

    except AttributeError as ae:
         print(f"  Attribute Error extracting content from {os.path.basename(msg_filepath)}: {ae}. This might indicate an unexpected .msg format.")
         return False # Indicate failure
    except Exception as e:
        print(f"  Error extracting content from {os.path.basename(msg_filepath)}: {e}")
        return False # Indicate failure

    # --- Generate PDF using reportlab ---
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # Add Subject
        subject_style = styles['h2']
        subject_style.alignment = TA_LEFT
        # Ensure subject is treated as a string
        story.append(Paragraph(f"Subject: {str(subject)}", subject_style))
        story.append(Spacer(1, 0.2*inch))

        # Add Body
        if body: # Only proceed if body has some content (even if empty string initially)
            body_style = styles['BodyText']
            # Ensure body is treated as a string before splitting
            body_paragraphs = str(body).split('\n')
            for para in body_paragraphs:
                para_stripped = para.strip()
                if para_stripped: # Add non-empty paragraphs
                    # Handle potential encoding issues during paragraph creation
                    try:
                        story.append(Paragraph(para_stripped, body_style))
                        story.append(Spacer(1, 0.1*inch))
                    except Exception as para_e:
                        print(f"    Warning: Skipping paragraph due to encoding or formatting error: {para_e}")
                        print(f"    Problematic text snippet (first 50 chars): {para_stripped[:50]}")


        # Check if anything was actually added to the story list
        if not story or len(story) <= 2: # Check if only subject was added
             print(f"    Skipping PDF generation for {os.path.basename(msg_filepath)}: No substantial content added.")
             return False # Treat as failure if no real content

        doc.build(story)

        # Write the PDF from buffer to file
        with open(pdf_filepath, 'wb') as f:
            f.write(pdf_buffer.getvalue())

        print(f"    Successfully converted to: {os.path.basename(pdf_filepath)}")
        return True

    except Exception as e:
        print(f"  Error generating PDF for {os.path.basename(msg_filepath)}: {e}")
        print(traceback.format_exc())
        return False


def main():
    print("--- Starting .msg to PDF Conversion ---")

    # --- Ensure directories exist ---
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory not found: '{os.path.abspath(SOURCE_DIR)}'")
        print("Please create 'data/raw_files' and place your .msg files inside.")
        return

    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR)
            print(f"Created destination directory: '{os.path.abspath(DEST_DIR)}'")
        except OSError as e:
            print(f"Error creating destination directory '{DEST_DIR}': {e}")
            return

    print(f"Source: '{os.path.abspath(SOURCE_DIR)}'")
    print(f"Destination: '{os.path.abspath(DEST_DIR)}'")

    # --- Process files ---
    conversion_count = 0
    error_count = 0

    try:
        for filename in os.listdir(SOURCE_DIR):
            if filename.lower().endswith(".msg"):
                source_filepath = os.path.join(SOURCE_DIR, filename)
                # Create corresponding PDF filename
                pdf_filename = os.path.splitext(filename)[0] + ".pdf"
                dest_filepath = os.path.join(DEST_DIR, pdf_filename)

                if extract_and_convert(source_filepath, dest_filepath):
                    conversion_count += 1
                else:
                    error_count += 1

    except OSError as e:
        print(f"Error reading source directory '{SOURCE_DIR}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        print(traceback.format_exc())
        return


    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {conversion_count} files")
    print(f"Errors encountered during conversion/extraction: {error_count} files")
    print(f"PDF files saved to: '{os.path.abspath(DEST_DIR)}'")
    print("Conversion process finished.")

if __name__ == "__main__":
    main()
