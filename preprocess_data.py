"""
Script to read .pdf files from a local 'data/processed_files' folder,
extract text, detect language, generate embeddings for English chunks,
and store them in ChromaDB.
Run this script AFTER running convert_msg_to_pdf.py.
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback # For detailed error logging

# Import language detection library
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Error: 'langdetect' library not found. Please install it: pip install langdetect")
    sys.exit(1)

# Import PDF text extraction library
try:
    import fitz # PyMuPDF
except ImportError:
    print("Error: 'PyMuPDF' library not found. Please install it: pip install PyMuPDF")
    sys.exit(1)


# Add src directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    # Import only necessary modules from src
    # from src import processing # No longer needed for PDF generation here
    from src import utils # Might need cleaning utils
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Ensure the 'src' directory exists and contains utils.py.")
    sys.exit(1)

# --- Configuration ---
try:
    # CHANGE: Point to the processed files directory
    DATA_FOLDER_PATH = "./data/processed_files"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "msg_pdf_content"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 150
    TARGET_LANGUAGE = 'en'
except Exception as e:
    print(f"Error setting up configuration: {e}")
    sys.exit(1)

# CHANGE: Function to extract text directly from PDF bytes
def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> str | None:
    """Extracts text content from PDF bytes using PyMuPDF."""
    if not pdf_bytes:
        return None
    try:
        # Open PDF from bytes stream
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n" # Add newline between pages
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  Error extracting text from PDF '{filename}': {e}")
        # print(traceback.format_exc()) # Optionally print full traceback
        return None


def main():
    print("--- Starting PDF Preprocessing with Language Filtering ---")

    # --- 1. Check if data folder exists ---
    if not os.path.isdir(DATA_FOLDER_PATH):
        print(f"Error: Processed data folder not found: '{os.path.abspath(DATA_FOLDER_PATH)}'")
        print("Please run the 'convert_msg_to_pdf.py' script first to generate PDF files.")
        return

    print(f"Reading .pdf files from local folder: '{os.path.abspath(DATA_FOLDER_PATH)}'")

    # --- 2. Initialize ChromaDB Client and Collection ---
    # (This part remains the same)
    print(f"Initializing ChromaDB client (persistent path: {CHROMA_DB_PATH})...")
    if not os.path.exists(CHROMA_DB_PATH):
        try:
            os.makedirs(CHROMA_DB_PATH)
        except OSError as e:
            print(f"Error creating ChromaDB directory '{CHROMA_DB_PATH}': {e}")
            return
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        print(f"Getting or creating collection: {COLLECTION_NAME}...")
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=sentence_transformer_ef)
        print("ChromaDB setup complete.")
    except Exception as e:
        print(f"Error initializing ChromaDB or embedding model: {e}")
        print(traceback.format_exc())
        return

    # --- 3. List, Read PDF, Extract Text, Chunk, Filter, Embed, and Store Files ---
    print("Processing PDF files...")
    processed_count = 0 # Counts files that contributed at least one English chunk
    skipped_files = 0   # Counts files skipped entirely due to errors or no text
    total_chunks_processed = 0
    english_chunks_added = 0
    other_lang_chunks_skipped = 0
    detection_errors = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

    try:
        all_files = os.listdir(DATA_FOLDER_PATH)
    except OSError as e:
        print(f"Error listing files in '{DATA_FOLDER_PATH}': {e}")
        return

    # CHANGE: Look for .pdf files
    pdf_filenames = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_filenames:
        print(f"No .pdf files found in '{DATA_FOLDER_PATH}'. Exiting.")
        return

    print(f"Found {len(pdf_filenames)} .pdf files.")

    for i, file_name in enumerate(pdf_filenames):
        print(f"\nProcessing {file_name} ({i+1}/{len(pdf_filenames)})...")
        file_path = os.path.join(DATA_FOLDER_PATH, file_name)
        file_processed_flag = False

        try:
            # Read PDF file content as bytes
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            if not pdf_bytes:
                 print(f"Skipping {file_name}: File is empty.")
                 skipped_files += 1
                 continue

            # CHANGE: Extract text directly from PDF bytes
            extracted_text = extract_text_from_pdf_bytes(pdf_bytes, file_name)
            if not extracted_text:
                print(f"Skipping {file_name}: Failed to extract text from PDF.")
                skipped_files += 1
                continue

            # Clean text (optional, might be less necessary after PDF extraction)
            cleaned_text = utils.clean_text(extracted_text)

            # Chunk text
            chunks = text_splitter.split_text(cleaned_text)
            if not chunks:
                print(f"Skipping {file_name}: No text chunks generated after splitting.")
                skipped_files += 1
                continue

            print(f"Generated {len(chunks)} chunks. Detecting language...")

            # Prepare lists for English chunks
            english_chunk_docs = []
            english_chunk_ids = []
            english_chunk_metadatas = []

            # Iterate through chunks, detect language, and filter
            for j, chunk in enumerate(chunks):
                total_chunks_processed += 1
                # Basic check for meaningful content before detection
                if len(chunk.strip()) < 10: # Skip very short/empty chunks
                    continue
                try:
                    # Detect language of the current chunk
                    lang = detect(chunk)
                    if lang == TARGET_LANGUAGE:
                        # Prepare data for this English chunk
                        # Use PDF filename for source reference
                        chunk_id = f"{file_name}_chunk_{j}" # Create a unique ID
                        metadata = {"source_file": file_name, "chunk_index": j, "language": lang}
                        english_chunk_docs.append(chunk)
                        english_chunk_ids.append(chunk_id)
                        english_chunk_metadatas.append(metadata)
                        # Don't increment english_chunks_added here yet, wait for batch add
                    else:
                        # Log skipped non-English chunk (optional)
                        # print(f"  Skipping chunk {j} (detected: {lang})")
                        other_lang_chunks_skipped += 1
                except LangDetectException:
                    # Handle cases where language detection fails
                    # print(f"  Skipping chunk {j} (language detection failed)")
                    detection_errors += 1
                except Exception as lang_e:
                    # Catch any other unexpected errors during detection for a chunk
                    print(f"  Skipping chunk {j} due to unexpected language detection error: {lang_e}")
                    detection_errors += 1

            # Add all collected English chunks from this file to ChromaDB if any were found
            if english_chunk_docs:
                try:
                    # Using upsert is safer if re-running the script to avoid duplicate ID errors
                    collection.upsert(
                        ids=english_chunk_ids,
                        documents=english_chunk_docs,
                        metadatas=english_chunk_metadatas
                    )
                    added_count = len(english_chunk_docs)
                    english_chunks_added += added_count # Increment total count
                    print(f"Successfully upserted {added_count} English chunks into ChromaDB.")
                    file_processed_flag = True # Mark file as having contributed data
                except Exception as db_e:
                    print(f"Error adding/upserting English chunks for {file_name} into ChromaDB: {db_e}")
                    print(traceback.format_exc())
                    # Consider how to handle partial failures if needed (e.g., retry logic)

            # Increment processed file count only if English chunks were successfully added
            if file_processed_flag:
                processed_count += 1
            elif not english_chunk_docs and chunks: # If file had chunks but none were English
                 print(f"No English chunks found to add for {file_name}.")
                 # Don't count as skipped if it was processed but had no English content
            # If chunks list was empty, it was already counted as skipped earlier


        except FileNotFoundError:
             print(f"Skipping {file_name}: File not found at path '{file_path}'.")
             skipped_files += 1
        except OSError as e:
             print(f"Skipping {file_name}: OS Error reading file: {e}")
             skipped_files += 1
        except Exception as e:
             print(f"Skipping {file_name}: An unexpected error occurred during processing: {e}")
             print(traceback.format_exc())
             skipped_files += 1


    print("\n--- PDF Preprocessing Summary ---")
    print(f"Attempted to process: {len(pdf_filenames)} PDF files")
    print(f"Files contributing English chunks to DB: {processed_count}")
    # Calculate skipped files more accurately
    files_without_english = len(pdf_filenames) - processed_count - skipped_files
    print(f"Files skipped (errors, empty, no English text): {skipped_files + files_without_english}")
    print(f"Total text chunks processed: {total_chunks_processed}")
    print(f"English chunks added/updated in collection '{COLLECTION_NAME}': {english_chunks_added}")
    print(f"Non-English chunks skipped: {other_lang_chunks_skipped}")
    print(f"Chunks skipped due to language detection errors: {detection_errors}")
    print(f"Database stored in: {os.path.abspath(CHROMA_DB_PATH)}")
    print("PDF preprocessing finished.")

if __name__ == "__main__":
    # Run this script directly using: python preprocess_local_folder.py
    # Ensure your environment has all necessary packages installed.
    main()
