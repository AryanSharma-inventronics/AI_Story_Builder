# AI Storyteller Chatbot (Local GGUF RAG Version)

This project implements a Streamlit chatbot that generates stories based on the content of `.msg` email files. It uses a Retrieval-Augmented Generation (RAG) approach with locally stored data and locally run GGUF language models via `llama-cpp-python`.

## Features

* **Chatbot Interface:** Interact with the AI through a Streamlit chat interface.
* **Local Data Processing:** Reads `.msg` files from a local `data` folder.
* **Language Filtering:** Preprocessing step filters and stores only English text chunks.
* **Vector Database:** Uses ChromaDB to store and retrieve relevant text chunks based on user topics.
* **Local LLM Execution:** Runs selected GGUF models (e.g., Mistral, DeepSeek) locally using `llama-cpp-python` for story generation.
* **Selectable Models:** Allows choosing between configured local GGUF models via a dropdown.
* **Customizable Story:** Ask for stories on specific topics found within the emails and specify desired length (short, medium, long).
* **Structured Output:** Prompts the model to generate stories following a defined company history structure.
* **Downloadable Output:** Option to download the generated story as a Word (.docx) or PDF file.

## Project Structure

.├── .streamlit/│   └── secrets.toml      # Optional: For future secrets (not currently used for local models)├── data/                 # <--- IMPORTANT: Place your .msg files here! (Create this folder)├── chroma_db/            # Directory where the vector database will be stored (created by preprocessing)├── src/                  # Source code modules│   ├── init.py│   ├── ai_story.py       # Handles LLM loading, RAG, and generation│   ├── processing.py     # .msg file processing and text extraction│   ├── utils.py          # Utility functions (e.g., document generation)│   └── ...               # (Potentially sharepoint.py if you re-add that functionality)├── app.py                # Main Streamlit application script├── preprocess_local_folder.py # Script to process local .msg files into ChromaDB├── preprocess_sharepoint.py # (Optional: If SharePoint functionality is re-added)└── requirements.txt      # Python dependencies
## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Create Python Environment:** (Recommended)
    ```bash
    python -m venv story_generator
    # Activate the environment
    # Windows:
    .\story_generator\Scripts\activate
    # macOS/Linux:
    source story_generator/bin/activate
    ```

3.  **Install Dependencies:**
    * Ensure you have Python installed (version 3.9+ recommended).
    * Install necessary C++ build tools if needed for `llama-cpp-python`. Check the [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for specific requirements for your OS.
    * Install Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    * **Note on `llama-cpp-python`:** Installation might require specific flags depending on whether you want CPU or GPU (CUDA/Metal) support. Refer to its documentation for optimized installation. The default `pip install llama-cpp-python` often builds for CPU.

4.  **Download GGUF Models:**
    * Download the GGUF model files you want to use (e.g., from Hugging Face). A convenient tool for finding and managing models is [LM Studio](https://lmstudio.ai/). You can search for and download models like `Mistral Instruct GGUF` or `DeepSeek Qwen GGUF` within LM Studio.
    * The current configuration in `app.py` expects:
        * `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
        * `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`
    * Note the location where LM Studio (or you) saves the downloaded `.gguf` files (often within a hidden `.cache` or `.lmstudio` folder in your user directory).
    * **Crucially, update the file paths in the `MODEL_PATHS` dictionary inside `app.py` to match the exact location where you saved the `.gguf` files on your system.**

5.  **Prepare Data:**
    * Create a folder named `data` in the root directory of the project (the same level as `app.py`).
    * **Copy all your `.msg` email files into this newly created `data` folder.** The preprocessing script will read from here.

## Running the Project

There are two main steps: Preprocessing the data and running the application.

**Step 1: Preprocess Local `.msg` Files**

* This step reads the `.msg` files from the `data` folder, extracts English text, chunks it, generates embeddings, and stores them in a local ChromaDB database (`./chroma_db`).
* **You only need to run this once initially**, or whenever you add/change the `.msg` files in the `data` folder.
* Run the script from your terminal (ensure your virtual environment is active):
    ```bash
    python preprocess_local_folder.py
    ```
* This process might take some time depending on the number and size of your `.msg` files. It will create the `./chroma_db` folder.

**Step 2: Run the Streamlit Application**

* Once preprocessing is complete, run the main chatbot application:
    ```bash
    streamlit run app.py
    ```
* This will start the Streamlit server and should open the application in your web browser automatically.
* The first time you select a specific model from the dropdown, it will be loaded into memory (this can take time and significant RAM/VRAM). Subsequent uses of the *same* model within the same session should be faster due to caching. Switching models will trigger a reload.

## Hardware Considerations

* **CPU:** Running LLMs locally on CPU is possible but **very slow**, especially for larger models like DeepSeek or Mistral 7B. Expect long generation times.
* **RAM:** Loading these models requires a significant amount of RAM. 16GB might be barely sufficient for smaller quantized models; 32GB+ is strongly recommended, especially for larger models or if running on CPU. Monitor your RAM usage.
* **GPU (Recommended):** For acceptable performance, a powerful NVIDIA GPU with ample VRAM (e.g., 8GB+, ideally 16GB or more for 7B models) is highly recommended. Ensure you install `llama-cpp-python` with the correct GPU support flags (e.g., CUDA/cuBLAS).

## Configuration

* **Model Paths:** Update the file paths in the `MODEL_PATHS` dictionary in `app.py` to point to your downloaded GGUF files.
* **Preprocessing:** You can adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `EMBEDDING_MODEL_NAME` in `preprocess_local_folder.py` if needed. Ensure `EMBEDDING_MODEL_NAME` matches in both preprocessing and `ai_story.py` if changed.
* **LLM Parameters:** Parameters like `n_ctx` (context window), `n_gpu_layers` (GPU offloading), `temperature`, `max_tokens` can be adjusted within the `LlamaCpp(...)` initialization in `src/ai_story.py`.
