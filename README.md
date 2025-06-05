# AI Storyteller Chatbot from Your Emails

Welcome! This guide will help you set up and run the AI Storyteller project. This project reads your email files (.msg format), uses Artificial Intelligence (AI) to understand their content, and then lets you create stories or summaries based on that information through a chatbot.

## What This Project Does

Imagine you have a collection of email files (ending in `.msg`). This project will:
1.  Read these email files.
2.  Convert them into a format (PDFs) that's easier for the AI to work with. [cite: 1]
3.  Process these PDFs to extract the text and prepare it for an AI model. [cite: 1]
4.  Use an AI model that you'll download (specifically, `Qwen3-8B-Q4_K_M-GGUF` [cite: 2]) to understand the content. [cite: 2]
5.  Provide a chatbot interface where you can ask the AI to generate a story or summary based on the information found in your emails. [cite: 2]
6.  Allow you to download these generated stories as Word or PDF documents. [cite: 2]

**You don't need any prior coding experience to use this!** Just follow the steps below carefully.

## Before You Start: Software You Need

You'll need to install a few things on your computer. Some of these are programming tools, but don't worry, we'll guide you through it.

### 1. Python
Python is the programming language the project is written in.
* **Download**: Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
* **Install**: Download the latest stable version for your operating system (Windows, macOS).
    * **VERY IMPORTANT (for Windows users)**: During installation, make sure to check the box that says **"Add Python to PATH"** or **"Add python.exe to PATH"**. This is usually on the first screen of the installer.
    * Follow the on-screen instructions to complete the installation.

### 2. Visual Studio Code (VS Code) and C++ Build Tools

**VS Code (Text Editor)**:
VS Code is a powerful, free text editor that's excellent for viewing and editing code files like `app.py` (which you'll need to do later).
* **Download**: Go to the VS Code website: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
* **Install**: Download the version for your operating system and install it like any other application.

**C++ Build Tools (Compiler)**:
Some Python packages in this project (specifically `llama-cpp-python` which helps run the AI model [cite: 2]) need to be "built" or "compiled" during installation. This requires C++ build tools (essentially a C++ compiler).

* **Windows**:
    1.  Go to the Visual Studio downloads page: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
    2.  Scroll down to "All downloads" -> "Tools for Visual Studio".
    3.  Find "Build Tools for Visual Studio" and click Download.
    4.  Run the installer (`vs_buildtools.exe`).
    5.  A window will appear. In the "Workloads" tab, check the box for **"Desktop development with C++"**.
    6.  Click "Install" (usually in the bottom right). This might take some time.
    7.  Once installed, you might need to restart your computer.

* **macOS**:
    1.  Open the **Terminal** app (you can find it in `Applications > Utilities`).
    2.  Type the following command and press Enter:
        ```bash
        xcode-select --install
        ```
    3.  A dialog box will appear asking if you want to install the command line developer tools. Click "Install" and follow the prompts. If it says the software is already installed, you're all set.

* **Linux (Debian/Ubuntu based, e.g., Ubuntu, Mint)**:
    1.  Open your **Terminal**.
    2.  First, update your package list by typing:
        ```bash
        sudo apt update
        ```
        (You'll need to enter your user password for `sudo` commands.)
    3.  Then, install the build tools by typing:
        ```bash
        sudo apt install build-essential
        ```
    **Linux (Fedora based)**:
    1.  Open your **Terminal**.
    2.  Type:
        ```bash
        sudo dnf groupinstall "Development Tools" "C/C++ Development"
        ```

### 3. LM Studio
LM Studio is a program that makes it easy to download and run AI models on your own computer.
* **Download**: Go to the LM Studio website: [https://lmstudio.ai/](https://lmstudio.ai/)
* **Install**: Download the version for your operating system and install it like any other application.

### 4. The AI Model (Qwen3-8B-Q4_K_M-GGUF)
This is the specific AI brain the project uses. You'll download it using LM Studio.
* Open **LM Studio**.
* On the left sidebar, click the **Search icon** (magnifying glass).
* In the search bar at the top, type: `Qwen3-8B-GGUF`
* Look for a model named `Qwen3-8B-Q4_K_M.gguf`. It might be listed under a provider like "lmstudio-community".
    * **Important**: Ensure you pick the exact `Qwen3-8B-Q4_K_M.gguf` file[cite: 2]. There might be other versions; this is the one the project is configured for[cite: 2].
* Click the **Download** button next to the correct model file. This might take some time as the file is large.
* Wait for the download to complete. You'll need the location of this file later.

## Setting Up The Project: Step-by-Step

Now, let's get the project files and set them up.

### Step 1: Get the Project Code
* If you received this project as a ZIP file (e.g., `project.zip`), save it to your computer.
* If it's on a website like GitHub, look for a "Download ZIP" button (usually under a "Code" button).

### Step 2: Create a Project Folder
* Create a new folder on your computer where you want to keep the project. For example, `C:\AI_Story_Project` on Windows or `/Users/YourName/AI_Story_Project` on macOS.
* If you downloaded a ZIP file, extract all its contents into this new folder. You should see files like `app.py`, `requirements.txt`, etc., directly inside this folder.

### Step 3: Open a Command Line Tool in Your Project Folder
This is how you'll run the project's scripts.

* **Windows**:
    1.  Open your project folder in File Explorer.
    2.  In the address bar at the top of File Explorer (where it shows the folder path), type `cmd` and press Enter. This will open a black Command Prompt window directly in your project folder.
* **macOS**:
    1.  Open your project folder in Finder.
    2.  Go to Finder's menu: `Services` > `New Terminal at Folder`. (If you don't see this, you might need to enable it in `System Settings > Keyboard > Keyboard Shortcuts > Services > Files and Folders > New Terminal at Folder`).
    * Alternatively, open Terminal (from Applications > Utilities). Type `cd ` (note the space after `cd`), then drag your project folder from Finder into the Terminal window and press Enter.

You should now have a command line window open, and its current location should be your project folder.

### Step 4: Create a "Virtual Environment"
This creates an isolated space for this project's Python tools, so they don't interfere with other Python programs on your computer.
* In the command line window you opened in Step 3, type the following command and press Enter:
    ```bash
    python -m venv .venv
    ```
    (This tells Python to create a virtual environment named `.venv` inside your project folder.)

### Step 5: Activate the Virtual Environment
You need to "turn on" this environment before installing tools or running the project.
* **Windows**: In the command line, type this and press Enter:
    ```bash
    .venv\Scripts\activate
    ```
* **macOS/Linux**: In the command line, type this and press Enter:
    ```bash
    source .venv/bin/activate
    ```
* After running the command, you should see `(.venv)` at the beginning of your command line prompt. This means the virtual environment is active! (e.g., `(.venv) C:\AI_Story_Project>`)

### Step 6: Install Required Python Packages
These are the extra tools Python needs to run this specific project. The C++ Build Tools you installed earlier might be used during this step.
* Make sure your virtual environment is still active (you see `(.venv)`).
* In the command line, type this and press Enter:
    ```bash
    pip install -r requirements.txt
    ```
    (This tells `pip`, Python's package installer, to install all packages listed in the `requirements.txt` file[cite: 2]. This might take a few minutes, and you might see compilation messages if the C++ tools are being used.)

### Step 7: Prepare Your Email Files (.msg)
The project needs your email files to work with.
* Inside your main project folder, create a new folder named `data`.
* Inside the `data` folder, create another new folder named `raw_files`.
    * So the full path will be something like: `C:\AI_Story_Project\data\raw_files`
* **Copy all your `.msg` email files into this `data/raw_files` folder.** [cite: 1]

### Step 8: Important! Update the AI Model Path in `app.py`
The project needs to know exactly where you downloaded the `Qwen3-8B-Q4_K_M.gguf` AI model file using LM Studio.
* Open the `app.py` file (located in your main project folder). You can use **VS Code** (which you installed earlier) for this, or a simple text editor:
    * **VS Code**: Open VS Code, go to `File > Open Folder...` and select your project folder. Then, find `app.py` in the Explorer panel on the left and click to open it.
    * **Windows (alternative)**: Notepad (Search for "Notepad" in your Start Menu).
    * **macOS (alternative)**: TextEdit (Find it in Applications. **Important**: If using TextEdit, go to `Format` in the menu bar and select `Make Plain Text` before editing).
* Look for a section in `app.py` that looks like this (around line 17-19):
    ```python
    MODEL_PATHS = {
        "Qwen3-8B-Q4_K_M-GGUF": r"C:\Users\Aryan.Sharma\.lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf",
    }
    ```
* You need to **replace the example path** `r"C:\Users\Aryan.Sharma\.lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf"` [cite: 2] with the **actual path** to the `Qwen3-8B-Q4_K_M.gguf` file on your computer.

    **How to find the model path in LM Studio:**
    1.  Open **LM Studio**.
    2.  On the left sidebar, click the **My Models icon** (looks like a folder, usually the second icon from the top). This shows models you've downloaded.
    3.  Find your downloaded model, `Qwen3-8B-Q4_K_M.gguf`.
    4.  There should be an option to **Copy Path** (often a small copy icon next to the file name or path) or **Show in Finder/Explorer**. If you "Show in Explorer/Finder", you can then copy the path from your file explorer's address bar.
    5.  Once you have the path, paste it into `app.py`.

    **Path Formatting - VERY IMPORTANT:**
    * **Windows Users**: Your path will likely use backslashes (`\`). Make sure your pasted path is enclosed in `r"..."`. The `r` before the first quote is important.
        * Example: `r"C:\MyLMStudioModels\Qwen\Qwen3-8B-Q4_K_M.gguf"`
    * **macOS/Linux Users**: Your path will use forward slashes (`/`).
        * Example: `"/Users/YourName/.cache/lm-studio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"`
        (The `r` is not strictly needed for forward-slash paths but won't cause an error if it's there).

    * **Replace ONLY the path string inside the quotes.** Do NOT delete the `r` (if on Windows) or the surrounding quotes. Do NOT change the model key `"Qwen3-8B-Q4_K_M-GGUF":`.
* After replacing the path, **save the `app.py` file** and close the text editor.

## Running the Project: The Three Main Steps

You'll run three scripts in order. Make sure your virtual environment is still active (you see `(.venv)` in your command line prompt) and you are in your project folder.

### Step 1: Convert Emails to PDFs
This script reads your `.msg` files from `data/raw_files` [cite: 1] and converts them into PDF files, which are easier for the next step. The PDFs will be saved in a new folder called `data/processed_files` [cite: 1] (this folder will be created if it doesn't exist).
* In your command line window, type the following and press Enter:
    ```bash
    python msg_to_pdf.py
    ```
* Wait for it to finish. You'll see messages about which files are being processed. [cite: 1]

### Step 2: Prepare Data for the AI
This script reads the PDFs created in Step 1 from `data/processed_files`[cite: 1], extracts the text, and prepares it so the AI can understand and search through it quickly. It saves this prepared data in a new folder called `chroma_db` [cite: 1] (this folder will also be created). This step can take some time, especially if you have many emails.
* In your command line window, type the following and press Enter:
    ```bash
    python preprocess_data.py
    ```
* Wait for it to finish. You'll see messages about its progress. [cite: 1]

### Step 3: Start the AI Storyteller Chatbot
This starts the web application where you can chat with the AI and ask it to create stories.
* In your command line window, type the following and press Enter:
    ```bash
    streamlit run app.py
    ```
* After a few moments, this command should automatically open a new tab in your default web browser, showing the chatbot interface. If it doesn't, it will print a URL (like `Network URL: http://...` or `Local URL: http://localhost:8501`). You can copy and paste this URL into your web browser.
* **Keep the command line window open while you are using the chatbot.** Closing it will stop the application.

## How to Use the Chatbot

Once the chatbot interface is open in your web browser:
* **Select Model**: At the top, there might be a dropdown to select the AI model. It should already have `Qwen3-8B-Q4_K_M-GGUF` selected if it's the only one configured. [cite: 2]
* **Chat**: You'll see a chat interface. [cite: 2]
    * To start, you can type: `create story about [your topic]` (e.g., "create story about the Q3 project update"). [cite: 2]
    * The AI will respond, potentially asking for confirmation. [cite: 2]
    * If you confirm (e.g., by typing "yes"), it will generate a story based on the content from your emails. This can take some time as it's using the AI model running on your computer. [cite: 2]
* **Story Display**: The generated story will appear in the chat. [cite: 2]
* **Download**: After a story is generated, the chatbot will ask if you want to download it as a Word or PDF file. Respond with "Word", "PDF", or "No". If you choose to download, a download button should appear. [cite: 2]
* **History**: On the left sidebar, you can see a history of stories you've generated. [cite: 2]
* **Stopping the Chatbot**: To stop the chatbot application, go back to the command line window where you ran `streamlit run app.py` and press `Ctrl+C` (hold down the Ctrl key and press C).

## Troubleshooting Common Issues

* **`python` or `pip` is not recognized...**: This usually means Python was not added to your system's PATH during installation (Windows), or Python is not installed correctly. Reinstall Python, ensuring you check "Add Python to PATH".
* **ModuleNotFoundError: No module named 'xyz'**: This means a required Python package is missing.
    1.  Make sure your virtual environment is active (you see `(.venv)` in your prompt).
    2.  Try running `pip install -r requirements.txt` again. [cite: 2]
    3.  If it's a specific module, you can try `pip install xyz` (replacing `xyz` with the module name).
* **Errors during `pip install -r requirements.txt` (especially about C++ or compilation)**:
    1.  Ensure you have correctly installed the C++ Build Tools (Desktop development with C++ for Windows, Xcode command line tools for macOS, or build-essential/Development Tools for Linux) as described in "Software You Need - Step 2".
    2.  Try restarting your computer after installing the C++ Build Tools.
    3.  Then, activate your virtual environment and try `pip install -r requirements.txt` again.
* **Error loading GGUF model / Model file not found**:
    1.  Double-check the path you entered in `app.py` (Step 8 of Setup). Ensure it's the exact, correct path to your `.gguf` file. [cite: 2]
    2.  Make sure the model file (`Qwen3-8B-Q4_K_M.gguf`) was fully downloaded in LM Studio.
* **No files in `data/raw_files`**: Make sure you've copied your `.msg` files into the correct folder. [cite: 1]
* **Permission Denied errors**: This can happen if the script doesn't have rights to create folders (like `data/processed_files` or `chroma_db`) or write files. Try running your command line tool as an Administrator (Windows) or check folder permissions (macOS/Linux).

---

That's it! Hopefully, these instructions help you get the AI Storyteller up and running. Enjoy creating stories from your emails!
