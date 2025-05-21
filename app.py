import streamlit as st
import time
import random
import os # Needed for basename
import re # For validating year input

# --- Import functions from src modules ---
try:
    from src import ai_story # Now handles GGUF RAG logic with model-specific prompts
    from src import utils   # For file generation
    # NOTE: sharepoint and processing are used by the separate preprocess script
except ImportError as e:
    st.error(f"Error importing necessary modules: {e}")
    st.error("Please ensure all files exist in the 'src' directory and required libraries are installed (`pip install -r requirements.txt`).")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="AI Storyteller Chatbot", layout="wide")

# --- Constants ---
SUPPORTED_MODELS = [
    "Qwen3-8B-Q4_K_M-GGUF", # Only Qwen3 model remains
]

MODEL_PATHS = {
    # IMPORTANT: Replace with the ACTUAL, FULL path to your Qwen3-8B GGUF file
    "Qwen3-8B-Q4_K_M-GGUF": r"C:\Users\Aryan.Sharma\.lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf",
}

STORY_SIZES = ["short", "medium", "long"]

# --- Core Application Logic ---

def get_chatbot_response(user_message):
    """Handles the chatbot conversation flow and triggers RAG actions."""
    current_pending_action = st.session_state.get("pending_action", None)
    # chosen_model_name is the friendly name selected in the UI
    chosen_model_name = st.session_state.get("story_model", SUPPORTED_MODELS[0] if SUPPORTED_MODELS else "N/A")
    model_path = MODEL_PATHS.get(chosen_model_name) # This is the GGUF file path

    # --- Intent Handling ---
    # Step 1: Ask for topic
    if "create story about" in user_message.lower() or \
       "tell me a story about" in user_message.lower() or \
       (current_pending_action == "awaiting_topic" and user_message):

        if current_pending_action != "awaiting_topic":
            topic = user_message.split("about", 1)[-1].strip().rstrip('?.!')
            if not topic:
                return "Please specify what topic the story should be about (e.g., 'create story about the Q3 planning')."
            st.session_state.story_topic = topic
        else:
            st.session_state.story_topic = user_message.strip().rstrip('?.!')

        st.session_state.pending_action = "awaiting_timeframe"
        return f"Okay, a story about **'{st.session_state.story_topic}'**. Please specify the year or timeframe (after 2019) this story should focus on (e.g., '2020', 'late 2019', '2019-2021', 'Q1 2020')."

    elif user_message.lower() == "create story":
        st.session_state.pending_action = "awaiting_topic"
        return "Sure, what topic should the story be about?"

    # Step 2: Ask for timeframe (after topic is given)
    elif current_pending_action == "awaiting_timeframe":
        timeframe_input = user_message.strip()
        match = re.search(r'(20[1-9][0-9]|20[2-9][0-9]|[2-9][0-9]{3})', timeframe_input)
        if timeframe_input and match:
            year_found = int(match.group(0))
            if year_found >= 2019:
                st.session_state.story_timeframe = timeframe_input
                st.session_state.pending_action = "awaiting_size"
                return f"Got it, focusing on the timeframe **'{timeframe_input}'** for the story about **'{st.session_state.story_topic}'**. How long should the story be? ({', '.join(STORY_SIZES)})"
            else:
                return "Please provide a valid year (e.g., '2020'), range (e.g., '2019-2021'), or descriptive timeframe (e.g., 'late 2019', 'Q1 2020') including a year 2019 or later."
        else:
            return "Please provide a valid year (e.g., '2020'), range (e.g., '2019-2021'), or descriptive timeframe (e.g., 'late 2019', 'Q1 2020') after 2019."


    # Step 3: Ask for size (after topic and timeframe are given)
    elif current_pending_action == "awaiting_size":
        size_input = user_message.lower()
        selected_size = None
        for size_option in STORY_SIZES:
            if size_option in size_input:
                selected_size = size_option
                break
        if selected_size:
            st.session_state.story_size = selected_size
            st.session_state.pending_action = "awaiting_format"
            return f"Okay, a **{selected_size}** story about **'{st.session_state.story_topic}'** focusing on **'{st.session_state.story_timeframe}'**. How would you like the output? (View / Word / PDF)"
        else:
            return f"Please choose a story size: {', '.join(STORY_SIZES)}."

    # Step 4: Ask for format (after topic, timeframe, and size are given)
    elif current_pending_action == "awaiting_format":
        topic = st.session_state.get("story_topic", "the topic")
        size = st.session_state.get("story_size", "medium")
        timeframe = st.session_state.get("story_timeframe", "the specified period")
        chosen_format = None
        if any(fmt in user_message.lower() for fmt in ["view", "word", "pdf"]):
            if "word" in user_message.lower(): chosen_format = "word"
            elif "pdf" in user_message.lower(): chosen_format = "pdf"
            else: chosen_format = "view"

            st.session_state.story_format = chosen_format
            st.session_state.pending_action = "ready_to_create"
            return f"Okay: **{chosen_format}** format for a **{size}** story about **'{topic}'** during **'{timeframe}'** using **{chosen_model_name}**. Ready to generate?"
        else:
            return "Please choose a format: View, Word, or PDF."

    # Step 5: Generate story (after topic, timeframe, size, and format are confirmed)
    elif current_pending_action == "ready_to_create":
        if any(w in user_message.lower() for w in ["yes", "ok", "generate", "ready", "proceed"]):
            topic = st.session_state.get("story_topic", None)
            size = st.session_state.get("story_size", "medium")
            timeframe = st.session_state.get("story_timeframe", None)
            # model_path and chosen_model_name are derived at the start of the function

            if not chosen_model_name or chosen_model_name == "N/A" or not model_path:
                st.session_state.pending_action = None
                return f"Error: Model '{chosen_model_name}' not configured correctly with a valid path."
            if not topic:
                st.session_state.pending_action = None
                return "Error: Story topic is missing. Please start again."
            if not timeframe:
                st.session_state.pending_action = None
                return "Error: Story timeframe is missing. Please specify the timeframe again."

            st.session_state.pending_action = "generating"

            final_story = None
            with st.spinner(f"Generating {size} story about '{topic}' ({timeframe}) using {chosen_model_name}... (This can take time)"):
                final_story = ai_story.generate_story_with_rag(
                    story_topic=topic,
                    model_path=model_path,         # Pass the GGUF file path
                    model_name_key=chosen_model_name, # Pass the friendly model name for prompt formatting
                    story_size=size,
                    timeframe=timeframe
                )

            st.session_state.pending_action = None

            assistant_response_content = ""
            if final_story and not final_story.startswith("Error:"):
                output_format = st.session_state.get("story_format", "view")
                if output_format in ["word", "pdf"]:
                    if output_format == "word":
                        file_bytes = utils.generate_docx_bytes(final_story)
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        file_ext = "docx"
                    else: # PDF
                        file_bytes = utils.generate_pdf_bytes_from_text(final_story)
                        mime_type = "application/pdf"
                        file_ext = "pdf"

                    if file_bytes:
                        st.session_state.download_content = file_bytes
                        safe_topic = "".join(c if c.isalnum() else "_" for c in topic[:30])
                        safe_model_name = chosen_model_name.replace('/','-').replace('\\','-').replace(':','-')[:30]
                        st.session_state.download_filename = f"story_{safe_topic}_{safe_model_name}.{file_ext}"
                        st.session_state.download_mime = mime_type
                        st.session_state.trigger_download = True
                        assistant_response_content = f"Story about '{topic}' ({timeframe}) generated using {chosen_model_name}! Click the download button below."
                    else:
                        assistant_response_content = f"Story text about '{topic}' ({timeframe}) was generated, but failed to create the {output_format} file for download."
                else: # View mode
                    assistant_response_content = final_story
            else:
                if not final_story:
                    assistant_response_content = f"Sorry, I couldn't generate the story about '{topic}' ({timeframe}) due to an unexpected error."
                else:
                    assistant_response_content = final_story # Pass error message from generation

            return assistant_response_content
        else:
            st.session_state.pending_action = None
            st.session_state.story_topic = None
            st.session_state.story_size = None
            st.session_state.story_timeframe = None
            return "Okay, cancelling story generation. Let me know if you change your mind."

    # --- General Conversation ---
    elif any(word in user_message.lower() for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Greetings! Ask me to 'create story about [your topic]'."])
    elif "help" in user_message.lower():
        model_list_str = "\n - ".join(SUPPORTED_MODELS) if SUPPORTED_MODELS else "None configured"
        return (f"I can generate stories based on content from SharePoint emails using RAG. "
                f"Select the AI model using the dropdown above.\n"
                f"Then ask me to 'create story about [your topic]'.\n"
                f"I will ask for the timeframe (e.g., '2020', 'late 2019', '2019-2021') and desired story length ({', '.join(STORY_SIZES)}), then the output format (View/Word/PDF).\n\n"
                f"Available local models:\n - {model_list_str}\n\n"
                f"**Warning:** Local models require significant hardware (RAM/VRAM) and time to load. Ensure the preprocessing script has been run.")
    else:
        return "Sorry, I didn't understand. Try asking me to 'create story about [your topic]' or ask for 'help'."


# --- Streamlit App UI ---

st.title("✨✨ AI Storyteller Chatbot ✨✨")

if "story_model" not in st.session_state:
    st.session_state.story_model = SUPPORTED_MODELS[0] if SUPPORTED_MODELS else None

current_model_index = 0
if st.session_state.story_model and SUPPORTED_MODELS:
    try:
        current_model_index = SUPPORTED_MODELS.index(st.session_state.story_model)
    except ValueError:
        if SUPPORTED_MODELS:
            st.session_state.story_model = SUPPORTED_MODELS[0]
            current_model_index = 0
        else:
            st.session_state.story_model = None # Should not happen if SUPPORTED_MODELS has items

if SUPPORTED_MODELS:
    st.selectbox(
        "Select Local AI Model (GGUF):",
        options=SUPPORTED_MODELS, # These are the friendly names
        index=current_model_index,
        key="story_model", # This session state key holds the selected friendly model name
        help="Choose the local GGUF model to load and use. Loading may take time."
    )
    st.caption(f"Selected model: **{st.session_state.story_model}**")
else:
    st.error("No local models configured in SUPPORTED_MODELS list in app.py. Please add at least one GGUF model configuration.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Select a model and ask me to 'create story about [your topic]'."}]
if "pending_action" not in st.session_state: st.session_state.pending_action = None
if "story_format" not in st.session_state: st.session_state.story_format = None
if "story_topic" not in st.session_state: st.session_state.story_topic = None
if "story_size" not in st.session_state: st.session_state.story_size = None
if "story_timeframe" not in st.session_state: st.session_state.story_timeframe = None
if "trigger_download" not in st.session_state: st.session_state.trigger_download = False
if "assistant_responded" not in st.session_state: st.session_state.assistant_responded = False


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.get("trigger_download", False):
    st.download_button(
        label=f"Download Story ({st.session_state.get('story_format')})",
        data=st.session_state.get("download_content", b""),
        file_name=st.session_state.get("download_filename", "story.txt"),
        mime=st.session_state.get("download_mime", 'application/octet-stream'),
        key=f'download_button_key_{time.time()}', # Unique key for re-trigger
        on_click=lambda: st.session_state.update(trigger_download=False, download_content=None, download_filename=None, download_mime=None)
    )

if prompt := st.chat_input("What would you like to do?", disabled=(not SUPPORTED_MODELS)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.assistant_responded = False # Reset flag for new user input
    st.rerun()

latest_message = st.session_state.messages[-1] if st.session_state.messages else None

if latest_message and latest_message["role"] == "user" and not st.session_state.get("assistant_responded", False):
    with st.chat_message("assistant"): # Display assistant's placeholder while generating
        with st.spinner("Thinking..."):
            assistant_response_content = get_chatbot_response(latest_message["content"])
            st.markdown(assistant_response_content) # Display the actual response

    st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})
    st.session_state.assistant_responded = True # Mark that assistant has responded

    # Rerun if a download button needs to appear, or if state changes affect UI immediately
    if st.session_state.get("trigger_download"):
        st.rerun()