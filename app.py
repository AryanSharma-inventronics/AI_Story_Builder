import streamlit as st
import time
import random
import os
import json
from datetime import datetime

try:
    from src import ai_story
    from src import utils
except ImportError as e:
    st.error(f"Error importing necessary modules: {e}")
    st.error("Please ensure all files exist in the 'src' directory and required libraries are installed (`pip install -r requirements.txt`).")
    st.stop()

st.set_page_config(page_title="AI Storyteller Chatbot", layout="wide", initial_sidebar_state="expanded")

SUPPORTED_MODELS = [
    "Qwen3-8B-Q4_K_M-GGUF",
]

MODEL_PATHS = {
    "Qwen3-8B-Q4_K_M-GGUF": r"C:\Users\Aryan.Sharma\.lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf",
}
ACQUISITION_NOTE = "Note: OSRAM's Digital Systems division was acquired by Inventronics in September 2023."
HISTORY_FILE = "chat_history.json"

def load_history():
    """Loads story generation history from a JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Could not read history file, starting fresh.")
            return []
    return []

def save_history(history_list):
    """Saves story generation history to a JSON file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_list, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def add_to_history(topic, story_text):
    """Adds a new entry to the history."""
    history = st.session_state.get("story_history", [])
    new_entry = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "story": story_text
    }
    history.insert(0, new_entry)
    st.session_state.story_history = history
    save_history(history)

def get_chatbot_response(user_message):
    """Handles the chatbot conversation flow and triggers RAG actions."""
    current_pending_action = st.session_state.get("pending_action", None)
    chosen_model_name = st.session_state.get("story_model", SUPPORTED_MODELS[0] if SUPPORTED_MODELS else "N/A")
    model_path = MODEL_PATHS.get(chosen_model_name)

    if "create story about" in user_message.lower() or \
       "tell me a story about" in user_message.lower() or \
       (current_pending_action == "awaiting_topic" and user_message):

        if current_pending_action != "awaiting_topic":
            topic = user_message.split("about", 1)[-1].strip().rstrip('?.!')
            if not topic:
                st.session_state.pending_action = "awaiting_topic"
                return "Please specify what topic the story should be about (e.g., 'the Q3 planning')."
            st.session_state.story_topic = topic
        else:
            st.session_state.story_topic = user_message.strip().rstrip('?.!')

        if not st.session_state.story_topic:
            st.session_state.pending_action = "awaiting_topic"
            return "The topic seems empty. Please provide a clear topic for the story."

        st.session_state.pending_action = "ready_to_create"
        return f"Okay, a story about **'{st.session_state.story_topic}'** using **{chosen_model_name}**. Ready to generate?"

    elif user_message.lower() == "create story":
        st.session_state.pending_action = "awaiting_topic"
        return "Sure, what topic should the story be about?"

    elif current_pending_action == "ready_to_create":
        if any(w in user_message.lower() for w in ["yes", "ok", "generate", "ready", "proceed"]):
            topic = st.session_state.get("story_topic", None)

            if not chosen_model_name or chosen_model_name == "N/A" or not model_path:
                st.session_state.pending_action = None
                return f"Error: Model '{chosen_model_name}' not configured correctly with a valid path."
            if not topic:
                st.session_state.pending_action = None
                st.session_state.story_topic = None
                return "Error: Story topic is missing. Please start again."

            st.session_state.pending_action = "generating"
            final_story_raw = None
            with st.spinner(f"Generating story about '{topic}' using {chosen_model_name}... (This can take time)"):
                final_story_raw = ai_story.generate_story_with_rag(
                    story_topic=topic,
                    model_path=model_path,
                    model_name_key=chosen_model_name
                )

            if final_story_raw and not final_story_raw.startswith("Error:"):
                final_story_full = f"{final_story_raw}\n\n{ACQUISITION_NOTE}".strip()
                st.session_state.current_story_text = final_story_full
                st.session_state.current_story_topic = topic
                add_to_history(topic, final_story_full)
                st.session_state.pending_action = "awaiting_download_choice"
                st.session_state.story_topic = None
                return f"Here is the story about '{topic}':\n\n---\n\n{final_story_full}\n\n---\n\nWould you like to download this as a **Word** or **PDF** file? (Or say **No**)"
            else:
                st.session_state.pending_action = None
                st.session_state.story_topic = None
                return final_story_raw or f"Sorry, I couldn't generate the story about '{topic}'."

        else:
            st.session_state.pending_action = None
            st.session_state.story_topic = None
            return "Okay, cancelling story generation. Let me know if you change your mind."

    elif current_pending_action == "awaiting_download_choice":
        story_text = st.session_state.get("current_story_text")
        topic = st.session_state.get("current_story_topic", "story")
        chosen_model_name = st.session_state.get("story_model", "model")

        if not story_text:
             st.session_state.pending_action = None
             return "It seems I've lost the story text. Please try generating it again."

        chosen_format = None
        if "word" in user_message.lower(): chosen_format = "word"
        elif "pdf" in user_message.lower(): chosen_format = "pdf"
        elif any(w in user_message.lower() for w in ["no", "none", "cancel", "stop"]):
             st.session_state.pending_action = None
             st.session_state.current_story_text = None
             st.session_state.current_story_topic = None
             return "Okay, no download will be generated. What's next?"

        if chosen_format:
            st.session_state.pending_action = None
            file_bytes = None
            mime_type = None
            file_ext = None

            if chosen_format == "word":
                file_bytes = utils.generate_docx_bytes(story_text)
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                file_ext = "docx"
            else: # PDF
                file_bytes = utils.generate_pdf_bytes_from_text(story_text)
                mime_type = "application/pdf"
                file_ext = "pdf"

            if file_bytes:
                st.session_state.download_content = file_bytes
                safe_topic = "".join(c if c.isalnum() else "_" for c in topic[:30])
                safe_model_name = chosen_model_name.replace('/','-').replace('\\','-').replace(':','-')[:30]
                st.session_state.download_filename = f"story_{safe_topic}_{safe_model_name}.{file_ext}"
                st.session_state.download_mime = mime_type
                st.session_state.trigger_download = True
                st.session_state.current_story_text = None
                st.session_state.current_story_topic = None
                return f"Your {chosen_format.upper()} file is ready! Click the download button that appears below."
            else:
                st.session_state.current_story_text = None
                st.session_state.current_story_topic = None
                return f"Sorry, I failed to create the {chosen_format} file."
        else:
            return "Please choose **Word**, **PDF**, or **No**."

    elif any(word in user_message.lower() for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Greetings! Ask me to 'create story about [your topic]'."])
    elif "help" in user_message.lower():
        model_list_str = "\n - ".join(SUPPORTED_MODELS) if SUPPORTED_MODELS else "None configured"
        return (f"I can generate stories based on content from SharePoint emails using RAG. "
                f"Select the AI model using the dropdown above.\n"
                f"Then ask me to 'create story about [your topic]'.\n"
                f"I will generate the story in the chat, and then ask if you want a Word/PDF download.\n\n"
                f"Available local models:\n - {model_list_str}\n\n"
                f"Check the sidebar for past stories.\n\n"
                f"**Warning:** Local models require significant hardware and time. Ensure preprocessing is done.")
    else:
        if not current_pending_action:
            return "Sorry, I didn't understand. Try asking me to 'create story about [your topic]' or ask for 'help'."
        else:
            return "I'm waiting for your input. Could you try rephrasing or following the last prompt?"

st.title("✨✨ AI Storyteller Chatbot ✨✨")

st.sidebar.title("📜 Story History")
st.sidebar.caption("Recently generated stories. (Stored locally)")

if "story_history" not in st.session_state:
    st.session_state.story_history = load_history()

if not st.session_state.story_history:
    st.sidebar.info("No history yet. Generate a story to see it here.")
else:
    for i, entry in enumerate(st.session_state.story_history):
        ts = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M')
        with st.sidebar.expander(f"{ts} - {entry['topic']}"):
            st.markdown(entry['story'][:300] + "...")
            # Optional: Add a button to view the full story again or re-download
            # if st.button(f"View Full Story {i}", key=f"view_{i}"):
            #     st.session_state.messages.append({"role": "assistant", "content": entry['story']})
            #     st.rerun()

st.sidebar.divider()

if "story_model" not in st.session_state:
    st.session_state.story_model = SUPPORTED_MODELS[0] if SUPPORTED_MODELS else None

current_model_index = 0
if st.session_state.story_model and SUPPORTED_MODELS:
    try:
        current_model_index = SUPPORTED_MODELS.index(st.session_state.story_model)
    except ValueError:
        st.session_state.story_model = SUPPORTED_MODELS[0] if SUPPORTED_MODELS else None
        current_model_index = 0

if SUPPORTED_MODELS:
    st.selectbox(
        "Select Local AI Model (GGUF):",
        options=SUPPORTED_MODELS,
        index=current_model_index,
        key="story_model",
        help="Choose the local GGUF model. Loading may take time."
    )
    st.caption(f"Selected model: **{st.session_state.story_model}**")
else:
    st.error("No local models configured. Please add models to app.py.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Select a model and ask me to 'create story about [your topic]'."}]
if "pending_action" not in st.session_state: st.session_state.pending_action = None
if "story_topic" not in st.session_state: st.session_state.story_topic = None
if "current_story_text" not in st.session_state: st.session_state.current_story_text = None
if "current_story_topic" not in st.session_state: st.session_state.current_story_topic = None
if "trigger_download" not in st.session_state: st.session_state.trigger_download = False
if "assistant_responded" not in st.session_state: st.session_state.assistant_responded = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.get("trigger_download", False):
    st.download_button(
        label=f"Download Story ({os.path.splitext(st.session_state.get('download_filename', 'file.txt'))[-1].lstrip('.')})",
        data=st.session_state.get("download_content", b""),
        file_name=st.session_state.get("download_filename", "story.txt"),
        mime=st.session_state.get("download_mime", 'application/octet-stream'),
        key=f'download_button_key_{time.time()}',
        on_click=lambda: st.session_state.update(trigger_download=False, download_content=None, download_filename=None, download_mime=None)
    )

if prompt := st.chat_input("What would you like to do?", disabled=(not SUPPORTED_MODELS)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.assistant_responded = False
    st.session_state.trigger_download = False
    st.rerun()

latest_message = st.session_state.messages[-1] if st.session_state.messages else None

if latest_message and latest_message["role"] == "user" and not st.session_state.get("assistant_responded", False):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response_content = get_chatbot_response(latest_message["content"])
            st.markdown(assistant_response_content)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})
    st.session_state.assistant_responded = True

    if st.session_state.get("trigger_download"):
        st.rerun()