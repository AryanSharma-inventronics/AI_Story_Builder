import streamlit as st
import time
import random
import os # Needed for basename
# REMOVED: re import as year validation is removed

# --- Import functions from src modules ---
try:
    from src import ai_story # Now handles GGUF RAG logic with model-specific prompts
    from src import utils   # For file generation
except ImportError as e:
    st.error(f"Error importing necessary modules: {e}")
    st.error("Please ensure all files exist in the 'src' directory and required libraries are installed (`pip install -r requirements.txt`).")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="AI Storyteller Chatbot", layout="wide")

# --- Constants ---
SUPPORTED_MODELS = [
    "Qwen3-8B-Q4_K_M-GGUF", 
]

MODEL_PATHS = {
    "Qwen3-8B-Q4_K_M-GGUF": r"C:\Users\Aryan.Sharma\.lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf",
}

# REMOVED: STORY_SIZES

# --- Core Application Logic ---

def get_chatbot_response(user_message):
    """Handles the chatbot conversation flow and triggers RAG actions."""
    current_pending_action = st.session_state.get("pending_action", None)
    chosen_model_name = st.session_state.get("story_model", SUPPORTED_MODELS[0] if SUPPORTED_MODELS else "N/A")
    model_path = MODEL_PATHS.get(chosen_model_name)

    # --- Intent Handling ---
    # Step 1: Ask for topic OR process topic if given
    if "create story about" in user_message.lower() or \
       "tell me a story about" in user_message.lower() or \
       (current_pending_action == "awaiting_topic" and user_message):

        if current_pending_action != "awaiting_topic":
            topic = user_message.split("about", 1)[-1].strip().rstrip('?.!')
            if not topic: # If "create story about" was empty
                st.session_state.pending_action = "awaiting_topic" # Re-prompt for topic
                return "Please specify what topic the story should be about (e.g., 'the Q3 planning')."
            st.session_state.story_topic = topic
        else: # Was awaiting_topic and user provided something
            st.session_state.story_topic = user_message.strip().rstrip('?.!')
        
        if not st.session_state.story_topic: # Double check if topic is empty after processing
            st.session_state.pending_action = "awaiting_topic"
            return "The topic seems empty. Please provide a clear topic for the story."

        # MODIFIED: Transition directly to awaiting_format
        st.session_state.pending_action = "awaiting_format"
        return f"Okay, a story about **'{st.session_state.story_topic}'**. How would you like the output? (View / Word / PDF)"

    elif user_message.lower() == "create story":
        st.session_state.pending_action = "awaiting_topic"
        return "Sure, what topic should the story be about?"

    # MODIFIED: Removed "awaiting_timeframe" and "awaiting_size" blocks

    # Step 2 (was 4): Ask for format (after topic is given)
    elif current_pending_action == "awaiting_format":
        topic = st.session_state.get("story_topic", "the topic")
        chosen_format = None
        if any(fmt in user_message.lower() for fmt in ["view", "word", "pdf"]):
            if "word" in user_message.lower(): chosen_format = "word"
            elif "pdf" in user_message.lower(): chosen_format = "pdf"
            else: chosen_format = "view" # Default to view if "view" or other keywords present

            st.session_state.story_format = chosen_format
            st.session_state.pending_action = "ready_to_create"
            # MODIFIED: Message updated (removed size and timeframe)
            return f"Okay: **{chosen_format}** format for a story about **'{topic}'** using **{chosen_model_name}**. The story will use the maximum possible length per section. Ready to generate?"
        else:
            return "Please choose a format: View, Word, or PDF."

    # Step 3 (was 5): Generate story (after topic and format are confirmed)
    elif current_pending_action == "ready_to_create":
        if any(w in user_message.lower() for w in ["yes", "ok", "generate", "ready", "proceed"]):
            topic = st.session_state.get("story_topic", None)
            # MODIFIED: size and timeframe are no longer used from session_state here

            if not chosen_model_name or chosen_model_name == "N/A" or not model_path:
                st.session_state.pending_action = None
                return f"Error: Model '{chosen_model_name}' not configured correctly with a valid path."
            if not topic:
                st.session_state.pending_action = None # Reset
                st.session_state.story_topic = None
                return "Error: Story topic is missing. Please start again by asking to 'create story about [your topic]'."
            
            st.session_state.pending_action = "generating"
            final_story = None
            # MODIFIED: Spinner message updated
            with st.spinner(f"Generating story about '{topic}' using {chosen_model_name}... (This can take time as it aims for max length per section)"):
                # MODIFIED: Call to generate_story_with_rag updated
                final_story = ai_story.generate_story_with_rag(
                    story_topic=topic,
                    model_path=model_path,
                    model_name_key=chosen_model_name
                )

            st.session_state.pending_action = None # Reset after generation attempt

            assistant_response_content = ""
            if final_story and not final_story.startswith("Error:"):
                output_format = st.session_state.get("story_format", "view")
                # MODIFIED: Response messages updated
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
                        assistant_response_content = f"Story about '{topic}' generated using {chosen_model_name}! Click the download button below."
                    else:
                        assistant_response_content = f"Story text about '{topic}' was generated, but failed to create the {output_format} file for download."
                else: # View mode
                    assistant_response_content = final_story
            else:
                if not final_story: # Should ideally not happen if generate_story_with_rag returns error string
                    assistant_response_content = f"Sorry, I couldn't generate the story about '{topic}' due to an unexpected error."
                else: # Pass error message from generation
                    assistant_response_content = final_story 
            
            # Clear story-specific state after generation or error
            st.session_state.story_topic = None
            st.session_state.story_format = None
            # REMOVED: story_size, story_timeframe from session state clearing
            return assistant_response_content
        else: # User said "no" or similar to "Ready to generate?"
            st.session_state.pending_action = None
            st.session_state.story_topic = None
            st.session_state.story_format = None
            # REMOVED: story_size, story_timeframe from session state clearing
            return "Okay, cancelling story generation. Let me know if you change your mind. You can say 'create story about [your topic]'."

    # --- General Conversation ---
    elif any(word in user_message.lower() for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Greetings! Ask me to 'create story about [your topic]'."])
    elif "help" in user_message.lower():
        model_list_str = "\n - ".join(SUPPORTED_MODELS) if SUPPORTED_MODELS else "None configured"
        # MODIFIED: Help message updated
        return (f"I can generate stories based on content from SharePoint emails using RAG. "
                f"Select the AI model using the dropdown above.\n"
                f"Then ask me to 'create story about [your topic]'.\n"
                f"I will then ask for the output format (View/Word/PDF) and generate the story aiming for maximum length per section.\n\n"
                f"Available local models:\n - {model_list_str}\n\n"
                f"**Warning:** Local models require significant hardware (RAM/VRAM) and time to load. Ensure the preprocessing script has been run.")
    else:
        # If no pending action and not a recognized command, guide the user.
        if not current_pending_action:
             return "Sorry, I didn't understand. Try asking me to 'create story about [your topic]' or ask for 'help'."
        else: # If there's a pending action, the specific state should handle the response.
             # This path should ideally not be hit if states are managed well.
             return "I'm a bit confused. Could you try rephrasing or starting over with 'create story about [your topic]'?"


# --- Streamlit App UI ---
st.title("✨✨ AI Storyteller Chatbot ✨✨")

if "story_model" not in st.session_state:
    st.session_state.story_model = SUPPORTED_MODELS[0] if SUPPORTED_MODELS else None

current_model_index = 0
if st.session_state.story_model and SUPPORTED_MODELS: # Ensure SUPPORTED_MODELS is not empty
    try:
        current_model_index = SUPPORTED_MODELS.index(st.session_state.story_model)
    except ValueError: # If saved model is no longer in the list
        if SUPPORTED_MODELS: # Check again, just in case
            st.session_state.story_model = SUPPORTED_MODELS[0]
            current_model_index = 0
        else: # Should not happen if initial check passed
             st.session_state.story_model = None 

if SUPPORTED_MODELS:
    st.selectbox(
        "Select Local AI Model (GGUF):",
        options=SUPPORTED_MODELS, 
        index=current_model_index,
        key="story_model", 
        help="Choose the local GGUF model to load and use. Loading may take time."
    )
    st.caption(f"Selected model: **{st.session_state.story_model}**")
else:
    st.error("No local models configured in SUPPORTED_MODELS list in app.py. Please add at least one GGUF model configuration.")

st.divider()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Select a model and ask me to 'create story about [your topic]'."}]
if "pending_action" not in st.session_state: st.session_state.pending_action = None
if "story_topic" not in st.session_state: st.session_state.story_topic = None
if "story_format" not in st.session_state: st.session_state.story_format = None
# REMOVED: story_size, story_timeframe from session state init
if "trigger_download" not in st.session_state: st.session_state.trigger_download = False
if "assistant_responded" not in st.session_state: st.session_state.assistant_responded = False


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle download button display
if st.session_state.get("trigger_download", False):
    st.download_button(
        label=f"Download Story ({st.session_state.get('story_format', 'file')})", # Fallback label
        data=st.session_state.get("download_content", b""),
        file_name=st.session_state.get("download_filename", "story.txt"),
        mime=st.session_state.get("download_mime", 'application/octet-stream'),
        key=f'download_button_key_{time.time()}', 
        on_click=lambda: st.session_state.update(trigger_download=False, download_content=None, download_filename=None, download_mime=None)
    )
    # Ensure the download button is only shown once until a new download is triggered
    # The on_click already sets trigger_download to False. If it's still true, means it wasn't clicked.


# Handle user input and responses
if prompt := st.chat_input("What would you like to do?", disabled=(not SUPPORTED_MODELS)): # Disable input if no models
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.assistant_responded = False 
    st.rerun() # Rerun to display user message immediately

latest_message = st.session_state.messages[-1] if st.session_state.messages else None

if latest_message and latest_message["role"] == "user" and not st.session_state.get("assistant_responded", False):
    with st.chat_message("assistant"): 
        with st.spinner("Thinking..."):
            assistant_response_content = get_chatbot_response(latest_message["content"])
            st.markdown(assistant_response_content) 

    st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})
    st.session_state.assistant_responded = True 

    if st.session_state.get("trigger_download"):
        st.rerun() # Rerun to make download button appear immediately after response