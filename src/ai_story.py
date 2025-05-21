"""Functions for generating stories section by section using a selectable local GGUF LLM via RAG."""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import re # For checking model names and cleaning <think> tags
import traceback # For detailed error logging

# LangChain imports
try:
    from langchain_community.llms import LlamaCpp
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except ImportError as e:
    st.error(f"Required libraries (langchain, langchain-community, llama-cpp-python, chromadb) not found: {e}")
    st.error("Please install dependencies from requirements.txt, including `pip install llama-cpp-python langchain-community`")
    class LlamaCpp: pass
    class PromptTemplate: pass
    class LLMChain: pass


# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "msg_pdf_content"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS_RETRIEVAL_PER_SECTION = 5

SIZE_TO_MAX_TOKENS_PER_SECTION = {
    "short": 350,
    "medium": 750,
    "long": 1500,
}
DEFAULT_MAX_TOKENS_PER_SECTION = 750
MAX_PAST_SECTIONS_IN_CONTEXT = 2

# SECTIONS_CONFIG (remains the same as your provided version)
SECTIONS_CONFIG = [
    {
        "id": "storyline_overview",
        "title": "# Storyline Overview",
        "query_hint": "overall context and narrative flow for the main story",
        "instructions": "Generate a 5-6 line paragraph that sets the scene and summarizes the main narrative flow. This overview should be based *only* on the provided documents for this section and the specified timeframe. Ensure the language is engaging and provides a clear introduction for the subsequent detailed sections. Avoid specific details that will be covered later; focus on the broad context."
    },
    {
        "id": "important_events",
        "title": "# Important Events",
        "query_hint": "significant occurrences, key activities, or milestones",
        "instructions": "List 3-4 distinct bullet points using `-` as the bullet character. Each bullet point must detail a specific event, including a date or timeframe if available (sourced from the document), and a concise description (e.g., `- [Actual Date from Document, Document Identifier] Description of the specific event sourced from the provided document, highlighting key details as found.).`). **Replace all bracketed placeholders like '[Actual Date from Document]' or '[Description of the specific event]' with actual information extracted from the provided documents.** Each bullet point must present unique information not repeated from other bullets within this section or from the 'Story So Far'. Strive for varied sentence structures and precise language."
    },
    {
        "id": "key_decisions",
        "title": "# Key Decisions",
        "query_hint": "critical choices, strategic shifts, or important policy enactments",
        "instructions": "List 3-4 distinct bullet points using `-`. Each must detail a specific decision, including the decision-maker if known (sourced from the document), the rationale if available (from the document), and its implications (from the document) (e.g., `- [Decision Maker from Doc, Document Source/Date] Description of the decision, its rationale if available in the document, and its noted implications, all based on provided texts.`). **Replace placeholders with factual details from documents.** Ensure information is unique and offers new insights compared to previous content. Employ diverse sentence construction and impactful vocabulary."
    },
    {
        "id": "major_breakthroughs",
        "title": "# Major Breakthroughs",
        "query_hint": "significant achievements, notable innovations, or successful strategic outcomes",
        "instructions": "List 3-4 distinct bullet points using `-`. Detail specific breakthroughs with dates/sources (from the document) and their impact (from the document) (e.g., `- [Source Document/Report, Date from Doc] Description of successful breakthrough/innovation, its demonstrated impact or efficiency gain, and how it paves the way for future developments, all based on provided texts.`). **Replace placeholders with concrete information from source documents.** Information must be unique and clearly articulated. Vary your phrasing to maintain reader engagement."
    },
    {
        "id": "challenges_and_solutions",
        "title": "# Challenges & Solutions",
        "query_hint": "obstacles encountered, problems identified, and the strategies or actions taken to overcome them",
        "instructions": "List 3-4 distinct bullet points using `-`. Describe specific challenges and the solutions implemented, referencing sources or timeframes (from the document) (e.g., `- [Source Document for Challenge, Date/Quarter from Doc] Identified specific challenge or bottleneck; [Source Document for Solution, Date/Quarter from Doc] Implemented specific strategy or solution to address it, detailing the outcome or mitigation achieved, all based on provided texts.`). **Replace placeholders with specific details from the documents.** Content must be distinct and solutions clearly linked to challenges. Use varied language and sentence structures."
    },
    {
        "id": "executive_summary",
        "title": "[Executive Summary]", # Note: Not an H1 heading
        "query_hint": "a holistic overview, synthesizing the main points, key developments, and overall implications from the entire narrative created so far",
        "instructions": "Generate a detailed summary (aim for 6-7 sentences, or more if necessary for a comprehensive overview). This summary must capture the big picture of the entire storyline developed across all previously generated sections and its strategic implications. Base this synthesis *only* on the 'Story So Far' and any high-level documents provided specifically for this summary section. Ensure it provides a coherent and conclusive perspective on the topic within the timeframe. Avoid introducing new, granular details not already covered; instead, focus on aggregation, synthesis, and impact. Conclude with a forward-looking statement if appropriate from the documents."
    }
]

# --- Global Variables for Loaded Resources (Cache using Streamlit) ---

@st.cache_resource(show_spinner="Loading Local GGUF Model...")
def load_gguf_llm(model_path: str, model_name_key: str):
    st.info(f"Attempting to load GGUF model: {model_name_key} from: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at path: {model_path}")
        st.error("Please ensure the path is correct and the model file exists.")
        return None
    try:
        initial_max_tokens = max(SIZE_TO_MAX_TOKENS_PER_SECTION.values()) if SIZE_TO_MAX_TOKENS_PER_SECTION else DEFAULT_MAX_TOKENS_PER_SECTION

        stop_sequences = ["<|endoftext|>", "<|im_end|>"] # Default to Qwen-like stop sequences
        if "qwen3" in model_name_key.lower():
            st.info(f"Using Qwen-specific stop sequences for {model_name_key}.")
        else:
            # Fallback/generic stop sequences if not Qwen3 (though currently only Qwen3 is targeted after removal)
            stop_sequences = ["\nNote:", "Note:", "\n*Note", "*Note", "<|eot_id|>"]
            st.info(f"Using generic stop sequences for {model_name_key}.")


        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=0,   # Using CPU as per user context
            n_batch=512,
            n_ctx=4096,
            f16_kv=True,
            verbose=True,
            temperature=0.2,
            max_tokens=initial_max_tokens,
            stop=stop_sequences,
        )
        model_n_ctx = llm.n_ctx if hasattr(llm, 'n_ctx') else 4096 # type: ignore
        st.success(f"GGUF Model loaded: {os.path.basename(model_path)} (Default max new tokens: {initial_max_tokens}, Context window: {model_n_ctx})")
        return llm
    except ImportError:
        st.error("`llama-cpp-python` library not installed or accessible. Please ensure it's installed correctly (`pip install llama-cpp-python`).")
        return None
    except Exception as e:
        st.error(f"Error loading GGUF model ({os.path.basename(model_path)}): {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_resource(show_spinner="Connecting to Vector Database...")
def get_chroma_retriever(db_path: str, collection_name: str, embedding_model_name: str):
    try:
        if not os.path.exists(db_path):
            st.error(f"ChromaDB path not found: {db_path}. Ensure preprocessing was successful.")
            return None
        chroma_client = chromadb.PersistentClient(path=db_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        collection = chroma_client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
        st.success(f"Successfully connected to ChromaDB collection '{collection_name}'.")
        return collection
    except ImportError as e:
        st.error(f"Libraries needed for ChromaDB/Embeddings (e.g., sentence-transformers) not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
        if "does not exist" in str(e).lower():
            st.error(f"Collection '{collection_name}' not found. Was the preprocessing script run to create and populate the DB?")
        return None

def get_prompt_for_single_section(
    section_id: str,
    section_title_for_prompt: str,
    section_specific_instructions_from_config: str,
    story_topic: str,
    timeframe: str,
    limited_cumulative_story_so_far: str,
    model_name_key: str
) -> str:

    system_instructions = f"""You are an expert analyst and storyteller. Your current task is to generate ONLY the content for the story section titled "**{section_title_for_prompt}**".
You must focus specifically on the given timeframe: **{timeframe}**.
The overall story topic is: **{story_topic}**.

**Before you begin writing the content for this section, take a moment to think step-by-step to construct your response. Your internal thought process should cover aspects like:**
1. Understanding the core objective of the '{section_title_for_prompt}' section based on its title and specific instructions.
2. Identifying the key pieces of information that need to be extracted or synthesized from the 'Input Documents for this Section'.
3. Planning the structure and format of your response according to the detailed section-specific instructions that will follow.
4. Ensuring all information presented is unique to this section, avoids repetition from the 'Story So Far', and is accurately sourced ONLY from the provided 'Input Documents for this Section'.
5. Crafting clear, engaging language with varied sentence structures.
6. Making sure the section concludes naturally and completely, fully addressing the section's purpose.
**Your step-by-step thinking is for your internal guidance only and should NOT appear in the final output. After this internal planning, generate ONLY the content for the section as requested.**

You will be provided with relevant documents under "Input Documents for this Section". Base your response for this section EXCLUSIVELY on these documents.
Critically, **do NOT repeat information** that is already present in the "Story So Far" (if provided below). Each piece of information in your response must be distinct and add new value to the narrative.
Strive for varied sentence structures and clear, concise language.
**IMPORTANT: Ensure your response for this section is thorough but also concludes naturally and completely. Avoid abrupt endings or unfinished thoughts. Aim to finish your points well within any typical length constraints for this type of section.**
Adhere strictly to all formatting and content instructions provided for this specific section."""

    if "digital systems" in story_topic.lower():
        system_instructions += f"""
**Specific Guidance for "Digital Systems" Topic (relevant if documents for this section mention it):**
If the input documents for this section discuss the "Digital Systems" division, ensure your response for "**{section_title_for_prompt}**" incorporates details about its key operational areas like ITM (Idea-to-Market), M2O (Market-to-Order/Sales), and O2D (Order-to-Delivery/Operations), as relevant to this section's focus (e.g., an ITM breakthrough, an M2O decision).
"""

    story_so_far_segment = ""
    if limited_cumulative_story_so_far.strip():
        story_so_far_segment = f"""
---
STORY SO FAR (for your context and to avoid repetition; do NOT regenerate these parts):
{limited_cumulative_story_so_far.strip()}
---
"""
    else:
        story_so_far_segment = "\nThis is the first section, so there is no 'Story So Far'.\n"

    user_instructions = f"""{story_so_far_segment}
Input Documents for this Section (use ONLY these for generating this section's content):
---
{{context}}  # This is where your RAG results will be injected
---

Now, generate ONLY the content for the section: "**{section_title_for_prompt}**".

**VERY IMPORTANT: Regarding the "specific instructions" that follow (from the configuration):**
The examples provided within those instructions (often starting with "e.g.," and containing bracketed text like "[Actual Date from Document...]") are STRICTLY for illustrating the **DESIRED FORMAT AND STYLE** of your answer.
**DO NOT COPY, REPLICATE, OR USE THE ACTUAL TEXT OR DATA FROM THESE EXAMPLES.**
Your response for this section must be based **EXCLUSIVELY** on the information found in the "Input Documents for this Section" provided above.
You MUST replace all placeholders (like '[Actual Date from Document]', '[Description of the specific event]', '[Decision Maker from Doc]', etc.) with actual, factual information extracted *only* from the "Input Documents for this Section".

Follow these specific instructions meticulously, paying close attention to the requested format but sourcing all content from the documents provided above:
{section_specific_instructions_from_config}

**CRITICAL REMINDERS FOR YOUR OUTPUT (Reiteration):**
- Your response must ONLY be the content for this specific section. **Do NOT include the section title (like "{section_title_for_prompt}") itself in your response.** It will be added separately.
- **Source ALL factual information (dates, names, decisions, events, projects, breakthroughs, challenges, solutions, etc.) EXCLUSIVELY from the "Input Documents for this Section" (the RAG context provided above).**
- The examples in the section guidelines (e.g., `[Actual Date from Document, Document Identifier] Description of the specific event...`) are for structure and style demonstration only. **DO NOT USE THE CONTENT of these examples in your output.** You MUST find equivalent information in the "Input Documents for this Section." If no such information exists for a particular detail, omit that detail or follow the section's specific instructions for missing information.
- Ensure all information provided is unique to this section, not repeating from "Story So Far" or within this section's own content. Use fresh phrasing.
- When instructed by the section guidelines to include document sources (e.g., "[Doc A, p2]", "[Internal Memo, 2023-10-15]"), these references must also be derived from your analysis of the "Input Documents for this Section," not copied from the examples.
- **Conclude your response for this section naturally and completely. Do not stop abruptly mid-sentence or mid-thought. Manage your response length to allow for a proper conclusion.**
"""

    if "qwen3" in model_name_key.lower():
        formatted_prompt_template = f"<|im_start|>system\n{system_instructions.strip()}<|im_end|>\n<|im_start|>user\n{user_instructions.strip()}<|im_end|>\n<|im_start|>assistant\n"
        st.info("Using Qwen3 prompt format.")
    else:
        st.warning(f"Unknown or unspecified model type '{model_name_key}' for prompt formatting. Using a generic ChatML-like approach. This may need adjustment for optimal performance with non-Qwen models.")
        formatted_prompt_template = f"<|im_start|>system\n{system_instructions.strip()}<|im_end|>\n<|im_start|>user\n{user_instructions.strip()}<|im_end|>\n<|im_start|>assistant\n"

    return formatted_prompt_template

# --- Main RAG Story Generation Function (Section by Section) ---
def generate_story_with_rag(story_topic: str, model_path: str, model_name_key: str, story_size: str, timeframe: str) -> str:
    st.info(f"Initiating section-by-section story generation for topic: '{story_topic}', Timeframe: '{timeframe}', Detail Level: '{story_size}'.")

    llm = load_gguf_llm(model_path, model_name_key)
    if not llm:
        return f"Error: Failed to load GGUF LLM from '{model_path}'. Story generation cannot proceed."

    model_n_ctx = llm.n_ctx if hasattr(llm, 'n_ctx') else 4096

    chroma_collection = get_chroma_retriever(CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME)
    if not chroma_collection:
        return "Error: Failed to connect to the vector database. Story generation cannot proceed."

    all_generated_section_outputs = []

    for section_index, section_config in enumerate(SECTIONS_CONFIG):
        section_id = section_config["id"]
        section_title_display = section_config["title"]
        section_query_hint = section_config["query_hint"]
        section_instructions_for_llm = section_config["instructions"]

        st.markdown(f"--- \n### Generating section {section_index + 1}/{len(SECTIONS_CONFIG)}: {section_title_display}")

        query_text_section = f"Detailed information for {section_query_hint} concerning '{story_topic}' (Digital Systems aspects if relevant) within the timeframe {timeframe}."
        st.write(f"Retrieving context for '{section_title_display}' (using {N_RESULTS_RETRIEVAL_PER_SECTION} document chunks)...")

        section_context_for_rag = ""
        try:
            retriever_results = chroma_collection.query(
                query_texts=[query_text_section], n_results=N_RESULTS_RETRIEVAL_PER_SECTION, include=['documents']
            )
            documents = retriever_results.get('documents')
            if not documents or not documents[0]:
                st.warning(f"No relevant documents found for section '{section_title_display}'.")
                section_context_for_rag = f"No specific context documents were found from the database for the section '{section_title_display}' regarding '{story_topic}' in timeframe '{timeframe}'. Please state that information for this section is unavailable or provide a general statement if appropriate."
            else:
                context_chunks = documents[0]
                section_context_for_rag = "\n\n---\n\n".join(context_chunks)
                st.success(f"Retrieved {len(context_chunks)} context chunks for '{section_title_display}'.")
        except Exception as e:
            st.error(f"Error retrieving documents for '{section_title_display}': {e}")
            all_generated_section_outputs.append(f"\n{section_title_display}\n- Error retrieving content due to DB issue: {str(e)[:100]}...\n")
            continue

        limited_cumulative_story_for_prompt = ""
        if section_id == "executive_summary":
            intro_phrase = "[Full preceding story context follows for summary generation:]\n"
            full_story_so_far_content = ""
            valid_past_sections = [s for s in all_generated_section_outputs if not s.lower().startswith(("- error", "- failed", "information for this section"))]
            if valid_past_sections:
                full_story_so_far_content = "\n".join(valid_past_sections)

            current_full_ssf_text = intro_phrase + full_story_so_far_content

            base_prompt_shell_str = get_prompt_for_single_section(
                section_id, section_title_display, section_instructions_for_llm,
                story_topic, timeframe, "", model_name_key
            )
            base_instructions_tokens = llm.get_num_tokens(
                PromptTemplate(template=base_prompt_shell_str, input_variables=["context"])
                .format_prompt(context="").to_string()
            )

            rag_context_tokens = llm.get_num_tokens(section_context_for_rag)
            safety_buffer_tokens = 150

            max_allowed_story_so_far_tokens = model_n_ctx - base_instructions_tokens - rag_context_tokens - safety_buffer_tokens

            current_story_so_far_tokens = llm.get_num_tokens(current_full_ssf_text)

            if max_allowed_story_so_far_tokens < 0:
                st.error(f"Executive Summary: Base prompt ({base_instructions_tokens} tokens) + RAG context ({rag_context_tokens} tokens) "
                         f"already exceed model context window ({model_n_ctx} tokens) even before adding 'Story So Far'. "
                         f"Cannot generate this section. Try reducing RAG results for summary or use a larger context model.")
                all_generated_section_outputs.append(f"\n{section_title_display}\n- Error: Input prompt too large for model context window before adding story so far.\n")
                continue

            if current_story_so_far_tokens > max_allowed_story_so_far_tokens:
                st.warning(
                    f"Executive Summary: Original 'Story So Far' ({current_story_so_far_tokens} tokens) is too large. "
                    f"Available for SSF: {max_allowed_story_so_far_tokens} tokens. Truncating oldest content..."
                )

                content_lines = full_story_so_far_content.splitlines()
                temp_ssf_text = intro_phrase + "\n".join(content_lines)

                while llm.get_num_tokens(temp_ssf_text) > max_allowed_story_so_far_tokens and content_lines:
                    content_lines.pop(0)
                    temp_ssf_text = intro_phrase + "\n".join(content_lines)

                if not content_lines and llm.get_num_tokens(intro_phrase) > max_allowed_story_so_far_tokens :
                    st.error("Executive Summary: After truncation, 'Story So Far' is empty but still too large with intro. Critical prompt issue.")
                    limited_cumulative_story_for_prompt = intro_phrase
                else:
                    limited_cumulative_story_for_prompt = temp_ssf_text

                final_ssf_tokens = llm.get_num_tokens(limited_cumulative_story_for_prompt)
                st.write(f"Executive Summary: Truncated 'Story So Far' to {final_ssf_tokens} tokens.")
            else:
                limited_cumulative_story_for_prompt = current_full_ssf_text

        elif all_generated_section_outputs:
            valid_past_sections = [s for s in all_generated_section_outputs if not s.lower().startswith(("- error", "- failed", "information for this section"))]
            start_index = max(0, len(valid_past_sections) - MAX_PAST_SECTIONS_IN_CONTEXT)
            relevant_past_parts = valid_past_sections[start_index:]
            if relevant_past_parts:
                temp_ssf = "\n".join(relevant_past_parts)
                limited_cumulative_story_for_prompt = f"[Context from last {len(relevant_past_parts)} section(s) follows:]\n" + temp_ssf

        section_prompt_template_str = get_prompt_for_single_section(
            section_id, section_title_display, section_instructions_for_llm,
            story_topic, timeframe, limited_cumulative_story_for_prompt, model_name_key
        )
        prompt_object = PromptTemplate(template=section_prompt_template_str, input_variables=["context"])

        num_input_tokens = 0
        final_prompt_text_for_llm = ""
        try:
            final_prompt_text_for_llm = prompt_object.format_prompt(context=section_context_for_rag).to_string()
            num_input_tokens = llm.get_num_tokens(final_prompt_text_for_llm)
            st.write(f"Final estimated input tokens for '{section_title_display}': {num_input_tokens} (Model Context Window: {model_n_ctx})")
            if num_input_tokens >= model_n_ctx:
                st.error(f"⛔ CRITICAL: Input tokens ({num_input_tokens}) for '{section_title_display}' MEET OR EXCEED model context window ({model_n_ctx}). LLM call will likely fail.")
            elif num_input_tokens >= model_n_ctx * 0.95:
                st.warning(f"⚠️ Input tokens ({num_input_tokens}) for '{section_title_display}' are close to model context window ({model_n_ctx}). This may cause issues.")
        except Exception as e:
            st.warning(f"Could not estimate final input tokens for {section_title_display}: {e}")
            st.write(f"Input prompt length (chars) for '{section_title_display}': {len(final_prompt_text_for_llm)}")


        if section_id == "executive_summary":
            target_max_tokens_for_section = SIZE_TO_MAX_TOKENS_PER_SECTION.get("long", DEFAULT_MAX_TOKENS_PER_SECTION * 2)
        else:
            target_max_tokens_for_section = SIZE_TO_MAX_TOKENS_PER_SECTION.get(story_size, DEFAULT_MAX_TOKENS_PER_SECTION)

        min_meaningful_buffer = max(20, int(target_max_tokens_for_section * 0.02))
        effective_max_tokens = target_max_tokens_for_section - min_meaningful_buffer
        if effective_max_tokens < 50:
            effective_max_tokens = max(20, int(target_max_tokens_for_section * 0.9))

        if hasattr(llm, 'max_tokens'):
            llm.max_tokens = effective_max_tokens
        st.write(f"Generating content for '{section_title_display}' (User target: {target_max_tokens_for_section}, Effective LLM limit: {effective_max_tokens} new output tokens)...")


        llm_chain = LLMChain(llm=llm, prompt=prompt_object)
        generated_content_final = ""
        try:
            if num_input_tokens >= model_n_ctx:
                raise ValueError(f"Pre-flight check failed: Estimated input tokens ({num_input_tokens}) exceed context window ({model_n_ctx}).")

            inputs_for_llm = {"context": section_context_for_rag}
            result = llm_chain.invoke(inputs_for_llm)
            generated_content_raw = result["text"].strip()

            try:
                num_output_tokens = llm.get_num_tokens(generated_content_raw)
                st.write(f"Generated output tokens for '{section_title_display}': {num_output_tokens} (Effective LLM limit: {effective_max_tokens})")
                if num_output_tokens >= effective_max_tokens - 5:
                    st.warning(
                        f"⚠️ Output for '{section_title_display}' ({num_output_tokens} tokens) used most of the effective LLM token limit ({effective_max_tokens}). "
                        f"It might be slightly condensed or very close to the hard stop. (User target was {target_max_tokens_for_section})."
                    )
            except Exception as e:
                st.warning(f"Could not estimate output tokens for {section_title_display}: {e}")

            # --- MODIFICATION START: Clean <think> tags ---
            # Remove <think>...</think> blocks first
            generated_content_cleaned = re.sub(r"<think>.*?</think>\s*", "", generated_content_raw, flags=re.DOTALL | re.IGNORECASE)
            generated_content_cleaned = generated_content_cleaned.strip()
            # --- MODIFICATION END ---


            # For Qwen, the model should stop at <|im_end|> or <|endoftext|> if these are in stop_sequences.
            # Further cleaning might be needed if it includes the stop token itself.
            if "qwen3" in model_name_key.lower():
                generated_content_cleaned = generated_content_cleaned.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
            # General cleaning for other models that might include specific end tokens in output despite stop sequences
            elif "<|eot_id|>" in generated_content_cleaned: # Check for generic stop token often used
                 generated_content_cleaned = generated_content_cleaned.split("<|eot_id|>")[0].strip()


            # General cleaning (already present)
            if generated_content_cleaned.startswith("[/INST]"): # Should not happen with ChatML but good fallback
                generated_content_cleaned = generated_content_cleaned[len("[/INST]"):].strip()

            core_title_text = re.sub(r"^[#\s]*\[?([^\]]+)\]?$", r"\1", section_title_display).strip()
            if generated_content_cleaned.lower().startswith(section_title_display.lower()):
                generated_content_cleaned = generated_content_cleaned[len(section_title_display):].lstrip(" :\n")
            elif core_title_text and generated_content_cleaned.lower().startswith(core_title_text.lower()):
                # Check if removing the core title leaves a very short or unchanged string, which might indicate
                # the model just outputted the title. Avoid removing if it's a substantial part of the beginning.
                potential_remainder = generated_content_cleaned[len(core_title_text):]
                # Only strip if it clearly looks like a title prefix (e.g., followed by colon, newline, or is very short)
                if potential_remainder.startswith(":") or potential_remainder.startswith("\n") or len(potential_remainder) < len(generated_content_cleaned) * 0.8 :
                    generated_content_cleaned = potential_remainder.lstrip(" :\n")


            if not generated_content_cleaned.strip():
                st.warning(f"Model returned empty or only whitespace content for '{section_title_display}' after cleaning. Placeholder used.")
                generated_content_final = f"Information for this section ('{section_title_display}') was not available in provided documents or generation resulted in empty content."
            else:
                generated_content_final = generated_content_cleaned
            st.success(f"Successfully generated content for section: {section_title_display}")

        except Exception as e:
            st.error(f"Error during LLM chain execution for '{section_title_display}': {e}")
            st.error(traceback.format_exc())
            generated_content_final = f"- Failed to generate content for this section due to an LLM error: {str(e)[:200]}...\n"

        newline_separator = "\n\n" if section_title_display.startswith("#") else "\n"
        current_section_full_output = f"{section_title_display}{newline_separator}{generated_content_final.strip()}\n"
        all_generated_section_outputs.append(current_section_full_output)

    final_story_assembled = "".join(all_generated_section_outputs).strip()
    acquisition_note = "Note: OSRAM's Digital Systems division was acquired by Inventronics in September 2023."

    if acquisition_note not in final_story_assembled:
        final_story_with_note = f"{final_story_assembled}\n\n{acquisition_note}"
    else:
        if not final_story_assembled.endswith(f"\n\n{acquisition_note}"):
            final_story_assembled = final_story_assembled.replace(acquisition_note, "").strip()
            final_story_with_note = f"{final_story_assembled}\n\n{acquisition_note}"
        else:
            final_story_with_note = final_story_assembled

    if not final_story_assembled.strip() or final_story_assembled.strip() == acquisition_note.strip():
        st.error("Complete story generation failed or resulted in no meaningful content beyond the standard note.")
        return f"Error: Story generation resulted in no content. \n\n{acquisition_note}"

    st.balloons()
    st.success("🎉 Story generation complete for all sections!")
    return final_story_with_note