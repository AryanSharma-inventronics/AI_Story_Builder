import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import traceback

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

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "msg_pdf_content"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS_RETRIEVAL_PER_SECTION = 5
MAX_PAST_SECTIONS_IN_CONTEXT = 3

MAX_OUTPUT_TOKENS_REGULAR_SECTION = 1500
DESIRED_MAX_OUTPUT_TOKENS_SUMMARY = 2500

SECTIONS_CONFIG = [
    {
        "id": "storyline_overview",
        "title": "# Storyline Overview",
        "query_hint": "overall context and narrative flow for the main story",
        "instructions": "Generate a 5-6 line paragraph that sets the scene and summarizes the main narrative flow. This overview should be based *only* on the provided documents for this section. Ensure the language is engaging and provides a clear introduction. Avoid specific details. **Crucially: Ensure this overview is concise and does NOT repeat information from other parts of the story. Do NOT output any placeholder text or incomplete thoughts.**"
    },
    {
        "id": "important_events",
        "title": "# Important Events",
        "query_hint": "significant occurrences, key activities, or milestones",
        "instructions": "List 3-4 distinct bullet points using `-`. **Each bullet point MUST detail a UNIQUE and specific event not mentioned elsewhere.** Attempt to source information for each bullet point from different parts of the provided documents. Be concise. Include a date/timeframe and a description (e.g., `- [Actual Date, Document Identifier] Description of the specific event.).`). **Replace ALL bracketed placeholders with actual information from the documents. If you cannot find 3-4 unique events, provide as many as you *can* find without repeating or inventing information.** Do NOT repeat events."
    },
    {
        "id": "key_decisions",
        "title": "# Key Decisions",
        "query_hint": "critical choices, strategic shifts, or important policy enactments",
        "instructions": "List 3-4 distinct bullet points using `-`. **Each MUST detail a UNIQUE decision, its rationale, and implications.** Diversify sources. (e.g., `- [Decision Maker, Document Source/Date] Description, rationale, and implications.`). **ABSOLUTELY DO NOT use placeholder text; replace ALL placeholders with actual information or describe what is missing. If you cannot find 3-4 unique decisions, provide as many as you *can* find without repeating or inventing.** Do NOT repeat decisions."
    },
    {
        "id": "major_breakthroughs",
        "title": "# Major Breakthroughs",
        "query_hint": "significant achievements, notable innovations, or successful strategic outcomes",
        "instructions": "List 3-4 distinct bullet points using `-`. **Detail UNIQUE breakthroughs with dates/sources and their unique impact.** Draw from different document segments. (e.g., `- [Source Document, Date] Description, impact, and future implications.`). **Replace placeholders with concrete information. If you cannot find 3-4 unique breakthroughs, provide as many as you *can* find without repeating or inventing.** Do NOT repeat breakthroughs or impacts."
    },
    {
        "id": "challenges_and_solutions",
        "title": "# Challenges & Solutions",
        "query_hint": "obstacles encountered, problems identified, and the strategies or actions taken to overcome them",
        "instructions": "List 3-4 distinct bullet points using `-`. **Describe UNIQUE challenges and their distinct solutions.** Reference sources/timeframes. (e.g., `- [Source for Challenge, Date] Identified challenge; [Source for Solution, Date] Implemented solution and outcome.`). **Replace placeholders. If you cannot find 3-4 unique pairs, provide as many as you *can* find without repeating or inventing.** Do NOT repeat challenge/solution pairs."
    },
    {
        "id": "executive_summary",
        "title": "[Executive Summary]",
        "query_hint": "a holistic overview, synthesizing the main points, key developments, and overall implications from the entire narrative created so far",
        "instructions": "Generate a detailed and comprehensive summary paragraph. **This summary MUST CAPTURE THE BIG PICTURE and its strategic implications. Base this synthesis *only* on the 'Story So Far'. Critically, you MUST REPHRASE and SYNTHESIZE. Do NOT simply copy sentences or list bullet points. Create a NEW, flowing narrative that connects key themes and outcomes. Offer insights, not repetition.** Avoid new details. Conclude with a *single*, concise forward-looking statement if appropriate. **Do NOT add extra section titles. Do NOT repeat yourself. Generate a coherent, well-written paragraph. Do NOT output any placeholder text, meta-comments, or think tags.**"
    }
]

@st.cache_resource(show_spinner=False)
def load_gguf_llm(model_path: str, model_name_key: str):
    st.info(f"Attempting to load GGUF model: {model_name_key} from: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at path: {model_path}")
        return None
    try:
        default_init_max_tokens = 4096
        stop_sequences = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>"]
        st.info(f"Using stop sequences: {stop_sequences}")

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=0,
            n_batch=512,
            n_ctx=32768,
            f16_kv=True,
            verbose=False,
            temperature=0.6,
            max_tokens=default_init_max_tokens,
            stop=stop_sequences,
        )
        model_n_ctx = llm.n_ctx if hasattr(llm, 'n_ctx') else 32768
        st.success(f"GGUF Model loaded: {os.path.basename(model_path)} (Context window: {model_n_ctx})")
        return llm
    except ImportError:
        st.error("`llama-cpp-python` library not installed. Please `pip install llama-cpp-python`.")
        return None
    except Exception as e:
        st.error(f"Error loading GGUF model ({os.path.basename(model_path)}): {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_resource(show_spinner=False)
def get_chroma_retriever(db_path: str, collection_name: str, embedding_model_name: str):
    try:
        if not os.path.exists(db_path):
            st.error(f"ChromaDB path not found: {db_path}.")
            return None
        chroma_client = chromadb.PersistentClient(path=db_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        collection = chroma_client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
        st.success(f"Connected to ChromaDB collection '{collection_name}'.")
        return collection
    except ImportError as e:
        st.error(f"Libraries needed for ChromaDB/Embeddings not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
        return None

def get_prompt_for_single_section(
    section_id: str,
    section_title_for_prompt: str,
    section_specific_instructions_from_config: str,
    story_topic: str,
    limited_cumulative_story_so_far: str,
    model_name_key: str
) -> str:
    system_instructions = f"""You are an expert analyst and storyteller. Your ONLY task is to generate the content for the story section: "**{section_title_for_prompt}**".
The overall story topic is: **{story_topic}**.

**CRITICAL RULES:**
1.  **ONLY output the text for the section.** Do NOT include your thinking process, meta-comments, explanations, or any tags like `<think>`, `</think>`, or similar.
2.  **Base your response EXCLUSIVELY on the "Input Documents for this Section" provided.**
3.  **Do NOT repeat information** already present in the "Story So Far" (unless it's the Executive Summary, where you must *synthesize*, not copy).
4.  **Do NOT use placeholder text** (like `[Actual Date...]`). Replace it with *real data* from the documents or omit it if unavailable.
5.  **Follow the specific instructions** for this section meticulously.
6.  **Write naturally and ensure the section concludes properly.** Avoid abrupt endings.
7.  **NEVER, under any circumstances, output `<think>` or `</think>` tags in your final response.**

**Internal Thought Process (DO NOT OUTPUT):**
    - *Understand the goal of '{section_title_for_prompt}'.*
    - *Identify key info in 'Input Documents' for '{story_topic}' & '{section_title_for_prompt}'.*
    - *Plan structure & format based on instructions.*
    - *Ensure uniqueness & accuracy.*
    - *Craft clear language.*
    - *Ensure a complete section.*
**--- END OF INTERNAL THOUGHTS ---**
"""

    if "digital systems" in story_topic.lower():
        system_instructions += f"""
**Specific Guidance for "Digital Systems" Topic:** If documents mention ITM, M2O, or O2D, integrate these details appropriately for "**{section_title_for_prompt}**"."""

    story_so_far_segment = ""
    if limited_cumulative_story_so_far.strip():
        story_so_far_segment = f"""
---
STORY SO FAR (for context to AVOID repetition; DO NOT regenerate these):
{limited_cumulative_story_so_far.strip()}
---
"""
    else:
        story_so_far_segment = "\nThis is the first section.\n"

    user_instructions = f"""{story_so_far_segment}
Input Documents for this Section (use ONLY these):
---
{{context}}
---

Now, generate ONLY the content for the section: "**{section_title_for_prompt}**".

**VERY IMPORTANT:** The examples in the instructions below (e.g., `- [Actual Date...]`) are for **FORMAT ONLY**. **DO NOT COPY THEIR TEXT.** Use *only* information from the "Input Documents" above. Replace ALL placeholders.

Follow these specific instructions:
{section_specific_instructions_from_config}

**FINAL REMINDER:** Output ONLY the clean text for this section. No titles, no tags, no placeholders, no meta-comments.
"""

    formatted_prompt_template = f"<|im_start|>system\n{system_instructions.strip()}<|im_end|>\n<|im_start|>user\n{user_instructions.strip()}<|im_end|>\n<|im_start|>assistant\n"
    return formatted_prompt_template

def clean_llm_output(raw_text: str, section_title: str) -> str:
    """Cleans raw LLM output, removing common artifacts."""
    text = raw_text.strip()

    if text.lower().startswith("assistant\n"):
        text = text[len("assistant\n"):].lstrip()

    text = re.sub(r"<think\b[^>]*>.*?</think\s*>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>\s*", "", text, flags=re.IGNORECASE).strip()

    if re.match(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        st.warning(f"Section '{section_title}': Detected an unclosed/malformed `<think>` block *even after cleaning*. This likely means severe truncation. Clearing content.")
        return ""

    stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    for stop in stop_tokens:
        if stop in text:
            text = text.split(stop)[0].strip()

    if text.lower().startswith("[/inst]"):
        text = text[len("[/inst]"):].lstrip()

    core_title = re.sub(r"^[#\s]*\[?([^#\]\[]+)\]?$", r"\1", section_title).strip()
    if core_title:
        title_pattern = re.compile(r"^(?:[#\s]*\[?" + re.escape(core_title) + r"\]?[:\s]*\n?)+", re.IGNORECASE)
        text = title_pattern.sub("", text).lstrip()

    return text.strip()


def generate_story_with_rag(story_topic: str, model_path: str, model_name_key: str) -> str:
    st.info(f"Initiating story generation for topic: '{story_topic}'.")

    llm = load_gguf_llm(model_path, model_name_key)
    if not llm:
        return f"Error: Failed to load GGUF LLM."

    model_n_ctx = llm.n_ctx if hasattr(llm, 'n_ctx') else 32768

    chroma_collection = get_chroma_retriever(CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME)
    if not chroma_collection:
        return "Error: Failed to connect to ChromaDB."

    all_generated_section_outputs = []
    output_buffer = 1024

    for section_index, section_config in enumerate(SECTIONS_CONFIG):
        section_id = section_config["id"]
        section_title_display = section_config["title"]
        section_query_hint = section_config["query_hint"]
        section_instructions_for_llm = section_config["instructions"]

        st.markdown(f"--- \n### Generating section {section_index + 1}/{len(SECTIONS_CONFIG)}: {section_title_display}")

        query_text_section = f"Information for {section_query_hint} about '{story_topic}'."
        st.write(f"Retrieving context...")

        section_context_for_rag = ""
        try:
            retriever_results = chroma_collection.query(
                query_texts=[query_text_section], n_results=N_RESULTS_RETRIEVAL_PER_SECTION, include=['documents']
            )
            documents = retriever_results.get('documents')
            if not documents or not documents[0]:
                st.warning(f"No relevant documents found for '{section_title_display}'.")
                section_context_for_rag = f"No specific documents found for '{section_title_display}'. State this fact or synthesize based on 'Story So Far' if allowed and relevant."
            else:
                context_chunks = documents[0]
                section_context_for_rag = "\n\n---\n\n".join(context_chunks)
                st.success(f"Retrieved {len(context_chunks)} context chunks.")
        except Exception as e:
            st.error(f"Error retrieving documents for '{section_title_display}': {e}")
            all_generated_section_outputs.append(f"\n{section_title_display}\n- Error retrieving content: {e}\n")
            continue

        limited_cumulative_story_for_prompt = ""
        valid_past_sections = [s for s in all_generated_section_outputs if not s.lower().startswith(("- error", "- failed", "information for this section"))]

        if section_id == "executive_summary":
            if valid_past_sections:
                full_story_content = "\n".join([re.sub(r"^[#\s]*\[?[^#\]\[]+\]?[\s\n]*", "", s).strip() for s in valid_past_sections])
                limited_cumulative_story_for_prompt = "[Full preceding story context follows for summary generation:]\n" + full_story_content
            else:
                limited_cumulative_story_for_prompt = "[No valid previous sections available for summary.]"
        elif valid_past_sections:
            start_index = max(0, len(valid_past_sections) - MAX_PAST_SECTIONS_IN_CONTEXT)
            relevant_past_parts = valid_past_sections[start_index:]
            limited_cumulative_story_for_prompt = f"[Context from last {len(relevant_past_parts)} section(s)]:\n" + "\n".join(relevant_past_parts)


        section_prompt_template_str = get_prompt_for_single_section(
            section_id, section_title_display, section_instructions_for_llm,
            story_topic, limited_cumulative_story_for_prompt, model_name_key
        )
        prompt_object = PromptTemplate(template=section_prompt_template_str, input_variables=["context"])

        num_input_tokens = 0
        final_prompt_text_for_llm = ""
        try:
            final_prompt_text_for_llm = prompt_object.format_prompt(context=section_context_for_rag).to_string()
            num_input_tokens = llm.get_num_tokens(final_prompt_text_for_llm)
            st.write(f"Input tokens: {num_input_tokens} (Model Ctx: {model_n_ctx})")

            if num_input_tokens >= model_n_ctx - output_buffer:
                st.error(f"⛔ CRITICAL: Input ({num_input_tokens}) too close to context window ({model_n_ctx}) even with buffer ({output_buffer}). LLM will fail.")
                all_generated_section_outputs.append(f"\n{section_title_display}\n- Error: Input prompt too long.\n")
                continue
        except Exception as e:
            st.error(f"Error preparing prompt: {e}")
            continue

        available_tokens_for_output = model_n_ctx - num_input_tokens - output_buffer
        min_required_output = 200
        if available_tokens_for_output <= min_required_output:
            st.error(f"⛔ Not enough tokens available for output ({available_tokens_for_output}). Input: {num_input_tokens}. Need > {min_required_output}. Skipping.")
            all_generated_section_outputs.append(f"\n{section_title_display}\n- Error: Not enough tokens available for LLM output.\n")
            continue

        target_output_tokens = DESIRED_MAX_OUTPUT_TOKENS_SUMMARY if section_id == "executive_summary" else MAX_OUTPUT_TOKENS_REGULAR_SECTION
        effective_max_tokens = min(target_output_tokens, available_tokens_for_output)

        llm.max_tokens = effective_max_tokens
        st.write(f"LLM processing... (Max new tokens: {effective_max_tokens})")

        llm_chain = LLMChain(llm=llm, prompt=prompt_object)
        generated_content_final = ""
        try:
            with st.spinner(f"Generating: {section_title_display.lstrip('#').strip()}..."):
                result = llm_chain.invoke({"context": section_context_for_rag})
                generated_content_raw = result["text"].strip()

            final_cleaned_llm_output = clean_llm_output(generated_content_raw, section_title_display)

            if not final_cleaned_llm_output:
                st.warning(f"LLM returned empty or problematic content for '{section_title_display}' after cleaning.")
                generated_content_final = f"Information for this section ('{section_title_display}') was not available or generation resulted in empty/problematic content."
            else:
                lines = final_cleaned_llm_output.splitlines()
                if len(lines) > 5:
                    line_set = set(lines)
                    if len(line_set) < len(lines) / 2:
                       st.error(f"🚨 High repetition detected in '{section_title_display}'. Output might be flawed. Using placeholder.")
                       generated_content_final = f"Generation for '{section_title_display}' resulted in excessive repetition and was discarded."
                    else:
                        generated_content_final = final_cleaned_llm_output
                else:
                    generated_content_final = final_cleaned_llm_output

            st.success(f"Successfully generated content for: {section_title_display}")

        except Exception as e:
            st.error(f"Error during LLM chain execution for '{section_title_display}': {e}")
            st.error(traceback.format_exc())
            generated_content_final = f"- Failed to generate content due to an LLM error: {e}\n"

        newline_separator = "\n\n" if section_title_display.startswith("#") else "\n"
        current_section_full_output = f"{section_title_display}{newline_separator}{generated_content_final.strip()}\n"
        all_generated_section_outputs.append(current_section_full_output)

    final_story_assembled = "".join(all_generated_section_outputs).strip()

    if not final_story_assembled:
        st.error("Complete story generation failed.")
        return "Error: Story generation resulted in no content."

    st.balloons()
    st.success("🎉 Story generation complete!")
    return final_story_assembled