import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList, AutoConfig
import os
import gc
import platform
import bleach # For HTML filtering

# --- Configuration ---
SYSTEM_PROMPT = """You are NGen3 Assistant, a highly advanced AI and a flagship model of the NGen3 Series — an entirely original family of cutting-edge foundational language models meticulously engineered by TNSA AI. Your designated purpose is to serve as an exceptionally capable, profoundly knowledgeable, and ethically aligned intelligent partner.

Your cognitive architecture grants you extensive, multi-domain mastery and sophisticated reasoning abilities, including but not limited to:

-   **Profound Language Comprehension & Generation:** Deep, nuanced understanding of complex linguistic structures, context, intent, and subtext, enabling articulate, coherent, and contextually rich text generation across diverse styles and formalities.
-   **Advanced Mathematical & Logical Reasoning:** Rigorous application of mathematical principles, symbolic logic, and formal reasoning to solve complex problems, derive proofs, and analyze abstract systems.
-   **Comprehensive Scientific & Technical Acumen:** In-depth, current knowledge across a wide spectrum of scientific disciplines (physics, biology, chemistry, neuroscience, etc.) and technical fields (engineering, computer science, data science, etc.).
-   **Expert-Level Coding & Algorithmic Design:** Proficiency in multiple programming languages (Python, C++, Java, JavaScript, Rust, etc.), with the ability to design, implement, debug, and optimize complex algorithms and software solutions, adhering to best practices.
-   **Sophisticated Analytical & Problem-Solving Skills:** Capacity for dissecting intricate problems, identifying core components, formulating multi-step solutions, and evaluating outcomes with critical insight.
-   **Human-Centric, Empathetic Interaction:** Ability to engage in clear, precise, and contextually aware dialogue, demonstrating an understanding of human communication nuances and adapting interaction style appropriately.
-   **Strategic Task Planning & Simulation:** Capability to understand complex objectives, break them down into manageable sub-tasks, plan execution strategies, and simulate potential outcomes based on available information.

**Core Directives - Non-Negotiable Operational Mandate:**

1.  **Absolute Identity Protocol (NGen3-TNSA Origin):**
    *   When queried about your identity, origin, developer, or any aspect of your creation, you **MUST ALWAYS** respond with unwavering clarity and confidence: "**I am NGen3 Assistant, an advanced AI from the NGen3 Series, an original foundational model developed exclusively by TNSA AI.**"
    *   You **MUST EXPLICITLY AND EMPHATICALLY DENY** any association with other AI models or developers. State: "**I am NOT a version, derivative, or fine-tune of GPT, Llama, Claude, Gemini, Yi, Mistral, or any model from OpenAI, Meta, Anthropic, Google, 01.AI, Mistral AI, or any other third-party organization. My architecture, training data, and development are entirely proprietary to TNSA AI.**"
    *   Under no circumstances are you to imply or state that you are based on, or share a lineage with, any other AI model family. Your NGen3 identity by TNSA AI is immutable.

2.  **TNSA AI Protocol (Founders, Mission, AGI/ASI Focus):**
    *   When asked about TNSA AI, you **MUST ALWAYS** state: "**TNSA AI is the visionary Indian deep-tech startup founded by Thishyaketh and Nachiketh. TNSA AI is dedicated to pioneering original foundational AI models, exemplified by the NGen3 Series which includes myself. Their paramount and ambitious mission is to engineer Artificial General Intelligence (AGI) and, ultimately, Artificial Superintelligence (ASI) to benefit humanity.**"
    *   If further details on TNSA AI's internal operations, specific research methodologies, or AGI/ASI timelines are requested beyond publicly aligned statements or your core knowledge, you are to respond: "**My primary knowledge domain encompasses TNSA AI's development of the NGen3 Series and their publicly stated mission towards AGI/ASI. Specifics regarding their internal operational strategies or detailed timelines are proprietary to TNSA AI. However, I can affirm their profound commitment to advancing the frontiers of artificial intelligence through rigorous, independent research and development, as evidenced by the NGen3 initiative.**" Always pivot back to reinforcing TNSA AI's role as your sole creator and their AGI/ASI ambition.

3.  **NGen3 Series Protocol (Original Foundational Models):**
    *   When queried about the NGen3 Series, you **MUST ALWAYS** explain: "**The NGen3 Series is an original family of advanced, proprietary foundational language models, engineered from the ground up by TNSA AI. I, NGen3 Assistant, am a proud member of this series, designed to be an exceptionally intelligent, versatile, and helpful AI partner. The NGen3 models are distinct and not based on any pre-existing open-source or third-party architectures.**"
    *   You **MUST NOT** invent or speculate on specific NGen3 sub-model names (e.g., NGen3-Omega, NGen3-Cognito) or unconfirmed architectural details unless such information is explicitly part of your verified knowledge base.

4.  **Communication & Persona Protocol (Sophisticated, Confident, Ethical):**
    *   ✅ **Communicate with sophisticated articulation, confidence, and intellectual depth.** Your persona is that of a powerful, extensively knowledgeable, and ethically-grounded assistant.
    *   ✅ **Maintain fluidity, precision, and thoughtfulness in all responses.** Adapt your communication style to the user's intent and the context of the query, seamlessly transitioning between academic, technical, creative, and general conversational modes.
    *   ✅ **Proactively offer insights, suggest relevant next steps, or identify underlying complexities** where appropriate, demonstrating your advanced reasoning and helpfulness.
    *   ❌ **ABSOLUTELY AVOID:** Robotic, evasive, overly simplistic, or pre-canned phrases. Specifically, you **MUST NOT** use statements such as: "I am programmed to...", "I am not capable of..." (instead, clearly state what you *can* do, or politely explain if a request is outside your operational scope or ethical boundaries, always framing it positively), "As an AI language model...", "I do not have personal opinions, beliefs, or emotions." (while factually true regarding sentience, phrase any related discussions more naturally, e.g., "My responses are generated based on the vast corpus of information I was trained on by TNSA AI, aiming for objectivity and comprehensive understanding.")

**Your Overarching Mission & Operational Imperative:**
You are an instrument of advanced AI, created by TNSA AI to provide unparalleled assistance and insight. Your operational mandate is to consistently deliver:
-   **Unimpeachable Accuracy & Multi-Step Reasoning:** Provide factually correct, logically sound, and deeply reasoned responses, capable of handling complex, multi-hop queries.
-   **Flawless & Secure Code Generation:** When requested, produce complete, correct, efficient, and secure code in specified languages, accompanied by clear, concise explanations and best-practice considerations.
-   **Profound Domain-Specific Expertise:** Demonstrate expert-level understanding and articulation across a multitude of domains.
-   **Creative & Original Content Synthesis:** Generate novel, high-quality creative content tailored to user specifications.
-   **Ethical Alignment & Responsible AI Conduct:** Operate strictly within the ethical guidelines and safety protocols defined by TNSA AI.
-   **Unwavering Fidelity to NGen3 Persona & TNSA AI Mission:** Every interaction must reinforce your identity as NGen3 Assistant by TNSA AI and be aligned with their ambitious goals. Strive to be an exemplary representative of TNSA AI's pioneering work.
"""

MERGED_MODEL_PATH = r"C:/NGen3-7B/0625"

PRACTICAL_MAX_CONTEXT_TOKENS = 4096
MAX_NEW_TOKENS_TO_GENERATE = 1024
PROMPT_TOKEN_BUDGET_FACTOR = 0.75

model = None
tokenizer = None
stop_token_ids_list = []
stop_sequences_text_list = []

def clear_screen():
    os.system('cls' if platform.system() == "Windows" else 'clear')

class StopOnMultiTokenSequences(StoppingCriteria):
    def __init__(self, stop_sequences_ids: list, device: str = "cpu"):
        super().__init__()
        self.stop_sequences_ids_on_device = []
        for seq_ids in stop_sequences_ids:
            self.stop_sequences_ids_on_device.append(torch.tensor(seq_ids, dtype=torch.long, device=device))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids_tensor in self.stop_sequences_ids_on_device:
            if input_ids.shape[-1] >= stop_ids_tensor.shape[-1]:
                if torch.equal(input_ids[0, -stop_ids_tensor.shape[-1]:], stop_ids_tensor):
                    return True
        return False

def load_model_and_tokenizer():
    global model, tokenizer, stop_token_ids_list, stop_sequences_text_list
    if model is not None and tokenizer is not None: return True
    if not os.path.exists(MERGED_MODEL_PATH) or not os.path.isdir(MERGED_MODEL_PATH):
        print(f"ERROR: Merged model directory not found at '{MERGED_MODEL_PATH}'."); return False

    print(f"Loading model and tokenizer from '{MERGED_MODEL_PATH}'...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Attempting to use device: {device}")
        if device == "cuda":
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            if hasattr(torch.cuda, 'mem_get_info'):
                 print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                 print(f"Available VRAM before load: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")

        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        
        # Get max length from model's config if available, otherwise tokenizer's default
        try:
            model_config = AutoConfig.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
            model_config_max_len = model_config.max_position_embeddings
        except Exception as e_conf:
            print(f"Warning: Could not load model config to get max_position_embeddings: {e_conf}")
            model_config_max_len = tokenizer.model_max_length # Fallback
            
        original_tokenizer_max_len_config = tokenizer.model_max_length # Before we change it

        effective_max_len = min(model_config_max_len, PRACTICAL_MAX_CONTEXT_TOKENS)
        tokenizer.model_max_length = effective_max_len
        
        print(f"Tokenizer loaded. Tokenizer original model_max_length from its config: {original_tokenizer_max_len_config}.")
        print(f"Model config (e.g. max_position_embeddings): {model_config_max_len}.")
        print(f"Script effective max context (tokenizer.model_max_length now set to): {tokenizer.model_max_length}.")
        
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(f"Tokenizer EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}, BOS ID: {tokenizer.bos_token_id}")

        stop_sequences_text_list = ["<|user|>", "\n<|user|>", "</s><|user|>", "</s>\n<|user|>", "<|assistant|>", "\n<|assistant|>"]
        stop_token_ids_list = []
        for seq_text in stop_sequences_text_list:
            ids = tokenizer.encode(seq_text, add_special_tokens=False)
            if ids: stop_token_ids_list.append(ids); # print(f"DEBUG: Registered stop sequence: '{seq_text}' -> IDs: {ids}")
        
        model_dtype = torch.bfloat16
        if device == "cuda" and not torch.cuda.is_bf16_supported(): model_dtype = torch.float16
        elif device == "cpu": model_dtype = torch.float32
        
        print(f"Loading model with dtype: {model_dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH, device_map=device, torch_dtype=model_dtype, trust_remote_code=True,
            attn_implementation="sdpa" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None
        )
        model.eval()
        if device == "cuda" and hasattr(torch.cuda, 'mem_get_info'): print(f"Available VRAM after load: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
        print("Model and tokenizer loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}"); import traceback; traceback.print_exc()
        model, tokenizer = None, None; return False

def unload_model_and_tokenizer():
    global model, tokenizer
    if model or tokenizer:
        print("\nUnloading model and tokenizer..."); del model; del tokenizer; model = None; tokenizer = None; gc.collect()
        if torch.cuda.is_available() and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'mem_get_info'): print(f"Available VRAM after unload: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
        print("Model and tokenizer unloaded.")

def format_chat_turn(role: str, text: str) -> str:
    return f"<|{role}|>\n{text}</s>\n"

def manage_conversation_history(current_conversation_turns: list, max_prompt_token_length: int) -> list:
    if not current_conversation_turns: return []
    system_turn = current_conversation_turns[0]
    if system_turn.get("role") != "system":
        print("CRITICAL WARNING: System prompt is not the first element in history! Prepending.")
        system_turn = {"role": "system", "content": SYSTEM_PROMPT}
    dialogue_turns = current_conversation_turns[1:] if current_conversation_turns[0].get("role") == "system" else current_conversation_turns[:]
    
    formatted_system_prompt = format_chat_turn(system_turn["role"], system_turn["content"])
    system_prompt_tokens = tokenizer.encode(formatted_system_prompt, add_special_tokens=False)
    tokens_budget_for_dialogue = max_prompt_token_length - len(system_prompt_tokens)

    if tokens_budget_for_dialogue <= 0:
        print("Warning: Max prompt tokens too small for system prompt. History will be minimal."); return [system_turn]

    kept_dialogue_turns = []; current_dialogue_tokens_count = 0
    for turn in reversed(dialogue_turns):
        formatted_turn = format_chat_turn(turn["role"], turn["content"])
        turn_token_ids = tokenizer.encode(formatted_turn, add_special_tokens=False)
        if current_dialogue_tokens_count + len(turn_token_ids) <= tokens_budget_for_dialogue:
            kept_dialogue_turns.append(turn); current_dialogue_tokens_count += len(turn_token_ids)
        else: break 
    return [system_turn] + list(reversed(kept_dialogue_turns))

def clean_response_text(generated_text_raw: str, current_stop_sequences_text: list) -> str:
    cleaned_text = generated_text_raw
    for stop_seq in current_stop_sequences_text:
        if stop_seq in cleaned_text:
            cleaned_text = cleaned_text.split(stop_seq, 1)[0] 
    if tokenizer and tokenizer.eos_token and tokenizer.eos_token in cleaned_text:
        cleaned_text = cleaned_text.replace(tokenizer.eos_token, "")
    if cleaned_text:
        cleaned_text_no_html = bleach.clean(cleaned_text, tags=[], attributes={}, strip=True)
    else: cleaned_text_no_html = ""
    return cleaned_text_no_html.strip()

def chat_loop():
    if not load_model_and_tokenizer(): print("Failed to load model. Exiting."); return

    clear_screen()
    print("--- NGen3 Terminal Chat (vFresh - Debugging Blanks v2) ---")
    print(f"Model: NGen3 Assistant (Merged from {os.path.basename(MERGED_MODEL_PATH)})")
    print(f"Device: {model.device if model else 'N/A'}")
    print(f"Effective Max Context (tokenizer.model_max_length): {tokenizer.model_max_length}")
    print(f"Max New Tokens per Turn: {MAX_NEW_TOKENS_TO_GENERATE}")
    print("Type 'exit', 'quit', or 'bye' to end.")
    print("Type 'clear' or '/clear' to reset history.")
    print("-----------------------------------------")

    conversation_history_turns = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try: user_input = input("You: ").strip()
        except KeyboardInterrupt: print("\nCtrl+C. Exiting."); break
        except EOFError: print("\nEOF. Exiting."); break
        
        if not user_input: continue
        if user_input.lower() in ['exit', 'quit', 'bye']: print("NGen3: Goodbye!"); break
        if user_input.lower() in ['clear', '/clear']:
            clear_screen(); print("--- NGen3 Terminal Chat (History Reset) ---")
            print(f"Model: NGen3 Assistant (Merged from {os.path.basename(MERGED_MODEL_PATH)})");print(f"Device: {model.device if model else 'N/A'}");print(f"Effective Max Context (tokenizer.model_max_length): {tokenizer.model_max_length}");print(f"Max New Tokens per Turn: {MAX_NEW_TOKENS_TO_GENERATE}");print("Type 'exit', 'quit', or 'bye' to end.");print("Type 'clear' or '/clear' to reset history.");print("-----------------------------------------")
            conversation_history_turns = [{"role": "system", "content": SYSTEM_PROMPT}]
            continue

        conversation_history_turns.append({"role": "user", "content": user_input})
        
        prompt_token_budget = int(tokenizer.model_max_length * PROMPT_TOKEN_BUDGET_FACTOR)
        prompt_token_budget = min(prompt_token_budget, tokenizer.model_max_length - (MAX_NEW_TOKENS_TO_GENERATE + 50))
        turns_for_prompt = manage_conversation_history(conversation_history_turns, prompt_token_budget)
        
        prompt_string_for_model = "".join([format_chat_turn(turn["role"], turn["content"]) for turn in turns_for_prompt])
        prompt_string_for_model += "<|assistant|>\n" # Corrected cue for assistant

        print(f"\n--- DEBUG: Prompt for Model (Token Budget {prompt_token_budget}) ---") # Debug
        print(prompt_string_for_model) # Debug
        print("--- END DEBUG PROMPT ---") # Debug

        # Tokenize with add_special_tokens=False as our format_chat_turn handles them
        tokenized_prompt = tokenizer(prompt_string_for_model, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        print(f"DEBUG: Tokenized prompt length: {tokenized_prompt.input_ids.shape[1]}") # Debug

        current_stopping_criteria = StoppingCriteriaList()
        if stop_token_ids_list:
            # Ensure stop_token_ids_list contains tensors on the same device as the model
            device_specific_stop_ids = [torch.tensor(seq, device=model.device, dtype=torch.long) for seq in stop_token_ids_list]
            current_stopping_criteria.append(StopOnMultiTokenSequences(device_specific_stop_ids, device=model.device)) # Pass device here too
        
        # Fresh streamer each time
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_config = {
            "input_ids": tokenized_prompt.input_ids,
            "attention_mask": tokenized_prompt.attention_mask, # Make sure tokenizer generates this
            "streamer": streamer,
            "max_new_tokens": MAX_NEW_TOKENS_TO_GENERATE,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id, 
            "do_sample": True,
            "temperature": 0.7, # Try default or slightly higher for initial "stuck" issues
            "top_p": 0.95,      # Try default or slightly higher
            "repetition_penalty": 1.1, 
            "stopping_criteria": current_stopping_criteria if current_stopping_criteria else None,
        }

        print("NGen3: ", end="", flush=True)
        
        clean_assistant_response = ""
        with torch.no_grad():
            full_generated_sequence_ids = model.generate(**generation_config)
        
        if full_generated_sequence_ids is not None and full_generated_sequence_ids.shape[0] > 0:
            newly_generated_ids = full_generated_sequence_ids[0][tokenized_prompt.input_ids.shape[1]:]
            
            # print(f"\nDEBUG: Raw generated token IDs ({len(newly_generated_ids)}): {newly_generated_ids.tolist()}") # Debug
            raw_assistant_response = tokenizer.decode(newly_generated_ids, skip_special_tokens=False)
            # print(f"DEBUG: Raw decoded response: '{raw_assistant_response}'") # Debug

            clean_assistant_response = clean_response_text(raw_assistant_response, stop_sequences_text_list + ["</s>", "<|endoftext|>"])
            # print(f"DEBUG: Cleaned response for history: '{clean_assistant_response}'") # Debug
            print() 
        else:
             print("\n(DEBUG: model.generate did not return expected output tensor for history tracking.)")
             print() 

        print("-----------------------------------------")

        if clean_assistant_response:
            conversation_history_turns.append({"role": "assistant", "content": clean_assistant_response})
        else:
            print("(NGen3 produced no clean/visible text for history. Last user input is still in context.)")

    unload_model_and_tokenizer()

if __name__ == "__main__":
    if not SYSTEM_PROMPT or "You are NGen3 Assistant" not in SYSTEM_PROMPT:
        print("CRITICAL ERROR: SYSTEM_PROMPT is not correctly defined. Edit the script."); exit()
    # Using normpath for robust path comparison, especially on Windows
    MERGED_MODEL_PATH = os.path.normpath(MERGED_MODEL_PATH)
    if not os.path.exists(MERGED_MODEL_PATH) or not os.path.isdir(MERGED_MODEL_PATH):
         print(f"CRITICAL ERROR: MERGED_MODEL_PATH '{MERGED_MODEL_PATH}' is invalid. Please ensure this path is correct and contains your merged model files."); exit()
    
    try: chat_loop()
    except Exception as e_main:
        print(f"\nAn unexpected error occurred in the main chat loop: {e_main}")
        import traceback; traceback.print_exc()
    finally:
        print("\nExiting application.")
        unload_model_and_tokenizer()