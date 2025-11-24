```text
██╗      ██████╗  ██████╗ █████╗ ██╗      ██████╗ ███████╗███╗   ██╗
██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔════╝ ██╔════╝████╗  ██║
██║     ██║   ██║██║     ███████║██║     ██║  ███╗█████╗  ██╔██╗ ██║
██║     ██║   ██║██║     ██╔══██║██║     ██║   ██║██╔══╝  ██║╚██╗██║
███████╗╚██████╔╝╚██████╗██║  ██║███████╗╚██████╔╝███████╗██║ ╚████║
╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝
```
# NGen3 Terminal Chat Interface

This project provides a full-featured, locally running terminal interface for interacting with the **NGen3 Assistant**, a flagship model from the NGen3 Series by TNSA AI. It is designed for users who want a powerful, customizable, offline chat experience with advanced debugging, context control, and VRAM-aware model loading.

## 🚀 Features
- **Local Model Runtime**: Load merged NGen3 models directly from disk.
- **Dynamic Context Management**: Automatically trims conversation history to prevent context overflows.
- **Custom Stopping Criteria**: Supports multi-token stop sequences for clean and controlled outputs.
- **HTML-Safe Output**: Responses are filtered to strip unsafe HTML using `bleach`.
- **VRAM Awareness**: Displays GPU memory usage before and after model loading.
- **Configurable Generation Settings**: Temperature, top-p, repetition penalty, and more.
- **Debug Mode**: Shows full prompts sent to the model for transparent debugging.

## 📦 Requirements
Ensure the following Python packages are installed:
```bash
pip install torch transformers bleach
```

## 📁 Model Setup
Place your merged NGen3 model in a folder and set the path in the script:
```python
MERGED_MODEL_PATH = r"C:/NGen3-7B/0625"
```

## ▶️ Running the Script
Run the script with:
```bash
python terminal_chat.py
```

If the model loads successfully, you will see:
- Model path
- Device information
- Effective max context length
- VRAM usage

Then the chat interface starts:
```
You: <your message>
NGen3: <model response>
```

## 🔧 Commands
- `exit`, `quit`, `bye` → Quit the program
- `clear`, `/clear` → Reset full conversation history

## 🧠 Conversation Handling
The script:
- Uses a system prompt defining NGen3 Assistant
- Manages token budget dynamically
- Supports streaming responses
- Cleans unwanted stop sequences and special tokens

## ⚠️ Notes
- Ensure the GPU has enough VRAM for the merged model
- Prefer BF16 when supported; fallback to FP16 or FP32 based on device
- On CPU, inference will be slow

## 🧹 Unloading
When exiting, the script:
- Clears CUDA cache
- Frees memory
- Resets model and tokenizer

## 🏁 Summary
This script is ideal for:
- Offline inference
- Debugging model behavior
- Building local AI tools on top of NGen3
- Testing merged or fine-tuned models

It gives you powerful control over your model’s inputs, outputs, and system behavior — all inside a simple terminal UI.
