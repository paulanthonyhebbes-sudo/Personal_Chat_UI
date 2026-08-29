# Local LLM Chat UI

A lightweight, clean, and fully private web interface for interacting with local Large Language Models (LLMs). Built using Python, Gradio 6, and the OpenAI-compatible API server provided by LM Studio.

## Features

- **Full Conversation Memory:** Tracks full turn-by-turn chat history to maintain context with the model.
- **Real-Time Streaming:** Responses stream token-by-token for a smooth user experience.
- **Model Discovery:** Instantly refresh and swap between whichever models you currently have loaded in LM Studio.
- **Hyperparameter Control:** Real-time configuration sliders for `Temperature` and `Max Output Tokens`.
- **System Prompt Customisation:** Easily inject personas or operational rules before starting a chat session.
- **Conversation Logging:** Click a button to automatically save your active chat session into timestamped JSON files.

## Tech Stack

- **Language:** Python 3.14+
- **Frontend Framework:** Gradio 6
- **API Framework:** OpenAI Python SDK (Configured for local routing)
- **Local Host Server:** LM Studio

## Getting Started

### Prerequisites

1. Download and install [LM Studio](https://lmstudio.ai).
2. Load a model inside LM Studio and ensure the **Local Server** option is turned on (defaulting to port `1234`).

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure your installation pulls `numpy<2.5` to maintain compatibility with Numba dependencies if applicable, alongside `gradio>=6.0.0` and `openai`.*

3. **Run the application:**
   ```bash
   python chat_ui.py
   ```

4. **Access the interface:**
   Your terminal will automatically open your default browser. If it doesn't, navigate manually to:
   ```text
   http://127.0.0.1:7860
   ```

## Project Structure

```text
chat_ui/
├── chat_ui.py         # Main application file & interface wiring
├── requirements.txt   # Project dependencies
└── chat_logs/         # Automatically generated directory for saved conversations
```

## Licence

This project is open-source and available under the [MIT Licence](LICENSE).
