# =============================================================================
# chat_ui.py — Local LLM Chat with Conversation Memory
# =============================================================================
# Stack : Python 3 · Gradio 6 · LM Studio (OpenAI-compatible API)
# Run   : python chat_ui.py
# Prereq: LM Studio must be open with at least one model loaded
# =============================================================================

import os
import json
from datetime import datetime

import gradio as gr
from openai import OpenAI

# ── LM Studio connection ──────────────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = "chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# =============================================================================
# Helper functions
# =============================================================================

def get_available_models() -> list[str]:
    """Query LM Studio for whichever models are currently loaded."""
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        return ids if ids else ["(No models loaded in LM Studio)"]
    except Exception:
        return ["(Cannot reach LM Studio — is it running?)"]


def build_message_list(history: list, system_prompt: str, user_message: str) -> list:
    """Convert Gradio's ChatMessage elements into the format the API expects."""
    messages = []

    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    # Gradio 6 passes history as a list of dicts/ChatMessages with 'role' and 'content'
    for turn in history:
        role = turn.get("role") if isinstance(turn, dict) else turn.role
        content = turn.get("content") if isinstance(turn, dict) else turn.content
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


def chat_stream(
    user_message: str,
    history: list,
    system_prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
):
    """Send the conversation to LM Studio and stream the reply token-by-token."""
    messages = build_message_list(history, system_prompt, user_message)

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
            stream=True,
        )

        partial_reply = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                partial_reply += token
                yield partial_reply

    except Exception as e:
        yield f"[Error from LM Studio: {e}]"


def save_conversation(history: list, system_prompt: str, model_id: str) -> str:
    """Write the current conversation to a JSON file in the chat_logs/ folder."""
    if not history:
        return "Nothing to save — start a conversation first."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"chat_{timestamp}.json")

    # Adapt formatting to process structural dictionaries
    formatted_turns = []
    for turn in history:
        role = turn.get("role") if isinstance(turn, dict) else turn.role
        content = turn.get("content") if isinstance(turn, dict) else turn.content
        formatted_turns.append({"role": role, "content": content})

    log = {
        "timestamp": timestamp,
        "model": model_id,
        "system_prompt": system_prompt,
        "turns": len(history) // 2,
        "conversation": formatted_turns,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return f"Saved → {filename}"


# =============================================================================
# Gradio UI layout
# =============================================================================

# REMOVED theme from here to comply with Gradio 6 standards
with gr.Blocks(title="Local LLM Chat") as demo:

    gr.Markdown(
        "## 🖥️ Local LLM Chat\n"
        "Powered by **LM Studio** · Full conversation memory · Streaming output"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                label="Active model",
                info="Pulled from LM Studio on launch",
                interactive=True,
            )
            refresh_btn = gr.Button("↺ Refresh model list", size="sm")

            gr.Markdown("---")

            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="Give the model a persona or set of rules…",
                lines=4,
                value="You are a helpful, concise assistant.",
            )

            gr.Markdown("---")

            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.05, value=0.7,
                label="Temperature",
                info="Lower = focused / deterministic · Higher = creative / varied",
            )
            max_tokens = gr.Slider(
                minimum=64, maximum=8192, step=64, value=1024,
                label="Max output tokens",
            )

            gr.Markdown("---")

            save_btn = gr.Button("💾 Save conversation", variant="secondary")
            save_status = gr.Textbox(
                show_label=False, interactive=False, lines=1,
                placeholder="Save status will appear here…",
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="Conversation",
                buttons=["copy"],
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your message and press Enter…",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                )
                submit_btn = gr.Button("Send ▶", variant="primary", scale=1)

            clear_btn = gr.Button("🗑 Clear conversation", size="sm")

    with gr.Row():
        turn_counter = gr.Textbox(
            show_label=False, interactive=False,
            value="Turns: 0", scale=1,
        )
        gr.Markdown("", scale=4)

    # =========================================================================
    # Event wiring
    # =========================================================================

    def refresh_models():
        return gr.Dropdown(choices=get_available_models())

    refresh_btn.click(fn=refresh_models, outputs=model_dropdown)

    def submit_message(user_msg, history, sys_prompt, model, temp, tokens):
        if not user_msg.strip():
            yield "", history, f"Turns: {len(history) // 2}"
            return

        # Gradio 6 expects appends as dictionary messaging components
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})

        # Send the history excluding the new blank slots we just created
        for partial_reply in chat_stream(
            user_msg, history[:-2], sys_prompt, model, temp, tokens
        ):
            history[-1]["content"] = partial_reply
            yield "", history, f"Turns: {len(history) // 2}"

    msg_box.submit(
        fn=submit_message,
        inputs=[msg_box, chatbot, system_prompt, model_dropdown, temperature, max_tokens],
        outputs=[msg_box, chatbot, turn_counter],
    )
    submit_btn.click(
        fn=submit_message,
        inputs=[msg_box, chatbot, system_prompt, model_dropdown, temperature, max_tokens],
        outputs=[msg_box, chatbot, turn_counter],
    )

    clear_btn.click(
        fn=lambda: ([], "", "Turns: 0"),
        outputs=[chatbot, msg_box, turn_counter],
    )

    save_btn.click(
        fn=save_conversation,
        inputs=[chatbot, system_prompt, model_dropdown],
        outputs=save_status,
    )


# =============================================================================
# Launch
# =============================================================================

if __name__ == "__main__":
    print("Starting Local LLM Chat UI…")
    print("Make sure LM Studio is running with a model loaded.")
    print("Navigate to http://127.0.0.1:7860 if the browser doesn't open automatically.\n")

    # PLACED theme processing elements here to fully align with Gradio 6
    demo.launch(
        inbrowser=True,
        share=False,
        server_port=7860,
        theme=gr.themes.Soft()
    )
