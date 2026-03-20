"""
Gradio chatbot app powered by Claude CLI — mirrors VSCode Claude Code behavior.

Run: python src/app/app.py
"""

import sys
import traceback
from pathlib import Path

import gradio as gr

# Make src/llm importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
from claude_cli import ClaudeSession

# ---------------------------------------------------------------------------
# Few-shot example helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SEPARATOR = "\n\n--------------------------------------------------------------------------------------------------------------------------------------------------\n\n"


def load_default_few_shot_examples() -> str:
    """Load and format the default few-shot QA examples from legacy/raw_text.txt."""
    raw_file = ROOT / "legacy" / "raw_text.txt"
    text = raw_file.read_text().replace(" }", "}").replace("}:", ":}")
    parts = text.split(SEPARATOR)
    examples = []
    for i in range(0, len(parts) - 1, 2):
        examples.append({"question": parts[i], "answer": parts[i + 1]})

    preamble = (
        "The following questions and respective answers are few-shot examples "
        "demonstrating the style and depth of reasoning expected:\n\n"
    )
    for ex in examples:
        preamble += f"{ex['question']}\n\n{ex['answer']}\n\n\n\n"
    return preamble


DEFAULT_FEW_SHOT_PREAMBLE = load_default_few_shot_examples()


def user_submit(message, history, session_id, model, system_prompt):
    """Append user message to history, return immediately."""
    history = history + [{"role": "user", "content": message}]
    return "", history, session_id


def bot_respond(history, session_id, model, system_prompt, use_few_shot, custom_few_shot):
    """Generate Claude's response, maintaining the session across turns."""
    raw_content = history[-1]["content"]
    # Gradio 6.x stores content as [{"text": "...", "type": "text"}, ...] blocks
    if isinstance(raw_content, list):
        user_message = " ".join(
            part["text"] for part in raw_content if isinstance(part, dict) and "text" in part
        )
    else:
        user_message = str(raw_content)

    # Gradio 6.x may pass widget values as lists — unwrap all of them
    if isinstance(model, list):
        model = model[0] if model else None
    if isinstance(system_prompt, list):
        system_prompt = system_prompt[0] if system_prompt else None
    if isinstance(use_few_shot, list):
        use_few_shot = use_few_shot[0] if use_few_shot else False
    if isinstance(custom_few_shot, list):
        custom_few_shot = custom_few_shot[0] if custom_few_shot else ""

    # Prepend few-shot examples only on the first message (no existing session)
    is_first_message = session_id is None
    if is_first_message:
        if use_few_shot:
            preamble = DEFAULT_FEW_SHOT_PREAMBLE
        elif custom_few_shot and custom_few_shot.strip():
            preamble = custom_few_shot.strip() + "\n\n"
        else:
            preamble = ""

        if preamble:
            user_message = preamble + "Now, here is the user's request:\n\n" + user_message

    print(f"[bot_respond] session_id={session_id}, model={model}, message={user_message[:80]!r}")

    if session_id:
        session = ClaudeSession.resume(
            session_id,
            model=model or None,
            system_prompt=system_prompt or None,
        )
    else:
        session = ClaudeSession(
            model=model or None,
            system_prompt=system_prompt or None,
        )

    try:
        response = session.prompt(user_message)
        text = response.get("result", "")
        session_id = response.get("session_id", session_id)
        print(f"[bot_respond] OK, session_id={session_id}, response length={len(text)}")
    except RuntimeError as e:
        traceback.print_exc()
        err = str(e).lower()
        if "rate_limit" in err or "rate limit" in err or "429" in err:
            text = (
                "**Rate limit reached.** Your account has hit the usage limit "
                "(shared with VSCode). The session is still active — wait a "
                "moment and try again."
            )
        else:
            text = f"**Error:** {e}"
    except Exception as e:
        traceback.print_exc()
        text = f"**Error:** {e}"

    history = history + [{"role": "assistant", "content": text}]
    return history, session_id


def clear_session():
    """Reset chat and session state (like opening a new VSCode chat)."""
    return [], None


with gr.Blocks(title="Claude Code Chat") as demo:
    gr.Markdown("# Claude Code Chat\nMulti-turn chatbot using the Claude CLI — same session behavior as VSCode.")

    # Persistent session_id across turns, stored in Gradio server-side state
    session_state = gr.State(value=None)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["sonnet", "opus", "haiku"],
            value="sonnet",
            label="Model",
        )
        system_prompt_box = gr.Textbox(
            label="System prompt (appended to built-in + CLAUDE.md)",
            placeholder="e.g. You are a Lean 4 expert...",
        )

    with gr.Accordion("Few-shot examples", open=False):
        few_shot_checkbox = gr.Checkbox(
            value=True,
            label="Include default few-shot reasoning examples",
        )
        custom_few_shot_box = gr.Textbox(
            label="Custom few-shot examples (used when default is unchecked)",
            placeholder="Paste your own Q&A examples here...",
            lines=6,
            interactive=True,
        )

    chatbot = gr.Chatbot(height=500, latex_delimiters=[
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ])
    msg = gr.Textbox(label="Message", placeholder="Type a message...", autofocus=True)
    clear = gr.Button("New session")

    # Two-step: first add user message to chat, then generate bot response.
    # This lets the user message appear instantly before the CLI call blocks.
    msg.submit(
        user_submit,
        inputs=[msg, chatbot, session_state, model_dropdown, system_prompt_box],
        outputs=[msg, chatbot, session_state],
    ).then(
        bot_respond,
        inputs=[chatbot, session_state, model_dropdown, system_prompt_box, few_shot_checkbox, custom_few_shot_box],
        outputs=[chatbot, session_state],
    )

    clear.click(clear_session, outputs=[chatbot, session_state])


if __name__ == "__main__":
    demo.launch()
