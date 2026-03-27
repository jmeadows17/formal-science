"""
Gradio Lean Code Generator powered by Claude or GPT CLI.

Feeds prompts from ``lean_prompt_data.json`` one at a time to an LLM and
displays the generated Lean code.  Also supports custom prompts.

Run: python src/app/lean_app.py
"""

import sys
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "llm"))

from claude_cli import ClaudeSession
from gpt_cli import GPTSession


DEFAULT_PROMPT_DATA_PATH = _SRC / "app_data" / "lean_prompt_data.json"
DEFAULT_OUTPUT_PATH = _SRC / "app_data" / "lean_output_data.json"


def _load_prompts(path: Path = DEFAULT_PROMPT_DATA_PATH) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, str)]


def _load_gpt_models() -> list[str]:
    cache_path = Path.home() / ".codex" / "models_cache.json"
    fallback = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2-codex", "gpt-5.2"]
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback
    models = [
        m.get("slug")
        for m in data.get("models", [])
        if isinstance(m, dict) and m.get("visibility") == "list" and m.get("slug")
    ]
    return models or fallback


def _load_saved_outputs() -> list[dict]:
    try:
        data = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data


def _autosave_outputs(outputs: list[dict]):
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


PROMPTS = _load_prompts()
TOTAL_PROMPTS = len(PROMPTS)

PROVIDER_MODELS = {
    "Claude": ["sonnet", "opus", "haiku"],
    "GPT": _load_gpt_models(),
}

CSS = """
.gradio-container { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.scroll-panel {
    border: 1px solid var(--border-color-primary, #d0d0d0);
    border-radius: 10px;
    padding: 20px 24px;
    background: var(--background-fill-primary, #fafafa);
    font-size: 0.95em;
    line-height: 1.6;
}
.scroll-panel .prose { max-width: none !important; }
.panel-label {
    text-transform: uppercase;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--body-text-color-subdued, #888);
    padding-bottom: 6px;
}
.review-row { justify-content: center !important; gap: 16px !important; padding: 12px 0; }
.status-bar { text-align: center; padding: 4px 0; font-size: 0.9em; }
.settings-row { align-items: end !important; }
"""

CODE_ONLY_SUFFIX = """

# Output format
Return only the final Lean source code for the requested file.
Do not include any explanation, commentary, bullet points, prose, or Markdown code fences.
Your entire response must be valid Lean file contents only.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(prompt_index, saved_count, mode, extra=""):
    parts = [f"**{saved_count}** outputs saved"]
    if mode == "Default Pipeline":
        idx = min(prompt_index + 1, TOTAL_PROMPTS)
        parts.append(f"Prompt **{idx} / {TOTAL_PROMPTS}**")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def _unwrap(val):
    return val[0] if isinstance(val, list) else val


def _normalize_model(provider, model):
    return _unwrap(model) or None


def _get_session_cls(provider):
    return GPTSession if provider == "GPT" else ClaudeSession


def _make_session(provider, session_id, model, system_prompt):
    model = _normalize_model(provider, model)
    system_prompt = _unwrap(system_prompt) or None
    session_cls = _get_session_cls(provider)
    if session_id:
        return session_cls.resume(session_id, model=model, system_prompt=system_prompt)
    return session_cls(model=model, system_prompt=system_prompt)


def _stream(provider, message, session_id, model, system_prompt):
    session = _make_session(provider, session_id, model, system_prompt)
    full = ""
    try:
        for chunk in session.prompt_stream(message):
            full += chunk
            yield full, session.session_id or session_id
    except Exception as e:
        traceback.print_exc()
        yield f"**Error:** {e}", session_id


def _heartbeat_message(elapsed: float) -> str:
    seconds = int(elapsed)
    return f"*Generating... still working ({seconds}s elapsed).*"


def _controls(visible=True):
    if not visible:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def _build_model_message(prompt_text: str) -> str:
    return f"{prompt_text.rstrip()}\n\n{CODE_ONLY_SUFFIX}\n"


# ---------------------------------------------------------------------------
# Outputs tuple order (11 elements):
#   prompt_panel, output_panel,
#   session_state, prompt_index_state, outputs_state, current_prompt_state,
#   approve_btn, regenerate_btn, skip_btn, status_md, custom_prompt_box
# ---------------------------------------------------------------------------

def send_prompt(provider, session_id, model, system_prompt,
                prompt_index, saved_outputs, mode, custom_prompt_text):
    """Send the current prompt (default or custom) and stream the Lean output."""
    if mode == "Default Pipeline":
        if prompt_index >= TOTAL_PROMPTS:
            done_msg = (
                f"All **{TOTAL_PROMPTS}** prompts processed.  \n"
                f"**{len(saved_outputs)}** outputs saved.  \n"
                "Results autosaved to `lean_output_data.json`."
            )
            yield ("*Pipeline complete.*", done_msg,
                   session_id, prompt_index, saved_outputs, "",
                   *_controls(False),
                   _status(prompt_index, len(saved_outputs), mode, "DONE"),
                   gr.update())
            return
        prompt_text = PROMPTS[prompt_index]
        header = f"### Prompt {prompt_index + 1} / {TOTAL_PROMPTS}\n\n---\n\n"
        prompt_display = header + prompt_text
    else:
        if not custom_prompt_text or not custom_prompt_text.strip():
            yield ("*Enter a custom prompt below and click Generate.*", "*Waiting...*",
                   session_id, prompt_index, saved_outputs, "",
                   *_controls(False),
                   _status(prompt_index, len(saved_outputs), mode),
                   gr.update())
            return
        prompt_text = custom_prompt_text.strip()
        prompt_display = f"### Custom Prompt\n\n---\n\n{prompt_text}"

    message = _build_model_message(prompt_text)

    st = _status(prompt_index, len(saved_outputs), mode)

    yield (prompt_display, "*Generating...*",
           session_id, prompt_index, saved_outputs, prompt_text,
           *_controls(False), st, gr.update())

    lean_text = ""
    last_sid = session_id
    updates: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def worker():
        try:
            for text, sid in _stream(provider, message, session_id, model, system_prompt):
                updates.put(("chunk", (text, sid)))
        finally:
            updates.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    started_at = time.monotonic()
    last_heartbeat = started_at
    heartbeat_interval = 2.0

    while True:
        try:
            event, payload = updates.get(timeout=0.2)
        except queue.Empty:
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                status_text = lean_text or _heartbeat_message(now - started_at)
                yield (prompt_display, status_text,
                       last_sid, prompt_index, saved_outputs, prompt_text,
                       *_controls(False), st, gr.update())
                last_heartbeat = now
            continue

        if event == "chunk":
            text, sid = payload
            lean_text = text
            last_sid = sid
            yield (prompt_display, text,
                   sid, prompt_index, saved_outputs, prompt_text,
                   *_controls(False), st, gr.update())
        elif event == "done":
            break

    yield (prompt_display, lean_text,
           last_sid, prompt_index, saved_outputs, prompt_text,
           *_controls(True), st, gr.update())


def on_approve(output_panel, provider, session_id, model, system_prompt,
               prompt_index, saved_outputs, current_prompt, mode, custom_prompt_text):
    """Save the current output and advance to the next prompt."""
    entry = {"prompt": current_prompt or "", "output": output_panel or ""}
    saved_outputs = saved_outputs + [entry]
    _autosave_outputs(saved_outputs)

    if mode == "Default Pipeline":
        prompt_index += 1
        yield from send_prompt(
            provider, None, model, system_prompt,
            prompt_index, saved_outputs, mode, custom_prompt_text,
        )
    else:
        yield ("*Enter a custom prompt below and click Generate.*", "*Waiting...*",
               None, prompt_index, saved_outputs, "",
               *_controls(False),
               _status(prompt_index, len(saved_outputs), mode, "SAVED"),
               gr.update(value=""))


def on_regenerate(prompt_panel, provider, session_id, model, system_prompt,
                  prompt_index, saved_outputs, current_prompt, mode, custom_prompt_text):
    """Regenerate the current prompt output."""
    yield from send_prompt(
        provider, None, model, system_prompt,
        prompt_index, saved_outputs, mode, custom_prompt_text,
    )


def on_skip(provider, session_id, model, system_prompt,
            prompt_index, saved_outputs, mode, custom_prompt_text):
    """Skip this prompt without saving and advance."""
    if mode == "Default Pipeline":
        prompt_index += 1
    yield from send_prompt(
        provider, None, model, system_prompt,
        prompt_index, saved_outputs, mode, custom_prompt_text,
    )


def clear_session():
    saved = _load_saved_outputs()
    return (
        "*Waiting to start...*", "*Waiting...*",
        None, 0, saved, "",
        *_controls(False),
        _status(0, len(saved), "Default Pipeline",
                "LOADED FROM DISK" if saved else ""),
        gr.update(value=""),
    )


def update_model_dropdown(provider):
    choices = PROVIDER_MODELS[provider]
    return gr.update(choices=choices, value=choices[0])


def on_provider_change(provider):
    return (update_model_dropdown(provider), *clear_session())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_INITIAL_SAVED = _load_saved_outputs()
_INITIAL_STATUS = (
    _status(0, len(_INITIAL_SAVED), "Default Pipeline",
            "LOADED FROM DISK" if _INITIAL_SAVED else "")
)

with gr.Blocks(title="Lean Code Generator") as demo:
    gr.Markdown(
        "# Lean Code Generator\n"
        "Feed prompts from `lean_prompt_data.json` to an LLM one at a time, "
        "or enter your own custom prompt.  \n"
        "Review each generated Lean output before approving."
    )

    # --- State ---
    session_state = gr.State(None)
    prompt_index_state = gr.State(0)
    outputs_state = gr.State(_INITIAL_SAVED)
    current_prompt_state = gr.State("")

    # --- Settings ---
    with gr.Row(elem_classes=["settings-row"]):
        provider_dropdown = gr.Dropdown(
            choices=list(PROVIDER_MODELS.keys()),
            value="Claude",
            label="Provider",
            scale=1,
        )
        mode_radio = gr.Radio(
            ["Default Pipeline", "Custom"],
            value="Default Pipeline",
            label="Mode",
            info="Default: iterates through lean_prompt_data.json. Custom: enter your own prompt.",
            scale=2,
        )
        model_dropdown = gr.Dropdown(
            choices=PROVIDER_MODELS["Claude"],
            value="sonnet",
            label="Model",
            scale=1,
        )
    system_prompt_box = gr.Textbox(
        label="System prompt",
        placeholder="e.g. You are a Lean 4 expert...",
        info="Optional system prompt passed to the LLM.",
    )

    # --- Custom prompt input ---
    custom_prompt_box = gr.Textbox(
        label="Custom prompt",
        placeholder="Enter your Lean generation prompt here...",
        lines=6,
        visible=True,
    )

    # --- Side-by-side panels ---
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("PROMPT", elem_classes=["panel-label"])
            prompt_panel = gr.Markdown(
                "*Waiting to start...*",
                elem_classes=["scroll-panel"],
                max_height="65vh",
                min_height="200px",
            )
        with gr.Column(scale=1):
            gr.Markdown("LEAN OUTPUT", elem_classes=["panel-label"])
            output_panel = gr.Textbox(
                value="*Waiting...*",
                lines=24,
                max_lines=40,
                interactive=False,
                autoscroll=False,
            )

    # --- Review buttons ---
    with gr.Row(elem_classes=["review-row"]):
        approve_btn = gr.Button("Approve", variant="primary", visible=False, min_width=140)
        regenerate_btn = gr.Button("Regenerate", variant="secondary", visible=False, min_width=140)
        skip_btn = gr.Button("Skip", variant="stop", visible=False, min_width=140)

    status_md = gr.Markdown(_INITIAL_STATUS, elem_classes=["status-bar"])

    # --- Action buttons ---
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        clear_btn = gr.Button("New Session")

    # --- Shared output list ---
    panel_outputs = [
        prompt_panel, output_panel,
        session_state, prompt_index_state, outputs_state, current_prompt_state,
        approve_btn, regenerate_btn, skip_btn, status_md, custom_prompt_box,
    ]

    # --- Show/hide custom prompt box based on mode ---
    def toggle_custom_prompt(mode):
        return gr.update(visible=(mode == "Custom"))

    mode_radio.change(toggle_custom_prompt, inputs=[mode_radio], outputs=[custom_prompt_box])

    # --- Events ---
    provider_dropdown.change(
        on_provider_change,
        inputs=[provider_dropdown],
        outputs=[model_dropdown, *panel_outputs],
    )

    generate_btn.click(
        send_prompt,
        inputs=[provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, outputs_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, outputs_state, current_prompt_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    regenerate_btn.click(
        on_regenerate,
        inputs=[prompt_panel, provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, outputs_state, current_prompt_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    skip_btn.click(
        on_skip,
        inputs=[provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, outputs_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    clear_btn.click(clear_session, outputs=panel_outputs)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
