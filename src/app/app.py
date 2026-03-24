"""
Gradio QA Dataset Builder powered by Claude or GPT CLI.

Side-by-side layout: input reasoning (left) vs generated QA pairs (right),
with an automatic Likert-scale alignment evaluation below.
Human review (Approve / Edit / Reject) at every step.

Run: python src/app/app.py
"""

import sys
import json
import tempfile
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "llm"))
sys.path.insert(0, str(_SRC / "qa"))

from claude_cli import ClaudeSession
from gpt_cli import GPTSession
from qa_prompt_generation import default_few_shot_prompt_generation
from qa_postprocessing import postprocess_raw_dataset

DEFAULT_PROMPTS = default_few_shot_prompt_generation()
TOTAL_PROMPTS = len(DEFAULT_PROMPTS)

LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

_EVAL_PREAMBLE = (
    "You are an expert evaluator. Given the INPUT REASONING and the GENERATED QA PAIRS below, "
    "determine how well the outputted QA pairs align with the input reasoning on a Likert scale.\n\n"
    "Use this scale:\n"
    "  1 — Poor: QA pairs are largely unrelated or misrepresent the input reasoning.\n"
    "  2 — Below Average: Some connection, but significant gaps or inaccuracies.\n"
    "  3 — Average: Reasonable alignment, but missing key aspects of the reasoning.\n"
    "  4 — Good: Strong alignment with only minor gaps or imprecisions.\n"
    "  5 — Excellent: QA pairs fully and accurately capture the input reasoning.\n\n"
    "Rate each QA pair individually, then give an **Overall** score. Be concise.\n\n---\n\n"
)

CSS = """
.gradio-container {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container .prose,
.gradio-container .md,
.gradio-container .panel-label,
.gradio-container .status-bar {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container .panel-label,
.gradio-container button,
.gradio-container label {
    font-weight: 700;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .prose,
.gradio-container .md,
.gradio-container .status-bar {
    font-weight: 600;
}

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

.eval-panel {
    border: 2px solid var(--color-accent, #6366f1);
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 0.95em;
    line-height: 1.6;
}
.eval-panel .prose { max-width: none !important; }

.review-row { justify-content: center !important; gap: 16px !important; padding: 12px 0; }
.status-bar { text-align: center; padding: 4px 0; font-size: 0.9em; }
.settings-row { align-items: end !important; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(prompt_index, approved_pairs, mode, extra=""):
    n = len(approved_pairs) if isinstance(approved_pairs, list) else 0
    parts = [f"**{n}** pairs approved"]
    if mode == "Default Pipeline":
        idx = min(prompt_index + 1, TOTAL_PROMPTS)
        parts.append(f"Prompt **{idx} / {TOTAL_PROMPTS}**")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def _build_refinement_prompt(input_text, current_output, user_instruction):
    return (
        "Revise the GENERATED QA PAIRS so they align better with the INPUT REASONING.\n\n"
        "Keep the task domain and content grounded in the input reasoning.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        "Return only the revised QA pairs.\n\n"
        "INPUT REASONING:\n"
        + (input_text or "")
        + "\n\nCURRENT GENERATED QA PAIRS:\n"
        + (current_output or "")
        + "\n\nREVISION REQUEST:\n"
        + user_instruction.strip()
    )


def _unwrap_model(model):
    return model[0] if isinstance(model, list) else model


def _normalize_model(provider, model):
    model = _unwrap_model(model)
    return model or None


def _load_gpt_models():
    cache_path = Path.home() / ".codex" / "models_cache.json"
    fallback_models = [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.2",
    ]

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback_models

    models = [
        model.get("slug")
        for model in data.get("models", [])
        if isinstance(model, dict) and model.get("visibility") == "list" and model.get("slug")
    ]
    return models or fallback_models


def _get_session_cls(provider):
    return GPTSession if provider == "GPT" else ClaudeSession


def _make_session(provider, session_id, model, system_prompt):
    model = _normalize_model(provider, model)
    system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
    session_cls = _get_session_cls(provider)
    if session_id:
        return session_cls.resume(
            session_id, model=model, system_prompt=system_prompt or None,
        )
    return session_cls(model=model, system_prompt=system_prompt or None)


def _stream(provider, message, session_id, model, system_prompt):
    """Yield (text_so_far, session_id) as the selected provider streams its reply."""
    session = _make_session(provider, session_id, model, system_prompt)
    full = ""
    try:
        for chunk in session.prompt_stream(message):
            full += chunk
            yield full, session.session_id or session_id
    except Exception as e:
        traceback.print_exc()
        yield f"**Error:** {e}", session_id


def _eval_stream(provider, input_text, output_text, model):
    """Yield evaluation text chunks using a separate one-shot session."""
    # Build prompt via concatenation (no .format()) to avoid issues with
    # LaTeX curly braces in input/output text.
    eval_prompt = (
        _EVAL_PREAMBLE
        + "INPUT REASONING:\n" + input_text
        + "\n\n---\n\nGENERATED QA PAIRS:\n" + output_text
    )
    session_cls = _get_session_cls(provider)
    session = session_cls(model=_normalize_model(provider, model), max_turns=1)
    full = ""
    try:
        for chunk in session.prompt_stream(eval_prompt):
            full += chunk
            yield full
    except Exception as e:
        traceback.print_exc()
        yield f"**Evaluation error:** {e}"


def _controls(editing=False, visible=True):
    if not visible:
        # Keep both "Back to Review" buttons visible while edit mode is active,
        # even during generation/refinement, so the user can always return to review.
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=editing),
            gr.update(visible=editing),
        )
    if editing:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )

EVAL_WAITING = "*Alignment evaluation will appear here after generation completes…*"
EVAL_RUNNING = "*Evaluating alignment…*"

PROVIDER_MODELS = {
    "Claude": ["sonnet", "opus", "haiku"],
    "GPT": _load_gpt_models(),
}


# ---------------------------------------------------------------------------
# Outputs tuple order (14 elements):
#   input_panel, output_panel, eval_panel,
#   session_state, prompt_index_state, approved_state, current_input_state,
#   edit_mode_state, approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md
# ---------------------------------------------------------------------------

def send_default_prompt(provider, session_id, model, system_prompt, prompt_index, approved_pairs, edit_mode):
    """Send the next default-pipeline prompt, stream QA, then stream evaluation."""
    if prompt_index >= TOTAL_PROMPTS:
        done = (
            f"All **{TOTAL_PROMPTS}** prompts processed.  \n"
            f"**{len(approved_pairs)}** pairs approved.  \n"
            "Click **Download Dataset** to save."
        )
        yield ("*Pipeline complete.*", done, "",
               session_id, prompt_index, approved_pairs, "", False,
               *_controls(False, False), _status(prompt_index, approved_pairs, "Default Pipeline", "DONE"))
        return

    message = DEFAULT_PROMPTS[prompt_index]
    header = f"### Prompt {prompt_index + 1} / {TOTAL_PROMPTS}\n\n---\n\n"
    input_display = header + message
    st = _status(prompt_index, approved_pairs, "Default Pipeline")

    # Phase 1: stream QA generation
    yield (input_display, "*Generating…*", EVAL_WAITING,
           session_id, prompt_index, approved_pairs, message, False,
           *_controls(False, False), st)

    qa_text = ""
    last_sid = session_id
    for text, sid in _stream(provider, message, session_id, model, system_prompt):
        qa_text = text
        last_sid = sid
        yield (input_display, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, message, edit_mode,
               *_controls(edit_mode, True), st)

    # Phase 2: stream alignment evaluation
    yield (input_display, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, message, edit_mode,
           *_controls(edit_mode, True), st)

    for eval_text in _eval_stream(provider, message, qa_text, model):
        yield (input_display, qa_text, eval_text,
               last_sid, prompt_index, approved_pairs, message, edit_mode,
               *_controls(edit_mode, True), st)


def on_approve(output_panel, provider, session_id, model, system_prompt,
               prompt_index, approved_pairs, current_input, mode):
    """Save the current pair and auto-advance (default mode)."""
    approved_pairs = approved_pairs + [{"input": current_input or "", "output": output_panel or ""}]

    if mode == "Default Pipeline":
        prompt_index += 1
        yield from send_default_prompt(
            provider, session_id, model, system_prompt, prompt_index, approved_pairs, False,
        )
    else:
        yield ("*Paste your next reasoning input below.*", "*Waiting…*", "",
               session_id, prompt_index, approved_pairs, "", False,
               *_controls(False, False), _status(prompt_index, approved_pairs, mode))


def on_reject(input_panel, provider, session_id, model, system_prompt,
              prompt_index, approved_pairs, current_input, mode, edit_mode):
    """Reject, retry, then re-evaluate."""
    retry = "The previous output was rejected. Please try again with a different approach."
    st = _status(prompt_index, approved_pairs, mode)

    yield (input_panel, "*Regenerating…*", EVAL_WAITING,
           session_id, prompt_index, approved_pairs, current_input, False,
           *_controls(False, False), st)

    qa_text = ""
    last_sid = session_id
    for text, sid in _stream(provider, retry, session_id, model, system_prompt):
        qa_text = text
        last_sid = sid
        yield (input_panel, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)

    # Evaluate the retry
    yield (input_panel, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, True), st)

    for eval_text in _eval_stream(provider, current_input, qa_text, model):
        yield (input_panel, qa_text, eval_text,
               last_sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)


def on_edit(prompt_index, approved_pairs, mode):
    """Enter edit mode and show a way back to review controls."""
    return (
        True,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        _status(prompt_index, approved_pairs, mode, "EDITING — type a refinement below"),
    )


def on_back_from_edit(prompt_index, approved_pairs, mode):
    """Return from edit mode to the normal review controls."""
    return (
        False,
        *_controls(False, True),
        _status(prompt_index, approved_pairs, mode),
    )


def capture_msg(message):
    """Clear the textbox and stash the message for the next handler."""
    return "", message


def user_chat_submit(message, input_panel, output_panel, provider, session_id, model, system_prompt,
                     prompt_index, approved_pairs, current_input, mode, edit_mode):
    """Handle typed messages: new custom input or edit refinement, then evaluate."""
    if not message or not message.strip():
        yield (input_panel, gr.update(), gr.update(),
               session_id, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, False), _status(prompt_index, approved_pairs, mode))
        return

    is_new_custom = (mode == "Custom" and not current_input)
    st = _status(prompt_index, approved_pairs, mode)

    if is_new_custom:
        current_input = message
        input_display = f"### Custom Input\n\n---\n\n{message}"
        llm_message = message
    else:
        input_display = input_panel
        llm_message = _build_refinement_prompt(current_input, output_panel, message)

    yield (input_display, "*Generating…*" if is_new_custom else "*Refining…*", EVAL_WAITING,
           session_id, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, False), st)

    qa_text = ""
    last_sid = session_id
    for text, sid in _stream(provider, llm_message, session_id, model, system_prompt):
        qa_text = text
        last_sid = sid
        yield (input_display, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)

    # Evaluate
    yield (input_display, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, True), st)

    for eval_text in _eval_stream(provider, current_input, qa_text, model):
        yield (input_display, qa_text, eval_text,
               last_sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)


# ---------------------------------------------------------------------------
# Download / reset
# ---------------------------------------------------------------------------

def download_dataset(approved_pairs):
    if not approved_pairs:
        return gr.update()
    cleaned = postprocess_raw_dataset(approved_pairs)
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = app_data_dir / "qa_data.json"
    canonical_path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return gr.update(value=f"Saved postprocessed dataset to `{canonical_path}`")


def clear_session():
    return (
        "*Waiting to start…*",
        "*Waiting…*",
        "",
        None, 0, [], "", False,
        *_controls(False, False), "",
    )


def update_model_dropdown(provider):
    choices = PROVIDER_MODELS[provider]
    return gr.update(choices=choices, value=choices[0])


def on_provider_change(provider):
    return (update_model_dropdown(provider), *clear_session())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="QA Dataset Builder") as demo:
    gr.Markdown(
        "# QA Dataset Builder\n"
        "Generate and curate QA pairs from reasoning data using Claude or GPT.  \n"
        "Review every generated pair side-by-side before approving."
    )

    # --- State ---
    session_state = gr.State(None)
    prompt_index_state = gr.State(0)
    approved_state = gr.State([])
    current_input_state = gr.State("")
    edit_mode_state = gr.State(False)
    pending_msg_state = gr.State("")

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
            info="Default: auto-sends physics derivation prompts. Custom: paste your own.",
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
        placeholder="e.g. You are a physics QA expert…",
        info="Claude: appended to built-in prompt and CLAUDE.md. GPT: prepended by the local wrapper.",
    )

    # --- Side-by-side comparison ---
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("INPUT REASONING", elem_classes=["panel-label"])
            input_panel = gr.Markdown(
                "*Waiting to start…*",
                latex_delimiters=LATEX,
                elem_classes=["scroll-panel"],
                max_height="65vh",
                min_height="200px",
            )
        with gr.Column(scale=1):
            gr.Markdown("GENERATED QA PAIRS", elem_classes=["panel-label"])
            output_panel = gr.Markdown(
                "*Waiting…*",
                latex_delimiters=LATEX,
                elem_classes=["scroll-panel"],
                max_height="65vh",
                min_height="200px",
            )

    # --- Alignment evaluation ---
    gr.Markdown("ALIGNMENT EVALUATION", elem_classes=["panel-label"])
    eval_panel = gr.Markdown(
        "",
        latex_delimiters=LATEX,
        elem_classes=["eval-panel"],
        max_height="40vh",
    )

    # --- Review buttons ---
    with gr.Row(elem_classes=["review-row"]):
        approve_btn = gr.Button("Approve", variant="primary", visible=False, min_width=140)
        edit_btn = gr.Button("Edit", variant="secondary", visible=False, min_width=140)
        reject_btn = gr.Button("Reject", variant="stop", visible=False, min_width=140)
        back_btn = gr.Button("Back to Review", visible=False, min_width=160)

    status_md = gr.Markdown("", elem_classes=["status-bar"])

    # --- Chat input (custom / edit) ---
    msg = gr.Textbox(
        label="Message",
        placeholder="Custom mode: paste reasoning here  ·  Edit mode: describe changes…",
        autofocus=True,
    )
    cancel_edit_btn = gr.Button("Back to Review", variant="secondary", visible=False)

    # --- Action buttons ---
    with gr.Row():
        start_btn = gr.Button("Start Default Pipeline", variant="primary")
        download_btn = gr.Button("Save Dataset")
        clear_btn = gr.Button("New Session")

    # --- Shared output list ---
    panel_outputs = [
        input_panel, output_panel, eval_panel,
        session_state, prompt_index_state, approved_state, current_input_state, edit_mode_state,
        approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md,
    ]

    # --- Events ---

    provider_dropdown.change(
        on_provider_change,
        inputs=[provider_dropdown],
        outputs=[model_dropdown, *panel_outputs],
    )

    start_btn.click(
        send_default_prompt,
        inputs=[provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, approved_state, edit_mode_state],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, approved_state, current_input_state, mode_radio],
        outputs=panel_outputs,
    )

    edit_btn.click(
        on_edit,
        inputs=[prompt_index_state, approved_state, mode_radio],
        outputs=[edit_mode_state, approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md],
    )

    back_btn.click(
        on_back_from_edit,
        inputs=[prompt_index_state, approved_state, mode_radio],
        outputs=[edit_mode_state, approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md],
    )

    cancel_edit_btn.click(
        on_back_from_edit,
        inputs=[prompt_index_state, approved_state, mode_radio],
        outputs=[edit_mode_state, approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md],
    )

    reject_btn.click(
        on_reject,
        inputs=[input_panel, provider_dropdown, session_state, model_dropdown, system_prompt_box,
                prompt_index_state, approved_state, current_input_state, mode_radio, edit_mode_state],
        outputs=panel_outputs,
    )

    msg.submit(
        capture_msg,
        inputs=[msg],
        outputs=[msg, pending_msg_state],
    ).then(
        user_chat_submit,
        inputs=[pending_msg_state, input_panel, output_panel, provider_dropdown, session_state, model_dropdown,
                system_prompt_box, prompt_index_state, approved_state,
                current_input_state, mode_radio, edit_mode_state],
        outputs=panel_outputs,
    )

    download_btn.click(download_dataset, inputs=[approved_state], outputs=[status_md])
    clear_btn.click(clear_session, outputs=panel_outputs)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
