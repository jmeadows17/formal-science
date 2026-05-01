"""
Gradio QA Dataset Builder powered by Claude or GPT CLI.

Side-by-side layout: input reasoning (left) vs generated QA pairs (right),
with an automatic Likert-scale alignment evaluation below.
Human review (Approve / Edit / Reject) at every step.

Run: python src/app/app.py
"""

import sys
import difflib
import json
import re
import tempfile
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "llm"))
sys.path.insert(0, str(_SRC / "qa"))

from claude_cli import ClaudeSession, CLAUDE_REASONING_EFFORTS
from gpt_cli import GPTSession, VALID_REASONING_EFFORTS
from lean_prompts import build_lean_prompt_dataset
from qa_prompt_generation import default_few_shot_prompt_generation
from qa_postprocessing import postprocess_raw_dataset

_DEFAULT_PROMPT_BODY_MARKER = "Now, the following **equation-only** derivations"


def _split_default_prompt(prompt):
    marker_idx = prompt.find(_DEFAULT_PROMPT_BODY_MARKER)
    if marker_idx < 0:
        return "", prompt.strip()
    return prompt[:marker_idx].strip(), prompt[marker_idx:].strip()


_DEFAULT_PROMPT_TEMPLATES = default_few_shot_prompt_generation()
_DEFAULT_PROMPT_PARTS = [_split_default_prompt(prompt) for prompt in _DEFAULT_PROMPT_TEMPLATES]
DEFAULT_FEW_SHOT_PROMPT = next((few_shot for few_shot, _ in _DEFAULT_PROMPT_PARTS if few_shot), "")
DEFAULT_PROMPTS = [body for _, body in _DEFAULT_PROMPT_PARTS]
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
    "If a patch difference is provided, begin with a brief **Patch Difference** note summarizing "
    "what changed and whether the change is substantive; explicitly say if there was no meaningful "
    "change.\n"
    "Rate each QA pair individually, then give an **Overall** score. Be concise.\n\n---\n\n"
)

_REASONING_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh")
_DEFAULT_CLAUDE_REASONING_EFFORT = "medium"

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


def _approved_entries_to_batches(approved_pairs):
    """Convert persisted and in-memory approved entries into cleaned QA batches."""
    batches = []
    for entry in approved_pairs or []:
        if not isinstance(entry, dict):
            continue
        if "batch" in entry:
            batch = entry.get("batch")
            if isinstance(batch, list) and batch:
                batches.append(batch)
            continue

        if "input" in entry and "output" in entry:
            parsed_batches = postprocess_raw_dataset([entry])
            if parsed_batches:
                batches.extend(parsed_batches)
    return batches


def _load_saved_progress():
    """Load saved QA batches and represent them as approved entries."""
    qa_path = _SRC / "app_data" / "qa_data.json"
    try:
        data = json.loads(qa_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []
    return [{"batch": batch} for batch in data if isinstance(batch, list) and batch]


def _autosave_datasets(approved_pairs):
    """Persist the cleaned QA dataset and derived Lean prompts to app_data."""
    if not approved_pairs:
        return None, None

    cleaned = _approved_entries_to_batches(approved_pairs)
    lean_prompt_data = build_lean_prompt_dataset(cleaned)
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    qa_path = app_data_dir / "qa_data.json"
    lean_prompt_path = app_data_dir / "lean_prompt_data.json"

    qa_path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lean_prompt_path.write_text(
        json.dumps(lean_prompt_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return qa_path, lean_prompt_path


def _normalize_review_feedback(feedback: str | None, waiting_text: str, running_text: str) -> str:
    normalized = (feedback or "").strip()
    if not normalized or normalized in {waiting_text, running_text}:
        return ""
    return normalized


def _build_brief_patch_diff(
    previous_text: str | None,
    current_text: str,
    *,
    from_label: str,
    to_label: str,
    context_lines: int = 1,
    max_lines: int = 40,
) -> str:
    if previous_text is None:
        return ""

    previous_lines = (previous_text or "").splitlines()
    current_lines = (current_text or "").splitlines()
    if previous_lines == current_lines:
        return "No textual differences detected."

    diff_lines = list(difflib.unified_diff(
        previous_lines,
        current_lines,
        fromfile=from_label,
        tofile=to_label,
        lineterm="",
        n=context_lines,
    ))
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... diff truncated ..."]
    return "\n".join(diff_lines)


def _build_refinement_prompt(input_text, current_output, user_instruction, evaluation_feedback=None):
    sections = [
        "Revise the GENERATED QA PAIRS so they align better with the INPUT REASONING.",
        "Keep the task domain and content grounded in the input reasoning.\n"
        "Make every generated question fully self-contained; do not rely on previous questions, "
        "previous results, or unstated context.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        "Return only the revised QA pairs.",
        "INPUT REASONING:\n" + (input_text or ""),
        "CURRENT GENERATED QA PAIRS:\n" + (current_output or ""),
    ]

    normalized_feedback = _normalize_review_feedback(
        evaluation_feedback,
        EVAL_WAITING,
        EVAL_RUNNING,
    )
    if normalized_feedback:
        sections.append("LATEST ALIGNMENT EVALUATION:\n" + normalized_feedback)

    sections.append("REVISION REQUEST:\n" + user_instruction.strip())
    return "\n\n".join(sections)


def _build_retry_prompt(input_text, rejected_output, evaluation_feedback=None):
    sections = [
        "The previous QA output was rejected. Generate a new version that better follows the "
        "input reasoning.",
        "Keep the task domain and content grounded in the input reasoning.\n"
        "Make every generated question fully self-contained; do not rely on previous questions, "
        "previous results, or unstated context.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        "Return only the regenerated QA pairs.",
        "INPUT REASONING:\n" + (input_text or ""),
        "PREVIOUS REJECTED OUTPUT:\n" + (rejected_output or ""),
    ]

    normalized_feedback = _normalize_review_feedback(
        evaluation_feedback,
        EVAL_WAITING,
        EVAL_RUNNING,
    )
    if normalized_feedback:
        sections.append("LATEST ALIGNMENT EVALUATION:\n" + normalized_feedback)

    return "\n\n".join(sections)


def _maybe_prepend_few_shot_examples(prompt_text, include_few_shot_examples):
    if not include_few_shot_examples or not DEFAULT_FEW_SHOT_PROMPT:
        return prompt_text
    return DEFAULT_FEW_SHOT_PROMPT + "\n\n" + (prompt_text or "").lstrip()


def _unwrap_model(model):
    return model[0] if isinstance(model, list) else model


def _normalize_model(provider, model):
    model = _unwrap_model(model)
    return model or None


def _load_gpt_model_metadata():
    cache_path = Path.home() / ".codex" / "models_cache.json"
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    metadata = {}
    for model in data.get("models", []):
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if model.get("visibility") != "list" or not slug:
            continue
        metadata[slug] = model
    return metadata


def _load_gpt_models():
    fallback_models = [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.2",
    ]
    return list(_GPT_MODEL_METADATA) or fallback_models


def _load_codex_reasoning_effort():
    config_path = Path.home() / ".codex" / "config.toml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r'^\s*model_reasoning_effort\s*=\s*"([^"]+)"', text, re.MULTILINE)
    effort = match.group(1) if match else None
    return effort if effort in VALID_REASONING_EFFORTS else None


def _supported_reasoning_efforts(provider, model):
    if provider == "Claude":
        return list(CLAUDE_REASONING_EFFORTS)

    metadata = _GPT_MODEL_METADATA.get(_unwrap_model(model), {})
    supported = []
    for level in metadata.get("supported_reasoning_levels", []):
        effort = level.get("effort") if isinstance(level, dict) else None
        if effort in _REASONING_EFFORT_ORDER and effort not in supported:
            supported.append(effort)
    return supported or ["low", "medium", "high", "xhigh"]


def _preferred_reasoning_effort(provider, model, current_effort=None):
    supported = _supported_reasoning_efforts(provider, model)
    current = _unwrap_model(current_effort)
    if current in supported:
        return current

    if provider == "Claude":
        if _DEFAULT_CLAUDE_REASONING_EFFORT in supported:
            return _DEFAULT_CLAUDE_REASONING_EFFORT
        return supported[0] if supported else None

    if _CODEX_DEFAULT_REASONING_EFFORT in supported:
        return _CODEX_DEFAULT_REASONING_EFFORT

    default_effort = _GPT_MODEL_METADATA.get(_unwrap_model(model), {}).get("default_reasoning_level")
    if default_effort in supported:
        return default_effort

    return supported[0] if supported else None


def _normalize_reasoning_effort(provider, model, reasoning_effort):
    effort = _preferred_reasoning_effort(provider, model, reasoning_effort)
    return effort if effort in _supported_reasoning_efforts(provider, model) else None


def update_reasoning_effort_dropdown(provider, model, current_effort=None):
    choices = _supported_reasoning_efforts(provider, model)
    if not choices:
        return gr.update(visible=False)

    return gr.update(
        choices=choices,
        value=_preferred_reasoning_effort(provider, model, current_effort),
        visible=True,
    )


def on_model_change(provider, model, current_effort):
    return update_reasoning_effort_dropdown(provider, model, current_effort)


def _get_session_cls(provider):
    return GPTSession if provider == "GPT" else ClaudeSession


def _make_session(provider, session_id, model, reasoning_effort=None, **kwargs):
    model = _normalize_model(provider, model)
    session_cls = _get_session_cls(provider)
    session_kwargs = {"model": model, **kwargs}
    normalized_effort = _normalize_reasoning_effort(provider, model, reasoning_effort)
    if normalized_effort:
        session_kwargs["reasoning_effort"] = normalized_effort
    if session_id:
        return session_cls.resume(session_id, **session_kwargs)
    return session_cls(**session_kwargs)


def _stream(provider, message, session_id, model, reasoning_effort):
    """Yield (text_so_far, session_id) as the selected provider streams its reply."""
    session = _make_session(provider, session_id, model, reasoning_effort)
    full = ""
    try:
        for chunk in session.prompt_stream(message):
            full += chunk
            yield full, session.session_id or session_id
    except Exception as e:
        traceback.print_exc()
        yield f"**Error:** {e}", session_id


def _eval_stream(provider, input_text, output_text, model, reasoning_effort, previous_output=None, session_id=None):
    """Yield (text_so_far, session_id) as the selected provider streams the evaluation."""
    # Build prompt via concatenation (no .format()) to avoid issues with
    # LaTeX curly braces in input/output text.
    sections = [
        _EVAL_PREAMBLE
        + "INPUT REASONING:\n" + input_text
        + "\n\n---\n\nGENERATED QA PAIRS:\n" + output_text
    ]
    patch_diff = _build_brief_patch_diff(
        previous_output,
        output_text,
        from_label="previous_output",
        to_label="current_output",
    )
    if patch_diff:
        sections.append("\n\n---\n\nPATCH DIFFERENCE FROM PREVIOUS OUTPUT:\n```diff\n" + patch_diff + "\n```")
    eval_prompt = "".join(sections)
    session = _make_session(provider, session_id, model, reasoning_effort, max_turns=1)
    full = ""
    try:
        for chunk in session.prompt_stream(eval_prompt):
            full += chunk
            yield full, session.session_id or session_id
    except Exception as e:
        traceback.print_exc()
        yield f"**Evaluation error:** {e}", session_id


# ---------------------------------------------------------------------------
# Structured QA extraction & verification
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT_RANGED = (
    "The original prompt asked you to generate QA pairs Q{start} through Q{end}. "
    "Extract ONLY those {count} generated QA pairs from your previous response "
    "as a JSON array. Do not include any few-shot examples that were provided "
    "in the prompt.\n"
    "Each element must have exactly two keys: \"question\" and \"answer\".\n"
    "For each question and answer, preserve the text EXACTLY as written, "
    "including all LaTeX, whitespace, and formatting, EXCEPT omit a single "
    "leading QA label if present. That means remove only an initial marker like "
    "\"Q{start}:\", \"A{start}:\", \"Qn:\", or \"An:\", including optional bold "
    "or wrapper formatting such as **Qn:**, \\textbf{{Qn:}}, **An:**, or "
    "\\textbf{{An:}}. Do not rewrite or clean anything else.\n"
    "Return ONLY the raw JSON array. Do not wrap it in code fences. "
    "Do not add any explanation."
)

_EXTRACTION_PROMPT_GENERIC = (
    "Extract the QA pairs you generated in your previous response as a "
    "JSON array. Do not include any examples that were provided to you "
    "in the prompt — only the pairs you wrote.\n"
    "Each element must have exactly two keys: \"question\" and \"answer\".\n"
    "For each question and answer, preserve the text EXACTLY as written, "
    "including all LaTeX, whitespace, and formatting, EXCEPT omit a single "
    "leading QA label if present. That means remove only an initial marker like "
    "\"Qn:\", \"An:\", \"Qn.\", or \"An.\", including optional bold or wrapper "
    "formatting such as **Qn:**, \\textbf{{Qn:}}, **An:**, or \\textbf{{An:}}. "
    "Do not rewrite, renumber, paraphrase, or clean anything else.\n"
    "Return ONLY the raw JSON array. Do not wrap it in code fences. "
    "Do not add any explanation."
)


def _verify_extraction(raw_output, batch, expected_count=None, prompt_text=None):
    """
    Deterministic verification of an extracted QA batch.

    Checks: schema, count, verbatim substring, ordering, few-shot contamination.
    Returns (ok, message).
    """
    if not isinstance(batch, list) or not batch:
        return False, "Empty or invalid batch"

    for i, pair in enumerate(batch):
        if not isinstance(pair, dict):
            return False, f"Pair {i+1} is not a dict"
        if "question" not in pair or "answer" not in pair:
            return False, f"Pair {i+1} missing required keys"
        if not isinstance(pair["question"], str) or not isinstance(pair["answer"], str):
            return False, f"Pair {i+1} has non-string values"

    if expected_count is not None and len(batch) != expected_count:
        return False, f"Expected {expected_count} pairs, got {len(batch)}"

    for i, pair in enumerate(batch):
        if pair["question"] not in raw_output:
            return False, f"Question {i+1} not found verbatim in approved output"
        if pair["answer"] not in raw_output:
            return False, f"Answer {i+1} not found verbatim in approved output"

    last_pos = -1
    for i, pair in enumerate(batch):
        q_pos = raw_output.find(pair["question"], last_pos + 1)
        if q_pos < 0:
            return False, f"Question {i+1} found but out of expected sequence"
        a_pos = raw_output.find(pair["answer"], q_pos)
        if a_pos < 0:
            return False, f"Answer {i+1} does not follow its question in output"
        last_pos = a_pos

    if prompt_text:
        for i, pair in enumerate(batch):
            if pair["question"] in prompt_text:
                return False, f"Question {i+1} matches prompt/few-shot content"
            if pair["answer"] in prompt_text:
                return False, f"Answer {i+1} matches prompt/few-shot content"

    return True, f"{len(batch)} pairs extracted and verified"


def _extract_qa_structured(provider, session_id, model, reasoning_effort,
                           raw_output, current_input, include_few_shot_examples):
    """
    Extract QA pairs via same-session LLM call and verify deterministically.

    Returns (batch, message) — batch is a list of dicts on success, None on failure.
    """
    if not raw_output or not raw_output.strip():
        return None, "No output to extract from"

    range_match = re.search(r"Q(\d+)\s*-\s*Q(\d+)", current_input or "", re.IGNORECASE)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        expected_count = end - start + 1 if end >= start else None
        extraction_prompt = _EXTRACTION_PROMPT_RANGED.format(
            start=start, end=end, count=expected_count,
        )
    else:
        expected_count = None
        extraction_prompt = _EXTRACTION_PROMPT_GENERIC

    prompt_text = _maybe_prepend_few_shot_examples(current_input, include_few_shot_examples)

    last_reason = "unknown error"
    for attempt in range(2):
        prompt = extraction_prompt
        if attempt > 0:
            prompt = (
                f"Your previous extraction was invalid: {last_reason}. "
                "Try again. " + extraction_prompt
            )

        try:
            session = _make_session(provider, session_id, model, reasoning_effort, max_turns=1)
            response = session.text(prompt)
        except Exception as e:
            last_reason = f"LLM call failed: {e}"
            continue

        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            last_reason = "response was not valid JSON"
            continue

        if not isinstance(parsed, list):
            last_reason = "response is not a JSON array"
            continue

        batch = []
        for p in parsed:
            if isinstance(p, dict) and "question" in p and "answer" in p:
                batch.append({"question": p["question"], "answer": p["answer"]})

        ok, reason = _verify_extraction(raw_output, batch, expected_count, prompt_text)
        if ok:
            return batch, reason
        last_reason = reason

    return None, last_reason


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

_GPT_MODEL_METADATA = _load_gpt_model_metadata()
_CODEX_DEFAULT_REASONING_EFFORT = _load_codex_reasoning_effort()

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

def _message_box_update(mode):
    if mode == "Custom":
        return gr.update(
            value="",
            placeholder="Paste reasoning here. To revise an existing output, click Edit first.",
        )
    return gr.update(
        value="",
        placeholder="Default mode: use Start Default Pipeline  ·  Edit mode: describe changes…",
    )


def _reset_ui_state(mode):
    saved_approved = _load_saved_progress()
    saved_count = len(saved_approved)
    input_text = (
        "*Paste your reasoning input below.*"
        if mode == "Custom"
        else "*Waiting to start…*"
    )
    status = _status(
        saved_count,
        saved_approved,
        mode,
        "LOADED FROM DISK" if saved_count else "",
    )
    return (
        input_text,
        "*Waiting…*",
        "",
        None, saved_count, saved_approved, "", False,
        *_controls(False, False), status,
    )


def start_default_pipeline(provider, model, reasoning_effort, include_few_shot_examples,
                           prompt_index, approved_pairs, edit_mode):
    yield from send_default_prompt(
        provider, None, model, reasoning_effort, include_few_shot_examples,
        prompt_index, approved_pairs, edit_mode,
    )


def on_mode_change(mode):
    return (
        *_reset_ui_state(mode),
        gr.update(visible=(mode == "Default Pipeline")),
        _message_box_update(mode),
    )


def send_default_prompt(provider, session_id, model, reasoning_effort, include_few_shot_examples,
                        prompt_index, approved_pairs, edit_mode):
    """Send the next default-pipeline prompt, stream QA, then stream evaluation."""
    if prompt_index >= TOTAL_PROMPTS:
        done = (
            f"All **{TOTAL_PROMPTS}** prompts processed.  \n"
            f"**{len(approved_pairs)}** pairs approved.  \n"
            "Approved outputs have been autosaved."
        )
        yield ("*Pipeline complete.*", done, "",
               session_id, prompt_index, approved_pairs, "", False,
               *_controls(False, False), _status(prompt_index, approved_pairs, "Default Pipeline", "DONE | AUTOSAVED"))
        return

    message = DEFAULT_PROMPTS[prompt_index]
    llm_message = _maybe_prepend_few_shot_examples(message, include_few_shot_examples)
    header = f"### Prompt {prompt_index + 1} / {TOTAL_PROMPTS}\n\n---\n\n"
    input_display = header + message
    st = _status(prompt_index, approved_pairs, "Default Pipeline")

    # Phase 1: stream QA generation
    yield (input_display, "*Generating…*", EVAL_WAITING,
           session_id, prompt_index, approved_pairs, message, False,
           *_controls(False, False), st)

    qa_text = ""
    last_sid = session_id
    for text, sid in _stream(provider, llm_message, session_id, model, reasoning_effort):
        qa_text = text
        last_sid = sid
        yield (input_display, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, message, edit_mode,
               *_controls(edit_mode, True), st)

    # Phase 2: stream alignment evaluation
    yield (input_display, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, message, edit_mode,
           *_controls(edit_mode, True), st)

    for eval_text, sid in _eval_stream(provider, message, qa_text, model, reasoning_effort, session_id=last_sid):
        last_sid = sid
        yield (input_display, qa_text, eval_text,
               last_sid, prompt_index, approved_pairs, message, edit_mode,
               *_controls(edit_mode, True), st)


def on_approve(output_panel, provider, session_id, model, reasoning_effort,
               include_few_shot_examples, prompt_index, approved_pairs, current_input, mode):
    """Extract structured QA pairs, verify, save, then auto-advance."""
    # Phase 1: show extraction in progress
    yield (gr.update(), gr.update(), gr.update(),
           session_id, prompt_index, approved_pairs, current_input, False,
           *_controls(False, False),
           _status(prompt_index, approved_pairs, mode, "EXTRACTING QA PAIRS\u2026"))

    # Phase 2: LLM extraction + deterministic verification
    batch, extract_msg = _extract_qa_structured(
        provider, session_id, model, reasoning_effort,
        output_panel, current_input, include_few_shot_examples,
    )

    if batch is None:
        # Extraction failed — stay on current output, show error, restore controls
        yield (gr.update(), gr.update(),
               f"**QA Extraction Failed**\n\n{extract_msg}\n\n"
               "Use **Edit** to refine the output or **Reject** to regenerate.",
               session_id, prompt_index, approved_pairs, current_input, False,
               *_controls(False, True),
               _status(prompt_index, approved_pairs, mode, "EXTRACTION FAILED"))
        return

    # Phase 3: save verified batch
    approved_pairs = approved_pairs + [{"batch": batch}]
    _autosave_datasets(approved_pairs)

    # Phase 4: confirm and advance
    verified_status = f"VERIFIED & SAVED \u2014 {extract_msg}"
    if mode == "Default Pipeline":
        prompt_index += 1
        yield (gr.update(), gr.update(),
               f"**Extraction Verified:** {extract_msg}",
               session_id, prompt_index, approved_pairs, current_input, False,
               *_controls(False, False),
               _status(prompt_index, approved_pairs, mode, verified_status))
        yield from send_default_prompt(
            provider, None, model, reasoning_effort, include_few_shot_examples,
            prompt_index, approved_pairs, False,
        )
    else:
        yield ("*Paste your next reasoning input below.*", "*Waiting\u2026*",
               f"**Extraction Verified:** {extract_msg}",
               None, prompt_index, approved_pairs, "", False,
               *_controls(False, False),
               _status(prompt_index, approved_pairs, mode, verified_status))


def on_reject(input_panel, provider, session_id, model, reasoning_effort,
              include_few_shot_examples, prompt_index, approved_pairs, current_input,
              mode, edit_mode, output_panel, eval_panel):
    """Reject, retry, then re-evaluate."""
    retry = _build_retry_prompt(current_input, output_panel, eval_panel)
    llm_retry = _maybe_prepend_few_shot_examples(retry, include_few_shot_examples)
    st = _status(prompt_index, approved_pairs, mode)

    yield (input_panel, "*Regenerating…*", EVAL_WAITING,
           None, prompt_index, approved_pairs, current_input, False,
           *_controls(False, False), st)

    qa_text = ""
    last_sid = None
    for text, sid in _stream(provider, llm_retry, None, model, reasoning_effort):
        qa_text = text
        last_sid = sid
        yield (input_panel, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)

    # Evaluate the retry
    yield (input_panel, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, True), st)

    for eval_text, sid in _eval_stream(
        provider,
        current_input,
        qa_text,
        model,
        reasoning_effort,
        previous_output=output_panel,
        session_id=last_sid,
    ):
        last_sid = sid
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


def user_chat_submit(message, input_panel, output_panel, eval_panel,
                     provider, session_id, model, reasoning_effort,
                     include_few_shot_examples, prompt_index, approved_pairs,
                     current_input, mode, edit_mode):
    """Handle typed messages: new custom input or edit refinement, then evaluate."""
    if not message or not message.strip():
        yield (input_panel, gr.update(), gr.update(),
               session_id, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, False), _status(prompt_index, approved_pairs, mode))
        return

    is_new_custom = (mode == "Custom" and not edit_mode)
    st = _status(prompt_index, approved_pairs, mode)

    if is_new_custom:
        current_input = message
        input_display = f"### Custom Input\n\n---\n\n{message}"
        llm_message = _maybe_prepend_few_shot_examples(message, include_few_shot_examples)
        stream_session_id = None
    else:
        input_display = input_panel
        llm_message = _build_refinement_prompt(current_input, output_panel, message, eval_panel)
        stream_session_id = session_id

    yield (input_display, "*Generating…*" if is_new_custom else "*Refining…*", EVAL_WAITING,
           stream_session_id, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, False), st)

    qa_text = ""
    last_sid = stream_session_id
    for text, sid in _stream(provider, llm_message, stream_session_id, model, reasoning_effort):
        qa_text = text
        last_sid = sid
        yield (input_display, text, EVAL_WAITING,
               sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)

    # Evaluate
    yield (input_display, qa_text, EVAL_RUNNING,
           last_sid, prompt_index, approved_pairs, current_input, edit_mode,
           *_controls(edit_mode, True), st)

    previous_output = None if is_new_custom else output_panel
    for eval_text, sid in _eval_stream(
        provider,
        current_input,
        qa_text,
        model,
        reasoning_effort,
        previous_output=previous_output,
        session_id=last_sid,
    ):
        last_sid = sid
        yield (input_display, qa_text, eval_text,
               last_sid, prompt_index, approved_pairs, current_input, edit_mode,
               *_controls(edit_mode, True), st)


def clear_session():
    return (
        *_reset_ui_state("Default Pipeline"),
        _message_box_update("Default Pipeline"),
    )


_INITIAL_APPROVED = _load_saved_progress()
_INITIAL_PROMPT_INDEX = len(_INITIAL_APPROVED)
_INITIAL_STATUS = (
    _status(_INITIAL_PROMPT_INDEX, _INITIAL_APPROVED, "Default Pipeline", "LOADED FROM DISK")
    if _INITIAL_PROMPT_INDEX
    else ""
)

def on_provider_change(provider, mode, current_effort):
    model_choices = PROVIDER_MODELS[provider]
    model = model_choices[0]
    return (
        gr.update(choices=model_choices, value=model),
        update_reasoning_effort_dropdown(provider, model, current_effort),
        *_reset_ui_state(mode),
        gr.update(visible=(mode == "Default Pipeline")),
        _message_box_update(mode),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_qa_builder_ui():
    gr.Markdown(
        "# QA Dataset Builder\n"
        "Generate and curate QA pairs from reasoning data using Claude or GPT.  \n"
        "Review every generated pair side-by-side before approving."
    )

    # --- State ---
    session_state = gr.State(None)
    prompt_index_state = gr.State(_INITIAL_PROMPT_INDEX)
    approved_state = gr.State(_INITIAL_APPROVED)
    current_input_state = gr.State("")
    edit_mode_state = gr.State(False)
    pending_msg_state = gr.State("")

    # --- Settings ---
    with gr.Row(elem_classes=["settings-row"]):
        default_provider = "GPT"
        default_model = "gpt-5.4" if "gpt-5.4" in PROVIDER_MODELS[default_provider] else PROVIDER_MODELS[default_provider][0]
        provider_dropdown = gr.Dropdown(
            choices=list(PROVIDER_MODELS.keys()),
            value=default_provider,
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
            choices=PROVIDER_MODELS[default_provider],
            value=default_model,
            label="Model",
            scale=1,
        )
        reasoning_effort_dropdown = gr.Dropdown(
            choices=_supported_reasoning_efforts(default_provider, default_model),
            value=_preferred_reasoning_effort(default_provider, default_model, "high"),
            label="Reasoning Effort",
            info="Claude: low/medium/high/max. GPT choices depend on the selected model.",
            visible=True,
            scale=1,
        )
        include_few_shot_checkbox = gr.Checkbox(
            value=True,
            label="Include Few-Shot Examples",
            info="Enabled by default. Few-shot examples are hidden from the displayed prompt but still sent to the LLM.",
            scale=1,
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

    status_md = gr.Markdown(_INITIAL_STATUS, elem_classes=["status-bar"])

    # --- Chat input (custom / edit) ---
    msg = gr.Textbox(
        label="Message",
        placeholder="Default mode: use Start Default Pipeline  ·  Edit mode: describe changes…",
        autofocus=True,
    )
    cancel_edit_btn = gr.Button("Back to Review", variant="secondary", visible=False)

    # --- Action buttons ---
    with gr.Row():
        start_btn = gr.Button("Start Default Pipeline", variant="primary")
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
        inputs=[provider_dropdown, mode_radio, reasoning_effort_dropdown],
        outputs=[model_dropdown, reasoning_effort_dropdown, *panel_outputs, start_btn, msg],
    )

    model_dropdown.change(
        on_model_change,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown],
        outputs=[reasoning_effort_dropdown],
    )

    mode_radio.change(
        on_mode_change,
        inputs=[mode_radio],
        outputs=[*panel_outputs, start_btn, msg],
    )

    start_btn.click(
        start_default_pipeline,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown, include_few_shot_checkbox,
                prompt_index_state, approved_state, edit_mode_state],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                include_few_shot_checkbox,
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
        inputs=[input_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                include_few_shot_checkbox,
                prompt_index_state, approved_state, current_input_state, mode_radio, edit_mode_state,
                output_panel, eval_panel],
        outputs=panel_outputs,
    )

    msg.submit(
        capture_msg,
        inputs=[msg],
        outputs=[msg, pending_msg_state],
    ).then(
        user_chat_submit,
        inputs=[pending_msg_state, input_panel, output_panel, eval_panel, provider_dropdown, session_state,
                model_dropdown, reasoning_effort_dropdown, include_few_shot_checkbox,
                prompt_index_state, approved_state,
                current_input_state, mode_radio, edit_mode_state],
        outputs=panel_outputs,
    )

    clear_btn.click(clear_session, outputs=[*panel_outputs, msg])


def create_qa_demo():
    with gr.Blocks(title="QA Dataset Builder", css=CSS, theme=gr.themes.Soft()) as demo:
        render_qa_builder_ui()
    return demo


def _switch_workspace_view(target: str):
    return (
        gr.update(visible=(target == "qa")),
        gr.update(visible=(target == "lean")),
        gr.update(visible=(target == "postprocessing")),
    )


def create_workbench_demo(initial_view: str = "qa"):
    import lean_app as lean_module
    import postprocessing_app as post_module

    combined_css = "\n".join([CSS, lean_module.CSS, post_module.CSS])
    with gr.Blocks(title="Formal Science Workbench", css=combined_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["review-row"]):
            qa_nav_btn = gr.Button("QA Dataset Builder", variant="primary", min_width=180)
            lean_nav_btn = gr.Button("Lean Code Generator", variant="secondary", min_width=180)
            post_nav_btn = gr.Button("Postprocessing", variant="secondary", min_width=180)

        with gr.Column(visible=(initial_view == "qa")) as qa_view:
            render_qa_builder_ui()

        with gr.Column(visible=(initial_view == "lean")) as lean_view:
            lean_module.render_lean_builder_ui()

        with gr.Column(visible=(initial_view == "postprocessing")) as post_view:
            post_module.render_postprocessing_ui()

        qa_nav_btn.click(lambda: _switch_workspace_view("qa"), outputs=[qa_view, lean_view, post_view])
        lean_nav_btn.click(lambda: _switch_workspace_view("lean"), outputs=[qa_view, lean_view, post_view])
        post_nav_btn.click(
            lambda: _switch_workspace_view("postprocessing"),
            outputs=[qa_view, lean_view, post_view],
        )

    return demo


if __name__ == "__main__":
    create_workbench_demo(initial_view="qa").launch()
