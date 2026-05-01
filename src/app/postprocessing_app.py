"""
Gradio postprocessing app for validated formal-proof splitting.

Run: python src/app/postprocessing_app.py
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "llm"))
sys.path.insert(0, str(_SRC / "app"))

from claude_cli import ClaudeSession, CLAUDE_REASONING_EFFORTS
from gpt_cli import GPTSession, VALID_REASONING_EFFORTS
from postprocessing_pipeline import (
    approved_batch_count,
    reconcile_audit_with_formal_qa,
    build_extraction_prompt,
    evaluate_extraction_response,
    load_formal_qa_data,
    load_postprocessing_audit,
    load_structured_proofs,
    next_batch_index,
    save_postprocessing_audit,
    set_attempt_status,
    upsert_audit_entry,
    write_formal_qa_data_from_audit,
)


REPO_ROOT = _SRC.parent
REASONING_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh")
_DEFAULT_CLAUDE_REASONING_EFFORT = "medium"


def _initialize_structured_entries() -> tuple[list[dict], str | None]:
    try:
        return load_structured_proofs(), None
    except RuntimeError as exc:
        return [], str(exc)


def _set_structured_entries(entries: list[dict], warning: str | None) -> None:
    global STRUCTURED_PROOF_ENTRIES, DATASET_WARNING, TOTAL_BATCHES
    STRUCTURED_PROOF_ENTRIES = entries
    DATASET_WARNING = warning
    TOTAL_BATCHES = len(entries)


def _refresh_structured_entries() -> None:
    entries, warning = _initialize_structured_entries()
    _set_structured_entries(entries, warning)


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


def _load_gpt_models() -> list[str]:
    fallback = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2-codex", "gpt-5.2"]
    return list(_GPT_MODEL_METADATA) or fallback


def _load_codex_reasoning_effort():
    config_path = Path.home() / ".codex" / "config.toml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r'^\s*model_reasoning_effort\s*=\s*"([^"]+)"', text, re.MULTILINE)
    effort = match.group(1) if match else None
    return effort if effort in VALID_REASONING_EFFORTS else None


def _unwrap(val):
    return val[0] if isinstance(val, list) else val


def _normalize_model(provider, model):
    return _unwrap(model) or None


def _supported_reasoning_efforts(provider, model):
    if provider == "Claude":
        return list(CLAUDE_REASONING_EFFORTS)

    metadata = _GPT_MODEL_METADATA.get(_unwrap(model), {})
    supported = []
    for level in metadata.get("supported_reasoning_levels", []):
        effort = level.get("effort") if isinstance(level, dict) else None
        if effort in REASONING_EFFORT_ORDER and effort not in supported:
            supported.append(effort)
    return supported or ["low", "medium", "high", "xhigh"]


def _preferred_reasoning_effort(provider, model, current_effort=None):
    supported = _supported_reasoning_efforts(provider, model)
    current = _unwrap(current_effort)
    if current in supported:
        return current

    if provider == "Claude":
        if _DEFAULT_CLAUDE_REASONING_EFFORT in supported:
            return _DEFAULT_CLAUDE_REASONING_EFFORT
        return supported[0] if supported else None

    if _CODEX_DEFAULT_REASONING_EFFORT in supported:
        return _CODEX_DEFAULT_REASONING_EFFORT

    default_effort = _GPT_MODEL_METADATA.get(_unwrap(model), {}).get("default_reasoning_level")
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


def _make_session(provider, session_id, model, reasoning_effort=None):
    model = _normalize_model(provider, model)
    session_cls = _get_session_cls(provider)
    session_kwargs = {"model": model, "cwd": str(REPO_ROOT)}
    normalized_effort = _normalize_reasoning_effort(provider, model, reasoning_effort)
    if normalized_effort:
        session_kwargs["reasoning_effort"] = normalized_effort

    if provider == "Claude":
        session_kwargs["max_turns"] = 6
        session_kwargs["tools"] = ["Read"]
    elif provider == "GPT":
        session_kwargs["tools"] = [str(REPO_ROOT)]

    if session_id:
        return session_cls.resume(session_id, **session_kwargs)
    return session_cls(**session_kwargs)


def _response_result_text(response) -> str:
    if isinstance(response, dict):
        result = response.get("result", "")
        return result if isinstance(result, str) else ""
    return ""


def _run_nonstream_turn(provider, message, session_id, model, reasoning_effort):
    session = _make_session(provider, session_id, model, reasoning_effort)
    response = session.prompt(message)
    next_session_id = getattr(session, "session_id", None) or session_id
    return _response_result_text(response), next_session_id


_GPT_MODEL_METADATA = _load_gpt_model_metadata()
_CODEX_DEFAULT_REASONING_EFFORT = _load_codex_reasoning_effort()
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


_INITIAL_STRUCTURED_ENTRIES, _INITIAL_WARNING = _initialize_structured_entries()
_set_structured_entries(_INITIAL_STRUCTURED_ENTRIES, _INITIAL_WARNING)


def _current_entry(batch_index: int) -> dict | None:
    if 1 <= batch_index <= TOTAL_BATCHES:
        return STRUCTURED_PROOF_ENTRIES[batch_index - 1]
    return None


def _load_audit_entries() -> list[dict]:
    try:
        audit_entries = load_postprocessing_audit()
    except RuntimeError:
        return []
    try:
        formal_qa_entries = load_formal_qa_data()
    except RuntimeError:
        return audit_entries

    reconciled = reconcile_audit_with_formal_qa(
        STRUCTURED_PROOF_ENTRIES,
        audit_entries,
        formal_qa_entries,
    )
    if reconciled != audit_entries:
        save_postprocessing_audit(reconciled)
    return reconciled


def _truncate(text: str, limit: int = 900) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n... truncated ..."


def _status(batch_index: int, audit_entries: list[dict], extra: str = "") -> str:
    approved = approved_batch_count(audit_entries)
    parts = [f"**{approved}** batches approved"]
    if TOTAL_BATCHES:
        if batch_index <= TOTAL_BATCHES:
            parts.append(f"Batch **{batch_index} / {TOTAL_BATCHES}**")
        else:
            parts.append(f"All **{TOTAL_BATCHES}** batches processed")
    if DATASET_WARNING:
        parts.append(DATASET_WARNING)
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def _review_controls(approve=False, retry=False, skip=False):
    return (
        gr.update(visible=approve),
        gr.update(visible=retry),
        gr.update(visible=skip),
    )


def _build_batch_context(entry: dict | None, batch_index: int) -> str:
    if entry is None:
        if TOTAL_BATCHES == 0:
            return "*No structured proofs are available.*"
        return f"*All {TOTAL_BATCHES} batches have been processed.*"

    qa_batch = entry.get("qa_batch", [])
    qa_lines = []
    for qa_index, qa_item in enumerate(qa_batch, start=1):
        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        qa_lines.append(f"### Q{qa_index}\n{question}")
        qa_lines.append(f"### A{qa_index}\n{answer}")

    prompt_text = entry.get("lean_prompt", "")
    return (
        f"## Batch {batch_index} / {TOTAL_BATCHES}\n\n"
        "### Lean Prompt\n"
        f"{prompt_text}\n\n"
        "### QA Batch\n"
        + "\n\n".join(qa_lines)
    )


def _build_boundary_text(attempt: dict | None) -> str:
    if not attempt:
        return "*No extraction decision yet.*"

    sections = []
    extraction_items = attempt.get("extraction_items") or attempt.get("boundary_items") or []
    if extraction_items:
        sections.append(json.dumps({"items": extraction_items}, indent=2, ensure_ascii=False))

    response_text = (attempt.get("llm_response") or "").strip()
    if response_text:
        sections.append("Raw model response:\n" + response_text)

    return "\n\n".join(sections) if sections else "*No extraction decision yet.*"


def _build_validation_text(attempt: dict | None) -> str:
    if not attempt:
        return "*No validation run yet.*"

    lines = [
        f"Status: {attempt.get('status', 'unknown')}",
        f"Batch index: {attempt.get('batch_index', '?')}",
    ]

    errors = attempt.get("validation_errors") or []
    if errors:
        lines.append("Validation errors:")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("Validation errors: none")

    extracted_items = attempt.get("extracted_items") or []
    if extracted_items:
        lines.append("")
        lines.append("Per-item checks:")
        for item in extracted_items:
            lines.append(
                f"- Q{item['qa_index']} -> {item['target_theorem_name']} | "
                f"lines {item['start_line']}-{item['end_line']} | "
                f"compile={'ok' if item['compile_ok'] else 'fail'}"
            )
            compile_summary = _truncate(item.get("compile_summary", ""))
            if compile_summary:
                lines.append(compile_summary)

    return "\n".join(lines)


def _build_compile_check_text(attempt: dict | None) -> str:
    if not attempt:
        return "*No compile check yet.*"

    extracted_items = attempt.get("extracted_items") or []
    if not extracted_items:
        return "*No compile check yet.*"

    all_compile_ok = all(item.get("compile_ok") for item in extracted_items)
    if all_compile_ok and len(extracted_items) == len(attempt.get("qa_batch") or []):
        return "✅ All `formal_answer` chunks in this batch compiled in Lean."

    failed_indices = [
        str(item.get("qa_index", "?"))
        for item in extracted_items
        if not item.get("compile_ok")
    ]
    if failed_indices:
        return "❌ Lean compilation failed for `formal_answer` item(s): " + ", ".join(failed_indices)

    return "❌ Lean compilation checks are incomplete for this batch."


def _attempt_has_compile_failures(attempt: dict | None) -> bool:
    if not isinstance(attempt, dict):
        return False
    extracted_items = attempt.get("extracted_items") or []
    return any(isinstance(item, dict) and not item.get("compile_ok") for item in extracted_items)


def _attempt_has_retryable_validation_errors(attempt: dict | None) -> bool:
    if not isinstance(attempt, dict):
        return False
    if attempt.get("status") == "validated":
        return False
    errors = attempt.get("validation_errors") or []
    return bool(errors)


def _build_flattened_text(attempt: dict | None) -> str:
    if not attempt:
        return "*No flattened output yet.*"
    flattened = attempt.get("flattened_records") or []
    if not flattened:
        return "*No flattened output yet.*"
    return json.dumps(flattened, indent=2, ensure_ascii=False)


def _placeholder_attempt(entry: dict, batch_index: int, status: str, error_message: str = "") -> dict:
    attempt = {
        "batch_index": batch_index,
        "status": status,
        "lean_prompt": entry.get("lean_prompt", ""),
        "qa_batch": entry.get("qa_batch", []),
        "formal_proofs": entry.get("formal_proofs", ""),
        "extraction_items": [],
        "boundary_items": [],
        "validation_errors": [error_message] if error_message else [],
        "flattened_records": [],
        "extracted_items": [],
        "llm_response": "",
    }
    return attempt


def _current_batch_panels(batch_index: int) -> tuple[str, str]:
    entry = _current_entry(batch_index)
    if entry is None:
        return _build_batch_context(None, batch_index), "*No active batch.*"
    formal_proofs = entry.get("formal_proofs", "")
    source_text = formal_proofs if isinstance(formal_proofs, str) else "*No formal proofs found.*"
    return _build_batch_context(entry, batch_index), source_text


def _save_attempt(audit_entries: list[dict], attempt: dict) -> list[dict]:
    next_audit = upsert_audit_entry(audit_entries, attempt)
    save_postprocessing_audit(next_audit)
    return next_audit


def _render_idle_state(batch_index: int, audit_entries: list[dict], extra: str = ""):
    batch_panel, source_panel = _current_batch_panels(batch_index)
    return (
        batch_panel,
        source_panel,
        "*No extraction decision yet.*",
        "*No validation run yet.*",
        "*No compile check yet.*",
        "*No flattened output yet.*",
        None,
        batch_index,
        audit_entries,
        None,
        *_review_controls(False, False, False),
        _status(batch_index, audit_entries, extra),
        gr.update(value=""),
    )


def run_split(provider, session_id, model, reasoning_effort,
              batch_index, audit_entries, attempt_state, instruction_text,
              max_auto_fix_attempts: int = 3):
    batch_panel, source_panel = _current_batch_panels(batch_index)

    if batch_index > TOTAL_BATCHES:
        yield (
            batch_panel,
            source_panel,
            "*No extraction decision yet.*",
            "*No validation run yet.*",
            "*No compile check yet.*",
            "*No flattened output yet.*",
            session_id,
            batch_index,
            audit_entries,
            None,
            *_review_controls(False, False, False),
            _status(batch_index, audit_entries, "COMPLETE"),
            gr.update(),
        )
        return

    entry = _current_entry(batch_index)
    if entry is None:
        yield (
            "*No structured proofs are available.*",
            "*No active batch.*",
            "*No extraction decision yet.*",
            "*No validation run yet.*",
            "*No compile check yet.*",
            "*No flattened output yet.*",
            session_id,
            batch_index,
            audit_entries,
            None,
            *_review_controls(False, False, False),
            _status(batch_index, audit_entries, "NO DATA"),
            gr.update(),
        )
        return

    yield (
        batch_panel,
        source_panel,
        "*Generating extracted formal answers...*",
        "*Validating extracted Lean code and compilation...*",
        "*Checking Lean compilation for extracted `formal_answer` chunks...*",
        "*No flattened output yet.*",
        session_id,
        batch_index,
        audit_entries,
        attempt_state,
        *_review_controls(False, False, False),
        _status(batch_index, audit_entries, "GENERATING"),
        gr.update(),
    )

    previous_attempt = None
    if isinstance(attempt_state, dict) and attempt_state.get("batch_index") == batch_index:
        previous_attempt = attempt_state

    current_session_id = session_id
    current_audit_entries = audit_entries

    for repair_attempt in range(1, max_auto_fix_attempts + 1):
        attempt_label = (
            "GENERATING"
            if repair_attempt == 1
            else f"AUTO-REPAIR {repair_attempt}/{max_auto_fix_attempts}"
        )
        yield (
            batch_panel,
            source_panel,
            "*Generating extracted formal answers...*",
            "*Validating extracted Lean code and compilation...*",
            (
                "*Checking Lean compilation for extracted `formal_answer` chunks...*"
                if repair_attempt == 1
                else "*Compile failures detected. Repairing extracted answers using Lean error output...*"
            ),
            "*No flattened output yet.*",
            current_session_id,
            batch_index,
            current_audit_entries,
            previous_attempt if repair_attempt > 1 else attempt_state,
            *_review_controls(False, False, False),
            _status(batch_index, current_audit_entries, attempt_label),
            gr.update(),
        )

        try:
            prompt = build_extraction_prompt(
                entry,
                batch_index=batch_index,
                extra_instruction=instruction_text,
                previous_attempt=previous_attempt,
            )
            response_text, current_session_id = _run_nonstream_turn(
                provider,
                prompt,
                current_session_id,
                model,
                reasoning_effort,
            )
            attempt = evaluate_extraction_response(entry, batch_index=batch_index, response_text=response_text)
        except Exception as exc:
            traceback.print_exc()
            error_attempt = _placeholder_attempt(entry, batch_index, "failed", str(exc))
            current_audit_entries = _save_attempt(current_audit_entries, error_attempt)
            yield (
                batch_panel,
                source_panel,
                "*No extraction decision yet.*",
                _build_validation_text(error_attempt),
                _build_compile_check_text(error_attempt),
                "*No flattened output yet.*",
                current_session_id,
                batch_index,
                current_audit_entries,
                error_attempt,
                *_review_controls(False, True, True),
                _status(batch_index, current_audit_entries, "FAILED"),
                gr.update(),
            )
            return

        current_audit_entries = _save_attempt(current_audit_entries, attempt)
        is_validated = attempt.get("status") == "validated"
        has_compile_failures = _attempt_has_compile_failures(attempt)
        has_retryable_errors = _attempt_has_retryable_validation_errors(attempt)

        if is_validated:
            yield (
                batch_panel,
                source_panel,
                _build_boundary_text(attempt),
                _build_validation_text(attempt),
                _build_compile_check_text(attempt),
                _build_flattened_text(attempt),
                current_session_id,
                batch_index,
                current_audit_entries,
                attempt,
                *_review_controls(False, False, False),
                _status(batch_index, current_audit_entries, "VALIDATED | AUTO-APPROVING"),
                gr.update(),
            )
            yield from approve_batch(
                provider,
                model,
                reasoning_effort,
                batch_index,
                current_audit_entries,
                attempt,
                instruction_text,
            )
            return

        if has_retryable_errors and repair_attempt < max_auto_fix_attempts:
            retry_reason = (
                "COMPILE FAILURES"
                if has_compile_failures
                else "STRUCTURAL VALIDATION FAILURES"
            )
            yield (
                batch_panel,
                source_panel,
                _build_boundary_text(attempt),
                _build_validation_text(attempt),
                _build_compile_check_text(attempt),
                _build_flattened_text(attempt),
                current_session_id,
                batch_index,
                current_audit_entries,
                attempt,
                *_review_controls(False, False, False),
                _status(
                    batch_index,
                    current_audit_entries,
                    f"{retry_reason} | RETRYING {repair_attempt + 1}/{max_auto_fix_attempts}",
                ),
                gr.update(),
            )
            previous_attempt = attempt
            continue

        yield (
            batch_panel,
            source_panel,
            _build_boundary_text(attempt),
            _build_validation_text(attempt),
            _build_compile_check_text(attempt),
            _build_flattened_text(attempt),
            current_session_id,
            batch_index,
            current_audit_entries,
            attempt,
            *_review_controls(False, True, True),
            _status(batch_index, current_audit_entries, "FAILED"),
            gr.update(),
        )
        return


def approve_batch(provider, model, reasoning_effort, batch_index, audit_entries, attempt_state, instruction_text):
    entry = _current_entry(batch_index)
    if entry is None:
        yield _render_idle_state(batch_index, audit_entries, "NO DATA")
        return

    if not isinstance(attempt_state, dict) or attempt_state.get("status") != "validated":
        batch_panel, source_panel = _current_batch_panels(batch_index)
        yield (
            batch_panel,
            source_panel,
            _build_boundary_text(attempt_state if isinstance(attempt_state, dict) else None),
            "Cannot approve because the current batch has not passed validation.",
            _build_compile_check_text(attempt_state if isinstance(attempt_state, dict) else None),
            _build_flattened_text(attempt_state if isinstance(attempt_state, dict) else None),
            None,
            batch_index,
            audit_entries,
            attempt_state,
            *_review_controls(False, True, True),
            _status(batch_index, audit_entries, "APPROVE BLOCKED"),
            gr.update(),
        )
        return

    approved_attempt = set_attempt_status(attempt_state, "approved")
    next_audit = _save_attempt(audit_entries, approved_attempt)
    write_formal_qa_data_from_audit(next_audit)
    upcoming_index = next_batch_index(next_audit, TOTAL_BATCHES)

    if upcoming_index > TOTAL_BATCHES:
        batch_panel, source_panel = _current_batch_panels(upcoming_index)
        yield (
            batch_panel,
            source_panel,
            "*No extraction decision yet.*",
            "*No validation run yet.*",
            "*No compile check yet.*",
            "*No flattened output yet.*",
            None,
            upcoming_index,
            next_audit,
            None,
            *_review_controls(False, False, False),
            _status(upcoming_index, next_audit, "APPROVED"),
            gr.update(value=""),
        )
        return

    next_batch_panel, next_source_panel = _current_batch_panels(upcoming_index)
    yield (
        next_batch_panel,
        next_source_panel,
            "*Preparing the next batch...*",
            "*Starting automatic split after approval...*",
        "*No compile check yet.*",
        "*No flattened output yet.*",
        None,
        upcoming_index,
        next_audit,
        None,
        *_review_controls(False, False, False),
        _status(upcoming_index, next_audit, "APPROVED | STARTING NEXT BATCH"),
        gr.update(),
    )

    yield from run_split(
        provider,
        None,
        model,
        reasoning_effort,
        upcoming_index,
        next_audit,
        None,
        instruction_text,
    )


def skip_batch(batch_index, audit_entries, attempt_state):
    entry = _current_entry(batch_index)
    if entry is None:
        return _render_idle_state(batch_index, audit_entries, "NO DATA")

    if isinstance(attempt_state, dict) and attempt_state.get("batch_index") == batch_index:
        skipped_attempt = set_attempt_status(attempt_state, "skipped")
        if not skipped_attempt.get("validation_errors"):
            skipped_attempt["validation_errors"] = ["Skipped by user."]
    else:
        skipped_attempt = _placeholder_attempt(entry, batch_index, "skipped", "Skipped by user.")

    next_audit = _save_attempt(audit_entries, skipped_attempt)
    upcoming_index = next_batch_index(next_audit, TOTAL_BATCHES)
    batch_panel, source_panel = _current_batch_panels(upcoming_index)
    return (
        batch_panel,
        source_panel,
        "*No extraction decision yet.*",
        "*No validation run yet.*",
        "*No compile check yet.*",
        "*No flattened output yet.*",
        None,
        upcoming_index,
        next_audit,
        None,
        *_review_controls(False, False, False),
        _status(upcoming_index, next_audit, "SKIPPED"),
        gr.update(value=""),
    )


def clear_session():
    _refresh_structured_entries()
    audit_entries = _load_audit_entries()
    batch_index = next_batch_index(audit_entries, TOTAL_BATCHES)
    return _render_idle_state(batch_index, audit_entries, "LOADED FROM DISK" if audit_entries else "")


def on_provider_change(provider, current_effort):
    model_choices = PROVIDER_MODELS[provider]
    model = model_choices[0]
    return (
        gr.update(choices=model_choices, value=model),
        update_reasoning_effort_dropdown(provider, model, current_effort),
        *clear_session(),
    )


_INITIAL_AUDIT = _load_audit_entries()
_INITIAL_BATCH_INDEX = next_batch_index(_INITIAL_AUDIT, TOTAL_BATCHES)
_INITIAL_BATCH_PANEL, _INITIAL_SOURCE_PANEL = _current_batch_panels(_INITIAL_BATCH_INDEX)
_INITIAL_STATUS = _status(
    _INITIAL_BATCH_INDEX,
    _INITIAL_AUDIT,
    "LOADED FROM DISK" if _INITIAL_AUDIT else "",
)


def render_postprocessing_ui():
    gr.Markdown(
        "# Proof Postprocessing\n"
        "Extract each batch-level `formal_proofs` blob into validated per-QA `formal_answer` records.  \n"
        "A batch can only be approved after the model-produced Lean snippets align with the QA targets and compile for every item."
    )

    session_state = gr.State(None)
    batch_index_state = gr.State(_INITIAL_BATCH_INDEX)
    audit_state = gr.State(_INITIAL_AUDIT)
    attempt_state = gr.State(None)

    with gr.Row(elem_classes=["settings-row"]):
        default_provider = "GPT"
        default_model = "gpt-5.4" if "gpt-5.4" in PROVIDER_MODELS[default_provider] else PROVIDER_MODELS[default_provider][0]
        provider_dropdown = gr.Dropdown(
            choices=list(PROVIDER_MODELS.keys()),
            value=default_provider,
            label="Provider",
            scale=1,
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

    instruction_box = gr.Textbox(
        label="Extra instruction",
        placeholder="Optional guidance for the splitter, e.g. `be strict about not crossing into C4`...",
        lines=4,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("BATCH CONTEXT", elem_classes=["panel-label"])
            batch_panel = gr.Markdown(
                _INITIAL_BATCH_PANEL,
                elem_classes=["scroll-panel"],
                max_height="65vh",
                min_height="200px",
            )
        with gr.Column(scale=1):
            gr.Markdown("FORMAL PROOFS", elem_classes=["panel-label"])
            source_panel = gr.Textbox(
                value=_INITIAL_SOURCE_PANEL,
                lines=20,
                max_lines=36,
                interactive=False,
                autoscroll=False,
            )
        with gr.Column(scale=1):
            gr.Markdown("BOUNDARY DECISION", elem_classes=["panel-label"])
            boundary_panel = gr.Textbox(
                value="*No extraction decision yet.*",
                lines=16,
                max_lines=26,
                interactive=False,
                autoscroll=False,
            )
            gr.Markdown("VALIDATION", elem_classes=["panel-label"])
            validation_panel = gr.Textbox(
                value="*No validation run yet.*",
                lines=12,
                max_lines=20,
                interactive=False,
                autoscroll=False,
            )
            gr.Markdown("LEAN COMPILE CHECK", elem_classes=["panel-label"])
            compile_check_panel = gr.Markdown(
                "*No compile check yet.*",
                elem_classes=["scroll-panel"],
            )

    gr.Markdown("FLATTENED OUTPUT PREVIEW", elem_classes=["panel-label"])
    flattened_panel = gr.Textbox(
        value="*No flattened output yet.*",
        lines=12,
        max_lines=22,
        interactive=False,
        autoscroll=False,
    )

    with gr.Row(elem_classes=["review-row"]):
        approve_btn = gr.Button("Approve Batch", variant="primary", visible=False, min_width=160)
        retry_btn = gr.Button("Retry Split", variant="secondary", visible=False, min_width=160)
        skip_btn = gr.Button("Skip Batch", variant="stop", visible=False, min_width=160)

    status_md = gr.Markdown(_INITIAL_STATUS, elem_classes=["status-bar"])

    with gr.Row():
        generate_btn = gr.Button("Generate Split", variant="primary")
        clear_btn = gr.Button("New Session")

    panel_outputs = [
        batch_panel, source_panel, boundary_panel, validation_panel, compile_check_panel, flattened_panel,
        session_state, batch_index_state, audit_state, attempt_state,
        approve_btn, retry_btn, skip_btn, status_md, instruction_box,
    ]

    provider_dropdown.change(
        on_provider_change,
        inputs=[provider_dropdown, reasoning_effort_dropdown],
        outputs=[model_dropdown, reasoning_effort_dropdown, *panel_outputs],
    )

    model_dropdown.change(
        on_model_change,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown],
        outputs=[reasoning_effort_dropdown],
    )

    generate_btn.click(
        run_split,
        inputs=[provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                batch_index_state, audit_state, attempt_state, instruction_box],
        outputs=panel_outputs,
    )

    retry_btn.click(
        run_split,
        inputs=[provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                batch_index_state, audit_state, attempt_state, instruction_box],
        outputs=panel_outputs,
    )

    approve_btn.click(
        approve_batch,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown,
                batch_index_state, audit_state, attempt_state, instruction_box],
        outputs=panel_outputs,
    )

    skip_btn.click(
        skip_batch,
        inputs=[batch_index_state, audit_state, attempt_state],
        outputs=panel_outputs,
    )

    clear_btn.click(clear_session, outputs=panel_outputs)


def create_postprocessing_demo():
    with gr.Blocks(title="Proof Postprocessing", css=CSS, theme=gr.themes.Soft()) as demo:
        render_postprocessing_ui()
    return demo


def _switch_workspace_view(target: str):
    return (
        gr.update(visible=(target == "qa")),
        gr.update(visible=(target == "lean")),
        gr.update(visible=(target == "postprocessing")),
    )


def create_workbench_demo(initial_view: str = "postprocessing"):
    import app as app_module
    import lean_app as lean_module

    combined_css = "\n".join([app_module.CSS, lean_module.CSS, CSS])
    with gr.Blocks(title="Formal Science Workbench", css=combined_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["review-row"]):
            qa_nav_btn = gr.Button("QA Dataset Builder", variant="secondary", min_width=180)
            lean_nav_btn = gr.Button("Lean Code Generator", variant="secondary", min_width=180)
            post_nav_btn = gr.Button("Postprocessing", variant="primary", min_width=180)

        with gr.Column(visible=(initial_view == "qa")) as qa_view:
            app_module.render_qa_builder_ui()

        with gr.Column(visible=(initial_view == "lean")) as lean_view:
            lean_module.render_lean_builder_ui()

        with gr.Column(visible=(initial_view == "postprocessing")) as post_view:
            render_postprocessing_ui()

        qa_nav_btn.click(lambda: _switch_workspace_view("qa"), outputs=[qa_view, lean_view, post_view])
        lean_nav_btn.click(lambda: _switch_workspace_view("lean"), outputs=[qa_view, lean_view, post_view])
        post_nav_btn.click(
            lambda: _switch_workspace_view("postprocessing"),
            outputs=[qa_view, lean_view, post_view],
        )

    return demo


if __name__ == "__main__":
    create_workbench_demo(initial_view="postprocessing").launch()
