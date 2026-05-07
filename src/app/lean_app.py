"""
Gradio Lean Code Generator powered by Claude or GPT CLI.

Feeds prompts from ``lean_prompt_data.json`` one at a time to an LLM and
displays the generated Lean code.  Also supports custom prompts.

Run: python src/app/lean_app.py
"""

import sys
import difflib
import json
import queue
import re
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "llm"))
sys.path.insert(0, str(_SRC / "app"))

from claude_cli import ClaudeSession, CLAUDE_REASONING_EFFORTS
from compile_lean import compile_lean
from gpt_cli import GPTSession, VALID_REASONING_EFFORTS
from lean_prompts import build_lean_prompt_dataset_from_file


DEFAULT_PROMPT_DATA_PATH = _SRC / "app_data" / "lean_prompt_data.json"
DEFAULT_QA_DATA_PATH = _SRC / "app_data" / "qa_data.json"
DEFAULT_OUTPUT_PATH = _SRC / "app_data" / "lean_output_data.json"
DEFAULT_STRUCTURED_OUTPUT_PATH = _SRC / "app_data" / "structured_proofs.json"
REPO_ROOT = _SRC.parent
DEFAULT_PROOF_PATH = REPO_ROOT / "FSLean" / "proof.lean"
DEFAULT_DATASET_SETUP_MESSAGE = (
    "Default pipeline data is missing. Populate `src/app_data/qa_data.json` first, or "
    "run the QA builder (`python src/app/app.py`) to generate both `qa_data.json` and "
    "`lean_prompt_data.json`. Custom mode is still available."
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required data file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _load_prompt_qa_pairs(
    prompt_path: Path = DEFAULT_PROMPT_DATA_PATH,
    qa_path: Path = DEFAULT_QA_DATA_PATH,
) -> list[dict]:
    qa_batches = _read_json(qa_path)
    try:
        prompts = _read_json(prompt_path)
    except RuntimeError as exc:
        if prompt_path.exists():
            raise
        try:
            prompts = build_lean_prompt_dataset_from_file(qa_path)
        except Exception as build_exc:
            raise RuntimeError(
                f"Failed to build {prompt_path.name} from {qa_path}: {build_exc}"
            ) from build_exc
        try:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(
                json.dumps(prompts, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as write_exc:
            raise RuntimeError(f"Failed to write {prompt_path}: {write_exc}") from write_exc

    if not isinstance(prompts, list):
        raise RuntimeError(f"{prompt_path} must contain a top-level JSON list.")
    if not isinstance(qa_batches, list):
        raise RuntimeError(f"{qa_path} must contain a top-level JSON list.")
    if len(prompts) != len(qa_batches):
        raise RuntimeError(
            "Prompt/QA batch count mismatch: "
            f"{prompt_path.name} has {len(prompts)} items but {qa_path.name} has {len(qa_batches)}."
        )

    paired_data = []
    for prompt_idx, (prompt, qa_batch) in enumerate(zip(prompts, qa_batches), start=1):
        if not isinstance(prompt, str):
            raise RuntimeError(f"{prompt_path.name}[{prompt_idx - 1}] must be a string prompt.")
        if not isinstance(qa_batch, list):
            raise RuntimeError(f"{qa_path.name}[{prompt_idx - 1}] must be a list of question/answer items.")

        normalized_batch = []
        for qa_idx, qa_item in enumerate(qa_batch, start=1):
            if not isinstance(qa_item, dict):
                raise RuntimeError(
                    f"{qa_path.name}[{prompt_idx - 1}][{qa_idx - 1}] must be an object "
                    "with `question` and `answer`."
                )

            question = qa_item.get("question")
            answer = qa_item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise RuntimeError(
                    f"{qa_path.name}[{prompt_idx - 1}][{qa_idx - 1}] must have string "
                    "`question` and `answer` values."
                )

            normalized_batch.append({
                "question": question,
                "answer": answer,
            })

        paired_data.append({
            "lean_prompt": prompt,
            "qa_batch": normalized_batch,
        })

    return paired_data


def _initialize_prompt_data() -> tuple[list[dict], str | None]:
    try:
        return _load_prompt_qa_pairs(), None
    except RuntimeError as exc:
        if not DEFAULT_QA_DATA_PATH.exists():
            return [], DEFAULT_DATASET_SETUP_MESSAGE
        raise


def _set_prompt_data(prompt_qa_pairs: list[dict], warning: str | None) -> None:
    global PROMPT_QA_PAIRS, PROMPTS, TOTAL_PROMPTS, DATASET_WARNING
    PROMPT_QA_PAIRS = prompt_qa_pairs
    PROMPTS = [entry["lean_prompt"] for entry in prompt_qa_pairs]
    TOTAL_PROMPTS = len(PROMPTS)
    DATASET_WARNING = warning


def _refresh_prompt_data() -> None:
    prompt_qa_pairs, warning = _initialize_prompt_data()
    _set_prompt_data(prompt_qa_pairs, warning)


def _load_gpt_models() -> list[str]:
    fallback = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2-codex", "gpt-5.2"]
    return list(_GPT_MODEL_METADATA) or fallback


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


def _load_codex_reasoning_effort():
    config_path = Path.home() / ".codex" / "config.toml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r'^\s*model_reasoning_effort\s*=\s*"([^"]+)"', text, re.MULTILINE)
    effort = match.group(1) if match else None
    return effort if effort in VALID_REASONING_EFFORTS else None


REASONING_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh")
_GPT_MODEL_METADATA = _load_gpt_model_metadata()
_CODEX_DEFAULT_REASONING_EFFORT = _load_codex_reasoning_effort()
_DEFAULT_CLAUDE_REASONING_EFFORT = "medium"


def _load_saved_outputs() -> list[dict]:
    try:
        data = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data


def _autosave_outputs(outputs: list[dict]):
    try:
        DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUTPUT_PATH.write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_OUTPUT_PATH}: {exc}") from exc


def _load_structured_proofs() -> list[dict]:
    try:
        data = json.loads(DEFAULT_STRUCTURED_OUTPUT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_STRUCTURED_OUTPUT_PATH}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"{DEFAULT_STRUCTURED_OUTPUT_PATH} must contain a top-level JSON list.")
    return data


def _autosave_structured_proofs(entries: list[dict]):
    try:
        DEFAULT_STRUCTURED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_STRUCTURED_OUTPUT_PATH.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_STRUCTURED_OUTPUT_PATH}: {exc}") from exc


def _build_structured_proof_entry(prompt_index: int, current_prompt: str, proof_code: str) -> dict:
    if prompt_index < 0 or prompt_index >= len(PROMPT_QA_PAIRS):
        raise RuntimeError(
            f"Prompt index {prompt_index} is out of range for the validated prompt/QA mapping."
        )

    aligned_entry = PROMPT_QA_PAIRS[prompt_index]
    aligned_prompt = aligned_entry["lean_prompt"]
    if current_prompt != aligned_prompt:
        raise RuntimeError(
            "Refusing to save structured proof because the current prompt text does not match "
            "the validated prompt/QA mapping for this index."
        )

    return {
        "lean_prompt": aligned_prompt,
        "qa_batch": aligned_entry["qa_batch"],
        "formal_proofs": proof_code,
    }


_INITIAL_PROMPT_QA_PAIRS, _INITIAL_DATASET_WARNING = _initialize_prompt_data()
_set_prompt_data(_INITIAL_PROMPT_QA_PAIRS, _INITIAL_DATASET_WARNING)

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

CODE_ONLY_SUFFIX = """

# Task
Write the complete Lean 4 source code directly to `FSLean/proof.lean`.
Do not write to any other file.
Do not output the code as part of your response.
After writing `FSLean/proof.lean`, stop immediately.
Do not read the file back.
Do not provide any explanation beyond a brief confirmation that the file was written.
""".strip()

COMPILE_FIX_SUFFIX = """
The Lean file did not compile.

Fix `FSLean/proof.lean` in place to resolve the compiler errors below.
Do not write to any other file.
After updating `FSLean/proof.lean`, stop immediately.
Do not read the file back.
Do not provide any explanation beyond a brief confirmation that the file was updated.
""".strip()

ALIGNMENT_PROMPT = (
    "Using a 5-point Likert scale to 1dp, determine how well each Lean code proof Ci successfully "
    "proves the target results from Qi and Ai, and aligns with the Requirements. Use any score in 0.1 "
    "increments from 1.0 to 5.0 when needed; do not default to whole-number scores. "
    "If a patch difference is provided, begin with a brief **Patch Difference** note summarizing "
    "what changed and whether the change is substantive; explicitly say if there was no meaningful "
    "change."
)
ALIGNMENT_WAITING = "*Alignment evaluation will appear here after successful compilation…*"
ALIGNMENT_RUNNING = "*Evaluating alignment…*"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(prompt_index, saved_count, mode, extra=""):
    parts = [f"**{saved_count}** outputs saved"]
    if mode == "Default Pipeline" and TOTAL_PROMPTS:
        idx = min(prompt_index + 1, TOTAL_PROMPTS)
        parts.append(f"Prompt **{idx} / {TOTAL_PROMPTS}**")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


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
    """
    Creates and returns an LLM session configured with agentic capabilities.
    """
    model = _normalize_model(provider, model)
    session_cls = _get_session_cls(provider)
    session_kwargs = {"model": model, "cwd": str(REPO_ROOT)}
    normalized_effort = _normalize_reasoning_effort(provider, model, reasoning_effort)
    if normalized_effort:
        session_kwargs["reasoning_effort"] = normalized_effort

    if provider == "Claude":
        # Give Claude multiple turns so it can run tools, read outputs, and finalize.
        session_kwargs["max_turns"] = 10
        # Grant explicit permission to use file system tools to prevent the CLI
        # from hanging while waiting for a terminal [Y/n] confirmation.
        session_kwargs["tools"] = ["Bash", "Edit", "Read", "Write", "Replace"]
    elif provider == "GPT":
        # GPT/Codex uses native sandbox configurations set in gpt_cli.
        session_kwargs["tools"] = [str(DEFAULT_PROOF_PATH.parent)]

    if session_id:
        return session_cls.resume(session_id, **session_kwargs)
    return session_cls(**session_kwargs)


def _stream(provider, message, session_id, model, reasoning_effort):
    session = _make_session(provider, session_id, model, reasoning_effort)
    full = ""
    try:
        for chunk in session.prompt_stream(message):
            full += chunk
            yield full, session.session_id or session_id
    except Exception as e:
        traceback.print_exc()
        yield f"**Error:** {e}", session_id


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


def _heartbeat_message(elapsed: float) -> str:
    seconds = int(elapsed)
    return f"*Generating... still working ({seconds}s elapsed).*"


def _review_controls(approve=False, regenerate=False, skip=False):
    return (
        gr.update(visible=approve),
        gr.update(visible=regenerate),
        gr.update(visible=skip),
    )


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


def _build_model_message(prompt_text: str) -> str:
    return f"{prompt_text.rstrip()}\n\n{CODE_ONLY_SUFFIX}\n"


def _read_proof_source() -> str:
    try:
        return DEFAULT_PROOF_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_PROOF_PATH}: {exc}") from exc


def _write_proof_source(code: str) -> str:
    try:
        DEFAULT_PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PROOF_PATH.write_text(code, encoding="utf-8")
        return _read_proof_source()
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_PROOF_PATH}: {exc}") from exc


def _delete_proof_source():
    try:
        DEFAULT_PROOF_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_PROOF_PATH}: {exc}") from exc


def _format_compile_output(returncode: int, stdout: str, stderr: str, iteration: int) -> str:
    sections = [f"Iteration: {iteration}", f"Exit code: {returncode}"]
    stdout = stdout.strip()
    stderr = stderr.strip()

    if returncode == 0:
        sections.append("Compilation succeeded.")
        if stdout:
            sections.append(f"STDOUT:\n{stdout}")
        if stderr:
            sections.append(f"STDERR:\n{stderr}")
        return "\n\n".join(sections)

    current_error = stderr or stdout or "No compiler output."
    sections.append(f"Current error:\n{current_error}")
    if stdout and stdout != current_error:
        sections.append(f"STDOUT:\n{stdout}")
    if stderr and stderr != current_error:
        sections.append(f"STDERR:\n{stderr}")
    return "\n\n".join(sections)


def _build_compile_fix_message(code: str, compiler_output: str, iteration: int) -> str:
    return (
        f"{COMPILE_FIX_SUFFIX}\n\n"
        f"Compile attempt: {iteration}\n\n"
        f"Compiler output:\n{compiler_output}\n\n"
        "Current Lean code:\n"
        "```lean\n"
        f"{code.rstrip()}\n"
        "```"
    )


def _build_alignment_message(
    prompt_text: str,
    proof_code: str,
    previous_proof_code: str | None = None,
) -> str:
    sections = [
        "Initial prompt:\n" + (prompt_text or "").strip(),
        "Current proof.lean code:\n```lean\n" + f"{proof_code.rstrip()}\n```",
    ]

    patch_diff = _build_brief_patch_diff(
        previous_proof_code,
        proof_code,
        from_label="previous_proof.lean",
        to_label="current_proof.lean",
    )
    if patch_diff:
        sections.append("Patch difference from previous evaluated proof:\n```diff\n" + f"{patch_diff}\n```")

    sections.append(ALIGNMENT_PROMPT)
    return "\n\n".join(sections)


def _build_refinement_message(
    user_instruction: str,
    prompt_text: str,
    proof_code: str,
    alignment_feedback: str | None,
) -> str:
    sections = [
        "Revise `FSLean/proof.lean` so it better satisfies the initial prompt and addresses "
        "the latest alignment feedback.\n"
        "Update the file directly. Do not write to any other file.\n"
        "After updating `FSLean/proof.lean`, stop immediately.\n"
        "Do not read the file back.\n"
        "Do not provide any explanation beyond a brief confirmation that the file was updated.",
        "Initial prompt:\n" + (prompt_text or "").strip(),
    ]

    normalized_feedback = _normalize_review_feedback(
        alignment_feedback,
        ALIGNMENT_WAITING,
        ALIGNMENT_RUNNING,
    )
    if normalized_feedback:
        sections.append("Latest alignment feedback:\n" + normalized_feedback)

    sections.extend([
        "User request:\n" + user_instruction.strip(),
        "Current proof.lean code:\n```lean\n" + f"{proof_code.rstrip()}\n```",
    ])
    return "\n\n".join(sections)


def capture_msg(message):
    return "", message


# ---------------------------------------------------------------------------
# Outputs tuple order (13 elements):
#   prompt_panel, output_panel, compile_panel, alignment_panel,
#   session_state, prompt_index_state, outputs_state, current_prompt_state,
#   approve_btn, regenerate_btn, skip_btn, status_md, custom_prompt_box
# ---------------------------------------------------------------------------

def _compile_and_align(prompt_display, provider, session_id, model, reasoning_effort,
                       prompt_index, saved_outputs, mode, current_prompt,
                       previous_evaluated_code: str | None = None,
                       max_compile_fix_attempts: int = 25):
    try:
        current_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            "*Waiting...*",
            str(exc),
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "COMPILE FAILED"),
            gr.update(),
        )
        return

    if not current_code.strip():
        yield (
            prompt_display,
            "*Waiting...*",
            "Nothing saved in `FSLean/proof.lean` yet.",
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "NO SAVED PROOF"),
            gr.update(),
        )
        return

    current_session_id = session_id
    last_compile_text = ""

    for iteration in range(1, max_compile_fix_attempts + 1):
        yield (
            prompt_display,
            current_code,
            f"Iteration: {iteration}\n\nCompiling current Lean output...",
            ALIGNMENT_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), mode, f"COMPILE ITERATION {iteration}"),
            gr.update(),
        )

        try:
            returncode, stdout, stderr = compile_lean(current_code)
        except Exception as exc:
            traceback.print_exc()
            yield (
                prompt_display,
                current_code,
                f"Compilation runner failed:\n{exc}",
                ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), mode, "COMPILATION RUNNER FAILED"),
                gr.update(),
            )
            return

        compile_text = _format_compile_output(returncode, stdout, stderr, iteration)
        last_compile_text = compile_text

        if returncode == 0:
            try:
                aligned_code = _read_proof_source()
            except RuntimeError as exc:
                yield (
                    prompt_display,
                    current_code,
                    f"{compile_text}\n\nFailed to reload `FSLean/proof.lean`:\n{exc}",
                    ALIGNMENT_WAITING,
                    current_session_id, prompt_index, saved_outputs, current_prompt,
                    *_review_controls(False, True, True),
                    _status(prompt_index, len(saved_outputs), mode, "ALIGNMENT BLOCKED"),
                    gr.update(),
                )
                return

            yield (
                prompt_display,
                aligned_code,
                compile_text,
                ALIGNMENT_RUNNING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, False, False),
                _status(prompt_index, len(saved_outputs), mode, f"COMPILED IN {iteration} ITERATION(S) | ALIGNMENT"),
                gr.update(),
            )

            alignment_message = _build_alignment_message(
                current_prompt,
                aligned_code,
                previous_evaluated_code,
            )
            try:
                alignment_text, current_session_id = _run_nonstream_turn(
                    provider, alignment_message, current_session_id, model, reasoning_effort
                )
            except Exception as exc:
                traceback.print_exc()
                yield (
                    prompt_display,
                    aligned_code,
                    compile_text,
                    f"**Error:** {exc}",
                    current_session_id, prompt_index, saved_outputs, current_prompt,
                    *_review_controls(False, True, True),
                    _status(prompt_index, len(saved_outputs), mode, "ALIGNMENT FAILED"),
                    gr.update(),
                )
                return

            yield (
                prompt_display,
                aligned_code,
                compile_text,
                alignment_text or ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(True, True, True),
                _status(prompt_index, len(saved_outputs), mode, f"COMPILED IN {iteration} ITERATION(S) | ALIGNMENT READY"),
                gr.update(),
            )
            return

        if returncode != 1:
            yield (
                prompt_display,
                current_code,
                compile_text,
                ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), mode, f"UNEXPECTED EXIT CODE {returncode}"),
                gr.update(),
            )
            return

        yield (
            prompt_display,
            current_code,
            f"{compile_text}\n\nRequesting an LLM revision based on the current compiler error...",
            ALIGNMENT_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), mode, f"REQUESTING FIX {iteration}"),
            gr.update(),
        )

        fix_message = _build_compile_fix_message(current_code, compile_text, iteration)

        try:
            _fix_result_text, current_session_id = _run_nonstream_turn(
                provider, fix_message, current_session_id, model, reasoning_effort
            )
        except Exception as exc:
            traceback.print_exc()
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nLLM revision request failed:\n{exc}",
                ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), mode, "LLM FIX FAILED"),
                gr.update(),
            )
            return

        # In agent mode the LLM updated proof.lean directly; read it back.
        try:
            current_code = _read_proof_source()
        except RuntimeError as exc:
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nFailed to read `FSLean/proof.lean` after LLM fix:\n{exc}",
                ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), mode, "FAILED TO READ PROOF"),
                gr.update(),
            )
            return

        if not current_code.strip():
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nAgent did not update `FSLean/proof.lean`.",
                ALIGNMENT_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), mode, "AGENT DID NOT WRITE PROOF"),
                gr.update(),
            )
            return

        yield (
            prompt_display,
            current_code,
            f"{compile_text}\n\nLLM revision applied. Recompiling...",
            ALIGNMENT_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), mode, f"RECOMPILING AFTER FIX {iteration}"),
            gr.update(),
        )

    yield (
        prompt_display,
        current_code,
        (
            f"{last_compile_text}\n\n"
            f"Reached the compile-fix safeguard after {max_compile_fix_attempts} iterations.\n\n"
            "The latest generated Lean code is still shown in the output panel."
        ),
        ALIGNMENT_WAITING,
        current_session_id, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, True, True),
        _status(prompt_index, len(saved_outputs), mode, "COMPILE FIX LIMIT REACHED"),
        gr.update(),
    )

def send_prompt(provider, session_id, model, reasoning_effort,
                prompt_index, saved_outputs, mode, custom_prompt_text):
    """Send the current prompt (default or custom), stop after proof.lean is written, then compile."""
    if mode == "Default Pipeline":
        _refresh_prompt_data()
        if TOTAL_PROMPTS == 0:
            setup_message = DATASET_WARNING or DEFAULT_DATASET_SETUP_MESSAGE
            yield (
                setup_message,
                "*Waiting...*",
                "*No compilation run yet.*",
                ALIGNMENT_WAITING,
                session_id, prompt_index, saved_outputs, "",
                *_review_controls(False, False, False),
                _status(prompt_index, len(saved_outputs), mode, "DEFAULT DATA UNAVAILABLE"),
                gr.update(),
            )
            return
        if prompt_index >= TOTAL_PROMPTS:
            done_msg = (
                f"All **{TOTAL_PROMPTS}** prompts processed.  \n"
                f"**{len(saved_outputs)}** outputs saved.  \n"
                "Results autosaved to `lean_output_data.json`."
            )
            yield ("*Pipeline complete.*", done_msg, "*No compilation run yet.*", ALIGNMENT_WAITING,
                   session_id, prompt_index, saved_outputs, "",
                   *_review_controls(False, False, False),
                   _status(prompt_index, len(saved_outputs), mode, "DONE"),
                   gr.update())
            return
        prompt_text = PROMPTS[prompt_index]
        header = f"### Prompt {prompt_index + 1} / {TOTAL_PROMPTS}\n\n---\n\n"
        prompt_display = header + prompt_text
    else:
        if not custom_prompt_text or not custom_prompt_text.strip():
            yield ("*Enter a custom prompt below and click Generate.*", "*Waiting...*", "*No compilation run yet.*", ALIGNMENT_WAITING,
                   session_id, prompt_index, saved_outputs, "",
                   *_review_controls(False, False, False),
                   _status(prompt_index, len(saved_outputs), mode),
                   gr.update())
            return
        prompt_text = custom_prompt_text.strip()
        prompt_display = f"### Custom Prompt\n\n---\n\n{prompt_text}"

    try:
        _delete_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            "*Waiting...*",
            str(exc),
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "FAILED TO RESET PROOF"),
            gr.update(),
        )
        return

    message = _build_model_message(prompt_text)
    st = _status(prompt_index, len(saved_outputs), mode, "WRITING PROOF")

    yield (
        prompt_display, "*Writing `FSLean/proof.lean`...*", "*No compilation run yet.*", ALIGNMENT_WAITING,
        session_id, prompt_index, saved_outputs, prompt_text,
        *_review_controls(False, False, False), st, gr.update()
    )

    try:
        agent_result, last_sid = _run_nonstream_turn(
            provider, message, session_id, model, reasoning_effort
        )
    except Exception as exc:
        traceback.print_exc()
        yield (
            prompt_display,
            "*Waiting...*",
            f"Initial write request failed:\n{exc}",
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "INITIAL WRITE FAILED"),
            gr.update(),
        )
        return

    try:
        persisted_lean_text = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            agent_result or "*Waiting...*",
            str(exc),
            ALIGNMENT_WAITING,
            last_sid, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "FAILED TO READ PROOF"),
            gr.update(),
        )
        return

    if not persisted_lean_text.strip():
        yield (
            prompt_display,
            agent_result or "*Waiting...*",
            "Agent did not write to `FSLean/proof.lean`.",
            ALIGNMENT_WAITING,
            last_sid, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "AGENT DID NOT WRITE PROOF"),
            gr.update(),
        )
        return

    yield from _compile_and_align(
        prompt_display, provider, last_sid, model, reasoning_effort,
        prompt_index, saved_outputs, mode, prompt_text,
    )


def on_approve(output_panel, provider, session_id, model, reasoning_effort,
               prompt_index, saved_outputs, current_prompt, mode, custom_prompt_text):
    """Save the current output and advance to the next prompt."""
    try:
        proof_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            str(exc),
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "APPROVE FAILED"),
            gr.update(),
        )
        return

    if not proof_code.strip():
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            "Cannot approve because `FSLean/proof.lean` is empty or missing.",
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "APPROVE BLOCKED"),
            gr.update(),
        )
        return

    entry = {"prompt": current_prompt or "", "output": proof_code}
    next_saved_outputs = saved_outputs + [entry]

    try:
        if mode == "Default Pipeline":
            structured_entry = _build_structured_proof_entry(
                prompt_index,
                current_prompt or "",
                proof_code,
            )
            structured_proofs = _load_structured_proofs()
            next_structured_proofs = structured_proofs + [structured_entry]
            _autosave_structured_proofs(next_structured_proofs)

        _autosave_outputs(next_saved_outputs)
    except RuntimeError as exc:
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            str(exc),
            ALIGNMENT_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "APPROVE FAILED"),
            gr.update(),
        )
        return

    saved_outputs = next_saved_outputs

    if mode == "Default Pipeline":
        prompt_index += 1
        yield from send_prompt(
            provider, None, model, reasoning_effort,
            prompt_index, saved_outputs, mode, custom_prompt_text,
        )
    else:
        yield ("*Enter a custom prompt below and click Generate.*", "*Waiting...*", "*No compilation run yet.*", ALIGNMENT_WAITING,
               None, prompt_index, saved_outputs, "",
               *_review_controls(False, False, False),
               _status(prompt_index, len(saved_outputs), mode, "SAVED"),
               gr.update(value=""))


def on_regenerate(prompt_panel, provider, session_id, model, reasoning_effort,
                  prompt_index, saved_outputs, current_prompt, mode, custom_prompt_text):
    """Regenerate the current prompt output."""
    yield from send_prompt(
        provider, None, model, reasoning_effort,
        prompt_index, saved_outputs, mode, custom_prompt_text,
    )


def on_skip(provider, session_id, model, reasoning_effort,
            prompt_index, saved_outputs, mode, custom_prompt_text):
    """Skip this prompt without saving and advance."""
    if mode == "Default Pipeline":
        prompt_index += 1
    yield from send_prompt(
        provider, None, model, reasoning_effort,
        prompt_index, saved_outputs, mode, custom_prompt_text,
    )


def user_refine_submit(message, prompt_panel, output_panel, compile_panel, alignment_panel,
                       provider, session_id, model, reasoning_effort,
                       prompt_index, saved_outputs, current_prompt, mode):
    if not message or not message.strip():
        yield (
            prompt_panel, output_panel, compile_panel, alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(True, True, True),
            _status(prompt_index, len(saved_outputs), mode),
            gr.update(),
        )
        return

    try:
        proof_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_panel, output_panel, str(exc), alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "REFINEMENT FAILED"),
            gr.update(),
        )
        return

    if not proof_code.strip():
        yield (
            prompt_panel, output_panel, "Nothing saved in `FSLean/proof.lean` yet.", alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "NO SAVED PROOF"),
            gr.update(),
        )
        return

    refinement_message = _build_refinement_message(
        message,
        current_prompt,
        proof_code,
        alignment_panel,
    )
    st = _status(prompt_index, len(saved_outputs), mode, "REFINING")

    yield (
        prompt_panel, proof_code, "Updating `proof.lean` in the current Claude session...", alignment_panel,
        session_id, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, False, False), st, gr.update(),
    )

    try:
        refine_result, last_sid = _run_nonstream_turn(
            provider, refinement_message, session_id, model, reasoning_effort
        )
    except Exception as exc:
        traceback.print_exc()
        yield (
            prompt_panel, output_panel, f"Refinement failed:\n{exc}", alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "REFINEMENT FAILED"),
            gr.update(),
        )
        return

    try:
        persisted_lean_text = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_panel, output_panel, str(exc), alignment_panel,
            last_sid, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "FAILED TO READ PROOF"),
            gr.update(),
        )
        return

    if not persisted_lean_text.strip():
        yield (
            prompt_panel, refine_result or output_panel, "Agent did not update `FSLean/proof.lean`.", alignment_panel,
            last_sid, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), mode, "AGENT DID NOT WRITE PROOF"),
            gr.update(),
        )
        return

    yield from _compile_and_align(
        prompt_panel, provider, last_sid, model, reasoning_effort,
        prompt_index, saved_outputs, mode, current_prompt,
        previous_evaluated_code=proof_code,
    )


def clear_session():
    _refresh_prompt_data()
    saved = _load_saved_outputs()
    start_index = len(_load_structured_proofs())
    status_extra = []
    if saved:
        status_extra.append("LOADED FROM DISK")
    if DATASET_WARNING:
        status_extra.append("DEFAULT DATA UNAVAILABLE")
    return (
        "*Waiting to start...*", "*Waiting...*", "*No compilation run yet.*", ALIGNMENT_WAITING,
        None, start_index, saved, "",
        *_review_controls(False, False, False),
        _status(start_index, len(saved), "Default Pipeline", " | ".join(status_extra)),
        gr.update(value=""),
        gr.update(value=""),
    )

def on_provider_change(provider, current_effort):
    model_choices = PROVIDER_MODELS[provider]
    model = model_choices[0]
    return (
        gr.update(choices=model_choices, value=model),
        update_reasoning_effort_dropdown(provider, model, current_effort),
        *clear_session(),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_INITIAL_SAVED = _load_saved_outputs()
_INITIAL_PROMPT_INDEX = len(_load_structured_proofs())
_INITIAL_STATUS = (
    _status(_INITIAL_PROMPT_INDEX, len(_INITIAL_SAVED), "Default Pipeline",
            " | ".join(
                part
                for part in [
                    "LOADED FROM DISK" if _INITIAL_SAVED else "",
                    "DEFAULT DATA UNAVAILABLE" if DATASET_WARNING else "",
                ]
                if part
            ))
)

def render_lean_builder_ui():
    gr.Markdown(
        "# Lean Code Generator\n"
        "Feed prompts from `lean_prompt_data.json` to an LLM one at a time, "
        "or enter your own custom prompt.  \n"
        "Review each generated Lean output before approving."
    )

    # --- State ---
    session_state = gr.State(None)
    prompt_index_state = gr.State(_INITIAL_PROMPT_INDEX)
    outputs_state = gr.State(_INITIAL_SAVED)
    current_prompt_state = gr.State("")
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
            info="Default: iterates through lean_prompt_data.json. Custom: enter your own prompt.",
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
                lines=18,
                max_lines=40,
                interactive=False,
                autoscroll=False,
            )
            gr.Markdown("COMPILATION OUTPUT", elem_classes=["panel-label"])
            compile_panel = gr.Textbox(
                value="*No compilation run yet.*",
                lines=8,
                max_lines=18,
                interactive=False,
                autoscroll=False,
            )
            gr.Markdown("ALIGNMENT EVALUATION", elem_classes=["panel-label"])
            alignment_panel = gr.Markdown(
                ALIGNMENT_WAITING,
                elem_classes=["eval-panel"],
            )

    # --- Review buttons ---
    with gr.Row(elem_classes=["review-row"]):
        approve_btn = gr.Button("Approve", variant="primary", visible=False, min_width=140)
        regenerate_btn = gr.Button("Regenerate", variant="secondary", visible=False, min_width=140)
        skip_btn = gr.Button("Skip", variant="stop", visible=False, min_width=140)

    status_md = gr.Markdown(_INITIAL_STATUS, elem_classes=["status-bar"])

    msg = gr.Textbox(
        label="Message",
        placeholder="After alignment, request changes such as `improve alignment`...",
        autofocus=True,
    )

    # --- Action buttons ---
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        clear_btn = gr.Button("New Session")

    # --- Shared output list ---
    panel_outputs = [
        prompt_panel, output_panel, compile_panel, alignment_panel,
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
        inputs=[provider_dropdown, reasoning_effort_dropdown],
        outputs=[model_dropdown, reasoning_effort_dropdown, *panel_outputs, msg],
    )

    model_dropdown.change(
        on_model_change,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown],
        outputs=[reasoning_effort_dropdown],
    )

    generate_btn.click(
        send_prompt,
        inputs=[provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, current_prompt_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    regenerate_btn.click(
        on_regenerate,
        inputs=[prompt_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, current_prompt_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    skip_btn.click(
        on_skip,
        inputs=[provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, mode_radio, custom_prompt_box],
        outputs=panel_outputs,
    )

    msg.submit(
        capture_msg,
        inputs=[msg],
        outputs=[msg, pending_msg_state],
    ).then(
        user_refine_submit,
        inputs=[pending_msg_state, prompt_panel, output_panel, compile_panel, alignment_panel,
                provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, current_prompt_state, mode_radio],
        outputs=panel_outputs,
    )

    clear_btn.click(clear_session, outputs=[*panel_outputs, msg])


def create_lean_demo():
    with gr.Blocks(title="Lean Code Generator", css=CSS, theme=gr.themes.Soft()) as demo:
        render_lean_builder_ui()
    return demo


def _switch_workspace_view(target: str):
    return (
        gr.update(visible=(target == "qa")),
        gr.update(visible=(target == "lean")),
        gr.update(visible=(target == "postprocessing")),
    )


def create_workbench_demo(initial_view: str = "lean"):
    import app as app_module
    import postprocessing_app as post_module

    combined_css = "\n".join([app_module.CSS, CSS, post_module.CSS])
    with gr.Blocks(title="Formal Science Workbench", css=combined_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["review-row"]):
            qa_nav_btn = gr.Button("QA Dataset Builder", variant="secondary", min_width=180)
            lean_nav_btn = gr.Button("Lean Code Generator", variant="primary", min_width=180)
            post_nav_btn = gr.Button("Postprocessing", variant="secondary", min_width=180)

        with gr.Column(visible=(initial_view == "qa")) as qa_view:
            app_module.render_qa_builder_ui()

        with gr.Column(visible=(initial_view == "lean")) as lean_view:
            render_lean_builder_ui()

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
    create_workbench_demo(initial_view="lean").launch()
