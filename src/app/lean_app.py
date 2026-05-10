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
DEFAULT_APPROVED_FORMAL_BATCHES_PATH = _SRC / "app_data" / "approved_formal_batches.json"
REPO_ROOT = _SRC.parent
DEFAULT_PROOF_PATH = REPO_ROOT / "FSLean" / "proof.lean"
DEFAULT_CRITIC_OUTPUT_PATH = REPO_ROOT / "critic_output.md"
DEFAULT_CRITIC_DECISION_PATH = REPO_ROOT / "critic_decision.json"
DEFAULT_CRITIC_COUNT_PATH = REPO_ROOT / "critic_count.json"
DEFAULT_COMPILE_FIX_COUNT_PATH = REPO_ROOT / "compile_fix_count.json"
DEFAULT_PIPELINE_TIMER_PATH = REPO_ROOT / "pipeline_timer.json"
DEFAULT_RUBRIC_PATH = _SRC / "app" / "autoformalisation_rubric.md"
DEFAULT_DATASET_SETUP_MESSAGE = (
    "Default pipeline data is missing. Populate `src/app_data/qa_data.json` first, or "
    "run the QA builder (`python src/app/app.py`) to generate both `qa_data.json` and "
    "`lean_prompt_data.json`."
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


def _normalize_qa_batches(qa_batches: object, qa_path: Path) -> list[list[dict[str, str]]]:
    if not isinstance(qa_batches, list):
        raise RuntimeError(f"{qa_path} must contain a top-level JSON list.")

    normalized_batches: list[list[dict[str, str]]] = []
    for batch_idx, qa_batch in enumerate(qa_batches, start=1):
        if not isinstance(qa_batch, list):
            raise RuntimeError(f"{qa_path.name}[{batch_idx - 1}] must be a list of question/answer items.")

        normalized_batch = []
        for qa_idx, qa_item in enumerate(qa_batch, start=1):
            if not isinstance(qa_item, dict):
                raise RuntimeError(
                    f"{qa_path.name}[{batch_idx - 1}][{qa_idx - 1}] must be an object "
                    "with `question` and `answer`."
                )

            question = qa_item.get("question")
            answer = qa_item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise RuntimeError(
                    f"{qa_path.name}[{batch_idx - 1}][{qa_idx - 1}] must have string "
                    "`question` and `answer` values."
                )

            normalized_batch.append({
                "question": question,
                "answer": answer,
            })

        normalized_batches.append(normalized_batch)

    return normalized_batches


def _flatten_qa_batches(qa_batches: list[list[dict[str, str]]]) -> list[dict]:
    flattened: list[dict] = []
    for batch_index, qa_batch in enumerate(qa_batches, start=1):
        batch_size = len(qa_batch)
        for qa_index_in_batch, qa_pair in enumerate(qa_batch, start=1):
            flattened.append(
                {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "source_batch_index": batch_index,
                    "qa_index_in_batch": qa_index_in_batch,
                    "source_batch_size": batch_size,
                }
            )
    return flattened


def _load_prompt_qa_pairs(
    prompt_path: Path = DEFAULT_PROMPT_DATA_PATH,
    qa_path: Path = DEFAULT_QA_DATA_PATH,
) -> list[dict]:
    normalized_batches = _normalize_qa_batches(_read_json(qa_path), qa_path)
    flattened_pairs = _flatten_qa_batches(normalized_batches)

    prompts: list[str] | None = None
    prompt_load_error: RuntimeError | None = None
    try:
        raw_prompts = _read_json(prompt_path)
        if not isinstance(raw_prompts, list):
            raise RuntimeError(f"{prompt_path} must contain a top-level JSON list.")
        if len(raw_prompts) != len(flattened_pairs):
            raise RuntimeError(
                "Prompt/QA pair count mismatch: "
                f"{prompt_path.name} has {len(raw_prompts)} items but flattened {qa_path.name} has "
                f"{len(flattened_pairs)} QA pairs."
            )
        prompts = []
        for prompt_idx, prompt in enumerate(raw_prompts, start=1):
            if not isinstance(prompt, str):
                raise RuntimeError(f"{prompt_path.name}[{prompt_idx - 1}] must be a string prompt.")
            prompts.append(prompt)
        if any(_prompt_requires_rebuild(prompt) for prompt in prompts):
            prompts = None
    except RuntimeError as exc:
        prompt_load_error = exc

    if prompts is None:
        try:
            prompts = build_lean_prompt_dataset_from_file(qa_path)
        except Exception as build_exc:
            if prompt_load_error is not None and prompt_path.exists():
                raise prompt_load_error
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

    paired_data = []
    for prompt, flattened_pair in zip(prompts, flattened_pairs):
        paired_data.append({
            "lean_prompt": prompt,
            "qa_batch": [{
                "question": flattened_pair["question"],
                "answer": flattened_pair["answer"],
            }],
            "source_batch_index": flattened_pair["source_batch_index"],
            "qa_index_in_batch": flattened_pair["qa_index_in_batch"],
            "source_batch_size": flattened_pair["source_batch_size"],
        })

    return paired_data


def _prompt_requires_rebuild(prompt: str) -> bool:
    stale_markers = (
        "must be named `C",
        "original batch of size",
        "single QA pair from a larger batch",
    )
    return any(marker in prompt for marker in stale_markers)


def _initialize_prompt_data() -> tuple[list[dict], str | None]:
    try:
        return _load_prompt_qa_pairs(), None
    except RuntimeError as exc:
        if not DEFAULT_QA_DATA_PATH.exists():
            return [], DEFAULT_DATASET_SETUP_MESSAGE
        raise


def _set_prompt_data(prompt_qa_pairs: list[dict], warning: str | None) -> None:
    global PROMPT_QA_PAIRS, PROMPTS, TOTAL_PROMPTS, DATASET_WARNING, ORIGINAL_QA_BATCHES
    PROMPT_QA_PAIRS = prompt_qa_pairs
    PROMPTS = [entry["lean_prompt"] for entry in prompt_qa_pairs]
    TOTAL_PROMPTS = len(PROMPTS)
    DATASET_WARNING = warning
    ORIGINAL_QA_BATCHES = _rebuild_original_qa_batches(prompt_qa_pairs)


def _rebuild_original_qa_batches(prompt_qa_pairs: list[dict]) -> list[list[dict[str, str]]]:
    batches: dict[int, list[dict[str, str] | None]] = {}
    for entry in prompt_qa_pairs:
        batch_index = entry.get("source_batch_index")
        qa_index = entry.get("qa_index_in_batch")
        batch_size = entry.get("source_batch_size")
        qa_batch = entry.get("qa_batch")
        if not isinstance(batch_index, int) or not isinstance(qa_index, int) or not isinstance(batch_size, int):
            continue
        if not isinstance(qa_batch, list) or len(qa_batch) != 1 or not isinstance(qa_batch[0], dict):
            continue
        slots = batches.setdefault(batch_index, [None] * batch_size)
        if len(slots) != batch_size or not (1 <= qa_index <= batch_size):
            continue
        slots[qa_index - 1] = {
            "question": qa_batch[0]["question"],
            "answer": qa_batch[0]["answer"],
        }

    rebuilt: list[list[dict[str, str]]] = []
    for batch_index in sorted(batches):
        batch_items = batches[batch_index]
        if any(item is None for item in batch_items):
            break
        rebuilt.append([item for item in batch_items if item is not None])
    return rebuilt


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


def _autosave_structured_proofs(entries: list[list[dict[str, str]]]):
    try:
        DEFAULT_STRUCTURED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_STRUCTURED_OUTPUT_PATH.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_STRUCTURED_OUTPUT_PATH}: {exc}") from exc


def _autosave_approved_formal_batches(entries: list[list[dict[str, str]]]):
    try:
        DEFAULT_APPROVED_FORMAL_BATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_APPROVED_FORMAL_BATCHES_PATH.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_APPROVED_FORMAL_BATCHES_PATH}: {exc}") from exc


def _load_rubric_text() -> str:
    try:
        return DEFAULT_RUBRIC_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_RUBRIC_PATH}: {exc}") from exc


def _write_critic_output(text: str):
    try:
        DEFAULT_CRITIC_OUTPUT_PATH.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_CRITIC_OUTPUT_PATH}: {exc}") from exc


def _read_critic_output() -> str:
    try:
        return DEFAULT_CRITIC_OUTPUT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_CRITIC_OUTPUT_PATH}: {exc}") from exc


def _delete_critic_output():
    try:
        DEFAULT_CRITIC_OUTPUT_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_CRITIC_OUTPUT_PATH}: {exc}") from exc


def _write_critic_decision(decision: bool):
    try:
        DEFAULT_CRITIC_DECISION_PATH.write_text(
            json.dumps({"decision": bool(decision)}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_CRITIC_DECISION_PATH}: {exc}") from exc


def _read_critic_decision() -> bool | None:
    try:
        raw = DEFAULT_CRITIC_DECISION_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_CRITIC_DECISION_PATH}: {exc}") from exc
    try:
        value = json.loads(raw).get("decision")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def _delete_critic_decision():
    try:
        DEFAULT_CRITIC_DECISION_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_CRITIC_DECISION_PATH}: {exc}") from exc


def _read_critic_count() -> int:
    try:
        raw = DEFAULT_CRITIC_COUNT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_CRITIC_COUNT_PATH}: {exc}") from exc
    try:
        value = json.loads(raw).get("count", 0)
    except (ValueError, AttributeError):
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _write_critic_count(count: int):
    try:
        DEFAULT_CRITIC_COUNT_PATH.write_text(
            json.dumps({"count": int(count)}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_CRITIC_COUNT_PATH}: {exc}") from exc


def _delete_critic_count():
    try:
        DEFAULT_CRITIC_COUNT_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_CRITIC_COUNT_PATH}: {exc}") from exc


def _read_compile_fix_count() -> int:
    try:
        raw = DEFAULT_COMPILE_FIX_COUNT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_COMPILE_FIX_COUNT_PATH}: {exc}") from exc
    try:
        value = json.loads(raw).get("count", 0)
    except (ValueError, AttributeError):
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _write_compile_fix_count(count: int):
    try:
        DEFAULT_COMPILE_FIX_COUNT_PATH.write_text(
            json.dumps({"count": int(count)}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_COMPILE_FIX_COUNT_PATH}: {exc}") from exc


def _delete_compile_fix_count():
    try:
        DEFAULT_COMPILE_FIX_COUNT_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_COMPILE_FIX_COUNT_PATH}: {exc}") from exc


def _read_pipeline_record() -> dict | None:
    try:
        raw = DEFAULT_PIPELINE_TIMER_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Failed to read {DEFAULT_PIPELINE_TIMER_PATH}: {exc}") from exc
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _read_pipeline_start_time() -> float | None:
    record = _read_pipeline_record()
    if record is None:
        return None
    value = record.get("start_time")
    return float(value) if isinstance(value, (int, float)) else None


def _write_pipeline_start_time(start_time: float, task_key: str | None = None):
    payload: dict[str, object] = {"start_time": float(start_time)}
    if task_key is not None:
        payload["task_key"] = task_key
    try:
        DEFAULT_PIPELINE_TIMER_PATH.write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {DEFAULT_PIPELINE_TIMER_PATH}: {exc}") from exc


def _ensure_pipeline_start_time(task_key: str) -> float:
    """Start the per-QA-pair timer if this is a new task, otherwise preserve it.

    Same task_key (regenerate, retry) keeps the existing start_time so cumulative
    wall-clock for one QA pair is preserved. A different task_key (skip → next
    prompt, new custom prompt) writes a fresh start_time.
    """
    record = _read_pipeline_record()
    if record is not None and record.get("task_key") == task_key:
        existing = record.get("start_time")
        if isinstance(existing, (int, float)):
            return float(existing)
    new_start_time = time.time()
    _write_pipeline_start_time(new_start_time, task_key=task_key)
    return new_start_time


def _pipeline_task_key(prompt_index: int, prompt_text: str) -> str:
    return f"{prompt_index}\x1f{prompt_text}"


def _delete_pipeline_timer():
    try:
        DEFAULT_PIPELINE_TIMER_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete {DEFAULT_PIPELINE_TIMER_PATH}: {exc}") from exc


def _extract_critic_decision_from_text(text: str) -> bool | None:
    if not text:
        return None
    match = re.search(r'\{\s*"decision"\s*:\s*(true|false)\s*\}', text)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _rebuild_batch_artifacts_from_saved_outputs(
    saved_outputs: list[dict],
) -> tuple[list[list[dict[str, str]]], list[list[dict[str, str]]]]:
    structured_entries: list[list[dict[str, str]]] = []
    approved_batches: list[list[dict[str, str]]] = []
    output_cursor = 0

    for batch_index, qa_batch in enumerate(ORIGINAL_QA_BATCHES, start=1):
        batch_size = len(qa_batch)
        if output_cursor + batch_size > len(saved_outputs):
            break

        batch_outputs = saved_outputs[output_cursor:output_cursor + batch_size]
        structured_batch: list[dict[str, str]] = []
        approved_batch: list[dict[str, str]] = []

        for qa_index_in_batch, (qa_item, saved_entry) in enumerate(zip(qa_batch, batch_outputs), start=1):
            prompt_entry = PROMPT_QA_PAIRS[output_cursor + qa_index_in_batch - 1]
            if (
                prompt_entry.get("source_batch_index") != batch_index
                or prompt_entry.get("qa_index_in_batch") != qa_index_in_batch
            ):
                raise RuntimeError(
                    "Saved output order no longer matches the flattened QA order; "
                    "cannot rebuild batch artifacts safely."
                )

            proof_code = saved_entry.get("output")
            if not isinstance(proof_code, str) or not proof_code.strip():
                raise RuntimeError(f"Saved output {output_cursor + qa_index_in_batch} is missing Lean code.")

            critic_output = saved_entry.get("critic_output", "")
            if not isinstance(critic_output, str):
                critic_output = ""

            critic_count = saved_entry.get("critic_count", 0)
            if not isinstance(critic_count, int) or critic_count < 0:
                critic_count = 0

            critic_decision = saved_entry.get("critic_decision")
            if not isinstance(critic_decision, bool):
                critic_decision = None

            elapsed_seconds = saved_entry.get("elapsed_seconds", 0.0)
            if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
                elapsed_seconds = 0.0

            avg_compile_fix_per_critic_call = saved_entry.get("avg_compile_fix_per_critic_call", 0.0)
            if not isinstance(avg_compile_fix_per_critic_call, (int, float)) or avg_compile_fix_per_critic_call < 0:
                avg_compile_fix_per_critic_call = 0.0

            structured_batch.append(
                {
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                    "formal_proof": proof_code,
                    "critic_output": critic_output,
                    "critic_count": critic_count,
                    "critic_decision": critic_decision,
                    "elapsed_seconds": float(elapsed_seconds),
                    "avg_compile_fix_per_critic_call": float(avg_compile_fix_per_critic_call),
                }
            )
            approved_batch.append(
                {
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                    "formal_answer": proof_code,
                }
            )

        structured_entries.append(structured_batch)
        approved_batches.append(approved_batch)
        output_cursor += batch_size

    return structured_entries, approved_batches


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
Return the complete Lean 4 source code as your entire response.
Do not use tools.
Do not write to any file yourself.
Do not wrap the code in markdown fences.
Do not include any explanation or commentary before or after the code.
""".strip()

COMPILE_FIX_SUFFIX = """
The Lean file did not compile.

Return the complete corrected Lean 4 source code as your entire response.
Do not use tools.
Do not write to any file yourself.
Do not wrap the code in markdown fences.
Do not include any explanation or commentary before or after the code.
""".strip()

MAX_AUTO_REPAIRS = 3

CRITIC_WAITING = "*Critic output will appear here after the next review pass…*"
CRITIC_RUNNING = "*Running rubric critic…*"
CRITIC_FIELD_LABELS = [
    "Decision",
    "Scope label",
    "Target theorem fidelity",
    "Object fidelity",
    "Burden discharge",
    "Assumption hygiene",
    "Overall score",
    "Rejection conditions applied",
    "Score caps applied",
    "Re-informalization",
    "Central burdens",
    "Surrogate objects",
    "Theorem-shaped assumptions",
    "Definitions encoding conclusions",
    "External facts requiring replacement or stronger justification",
    "Required repairs",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(prompt_index, saved_count, extra=""):
    parts = [f"**{saved_count}** outputs saved"]
    if TOTAL_PROMPTS:
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
        session_kwargs["tools"] = [str(REPO_ROOT)]

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


def _run_nonstream_turn_with_heartbeats(
    provider,
    message,
    session_id,
    model,
    reasoning_effort,
):
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            session = _make_session(provider, session_id, model, reasoning_effort)
            response = session.prompt(message)
            next_session_id = getattr(session, "session_id", None) or session_id
            result_queue.put(("result", (_response_result_text(response), next_session_id)))
        except Exception as exc:  # pragma: no cover - exercised through caller
            result_queue.put(("error", exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    start = time.monotonic()
    next_terminal_update = 5
    while True:
        try:
            status, payload = result_queue.get(timeout=0.25)
        except queue.Empty:
            elapsed = time.monotonic() - start
            seconds = int(elapsed)
            if seconds >= next_terminal_update:
                print(
                    f"[lean_app] waiting on model... {seconds}s elapsed",
                    file=sys.stderr,
                    flush=True,
                )
                next_terminal_update += 5

            yield ("heartbeat", elapsed)
            continue

        if status == "error":
            raise payload
        yield ("result", payload)
        return


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


def _format_patch_summary(previous_text: str | None, current_text: str) -> str:
    diff_text = _build_brief_patch_diff(
        previous_text,
        current_text,
        from_label="previous_proof.lean",
        to_label="current_proof.lean",
        context_lines=1,
        max_lines=24,
    )
    if not diff_text:
        return "Patch summary unavailable."
    if diff_text == "No textual differences detected.":
        return diff_text
    return f"Patch summary for `FSLean/proof.lean`:\n```diff\n{diff_text}\n```"


def _normalize_inline_code_response(text: str) -> str:
    normalized = (text or "").strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[A-Za-z0-9_-]*\s*\n?", "", normalized, count=1)
        normalized = re.sub(r"\n?```$", "", normalized, count=1)
    return normalized.strip() + ("\n" if normalized.strip() else "")


def _persist_inline_code_response(
    response_text: str,
    *,
    overwrite_existing: bool = False,
) -> str | None:
    existing = _read_proof_source()
    if existing.strip() and not overwrite_existing:
        return existing
    normalized = _normalize_inline_code_response(response_text)
    if not normalized:
        return None
    return _write_proof_source(normalized)


def _build_model_message(prompt_text: str) -> str:
    return f"{prompt_text.rstrip()}\n\n{CODE_ONLY_SUFFIX}\n"


def _qa_context_for_prompt(prompt_index: int, current_prompt: str) -> list[dict] | None:
    if prompt_index < 0 or prompt_index >= len(PROMPT_QA_PAIRS):
        return None
    return PROMPT_QA_PAIRS[prompt_index]["qa_batch"]


def _build_critic_message(
    prompt_text: str,
    proof_code: str,
    rubric_text: str,
    *,
    prompt_index: int,
    stage: str,
    previous_proof_code: str | None = None,
) -> str:
    qa_batch = _qa_context_for_prompt(prompt_index, prompt_text)
    if qa_batch is None:
        raise RuntimeError(
            f"No QA pair registered for prompt_index={prompt_index}; "
            "the critic requires a QA pair as the primary semantic target."
        )
    sections = [
        "You are the critic/auditor for Lean autoformalisation.",
        (
            "Stage: coarse semantic rubric screen before compile-fix.\n"
            "Evaluate whether the current `FSLean/proof.lean` is semantically faithful enough "
            "to justify starting compile-fix. If any sub-score is below 4.0, or any rejection "
            "condition or score cap applies, the decision must not be `accept`."
            if stage == "coarse"
            else
            "Stage: full rubric review after compilation.\n"
            "Audit the current `FSLean/proof.lean` for final acceptance against the rubric gate."
        ),
        (
            "When the file contains multiple target theorems, evaluate them in order against the "
            "corresponding QA items. The top-level decision and scores must be batch-level minima "
            "over the weakest theorem in the batch."
        ),
    ]

    sections.append(
        "Current QA batch (this is the primary semantic target):\n```json\n"
        + json.dumps(qa_batch, indent=2, ensure_ascii=False)
        + "\n```"
    )

    sections.extend([
        "Original Lean generation prompt:\n" + (prompt_text or "").strip(),
        "Current proof.lean code:\n```lean\n" + f"{proof_code.rstrip()}\n```",
    ])

    patch_diff = _build_brief_patch_diff(
        previous_proof_code,
        proof_code,
        from_label="previous_proof.lean",
        to_label="current_proof.lean",
    )
    if patch_diff:
        sections.append("Patch difference from previous reviewed proof:\n```diff\n" + f"{patch_diff}\n```")

    sections.extend([
        "Rubric source (`autoformalisation_rubric.md`):\n" + rubric_text,
        (
            "Return markdown only. Use these exact top-level field labels and keep every score to "
            "1 decimal place:\n- "
            + "\n- ".join(CRITIC_FIELD_LABELS)
        ),
        (
            "Scoring rules:\n"
            "- Use the four rubric dimensions directly.\n"
            "- Use scores from 0.0 to 5.0.\n"
            "- Set `Overall score` to the minimum of the four sub-scores, never the average.\n"
            "- Rejection conditions are the items in rubric §10 (Final Rejection Checklist). "
            "If ANY §10 item applies, you MUST set `Decision` to `reject` (or `repair` if the "
            "issue is fixable) and lower at least one relevant sub-score to ≤ 2.0 so the "
            "minimum-of-four rule blocks acceptance. Apply the §9 score caps (max 2) for the "
            "specific sub-score they target.\n"
            "- If rejection conditions apply, list each one under `Rejection conditions applied` "
            "with the rubric §10 item number it triggers and the sub-score(s) you lowered.\n"
            "- If no rejection condition applies, write `- None.` under `Rejection conditions applied`.\n"
            "- If no score cap applies, write `- None.` under `Score caps applied`.\n"
            "- Under `Required repairs`, give concrete edits to theorem statements, object models, burdens, or assumptions.\n"
            "- Under `Re-informalization`, explicitly restate the weakest theorem in plain English and say whether it answers the QA pair."
        ),
        (
            "Decision artifact:\n"
            "- At the very end of your markdown report, append a single fenced JSON block.\n"
            "- The block must contain exactly `{\"decision\": true}` if you accept the proof against the rubric gate, or `{\"decision\": false}` otherwise.\n"
            "- Do not attempt to write any file yourself; the app will persist this decision to `critic_decision.json`.\n"
            "- This decision is authoritative for stopping the auto-repair loop."
        ),
    ])
    return "\n\n".join(sections)


def _extract_critic_section(text: str, label: str) -> str:
    next_labels = [item for item in CRITIC_FIELD_LABELS if item != label]
    next_pattern = "|".join(re.escape(item) for item in next_labels)
    pattern = (
        rf"^{re.escape(label)}:\s*(.*?)(?=^(?:{next_pattern}):|\Z)"
        if next_pattern
        else
        rf"^{re.escape(label)}:\s*(.*)\Z"
    )
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _section_is_none(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .")
    if not normalized:
        return False
    return normalized in {
        "none",
        "- none",
        "n/a",
        "- n/a",
        "no",
        "not applied",
        "- not applied",
        "no rejection conditions applied",
        "- no rejection conditions applied",
        "no score caps applied",
        "- no score caps applied",
    }


SCALAR_CRITIC_FIELDS = (
    "Decision",
    "Scope label",
    "Target theorem fidelity",
    "Object fidelity",
    "Burden discharge",
    "Assumption hygiene",
    "Overall score",
)


def _normalize_critic_report_text(text: str) -> str:
    """Rewrite header-style labels into the inline `Label:` form the parser expects.

    Models commonly emit one of:
        **Decision**\n\nrepair          (markdown bold header + blank + value)
        ## Decision\n\nrepair            (markdown heading)
        Decision\nrepair                 (label-then-value)
        **Decision:** repair             (bold inline)

    All of these get rewritten to:
        Decision: repair

    For free-text sections (e.g. `Required repairs`), only the label line is
    normalized; the section body is left untouched so `_extract_critic_section`
    can still terminate at the next label.
    """
    lines = text.splitlines()
    output: list[str] = []
    scalar_set = set(SCALAR_CRITIC_FIELDS)
    label_set = set(CRITIC_FIELD_LABELS)
    label_line_pattern = re.compile(r"^[#*\s]*([^*#:\n]+?)[#*\s]*:?\s*$")
    inline_bold_pattern = re.compile(r"^\*+\s*([^*:\n]+?)\s*:\*+\s*(.+)$")

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        inline_match = inline_bold_pattern.match(stripped)
        if inline_match:
            candidate = inline_match.group(1).strip()
            if candidate in label_set:
                output.append(f"{candidate}: {inline_match.group(2).strip()}")
                i += 1
                continue

        label_match = label_line_pattern.match(stripped)
        canonical = None
        if label_match:
            candidate = label_match.group(1).strip()
            if candidate in label_set:
                canonical = candidate

        if canonical is None:
            output.append(raw)
            i += 1
            continue

        if canonical in scalar_set:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                output.append(f"{canonical}: {lines[j].strip()}")
                i = j + 1
            else:
                output.append(f"{canonical}:")
                i += 1
        else:
            output.append(f"{canonical}:")
            i += 1
    return "\n".join(output)


def _parse_critic_report(report_text: str) -> dict:
    raw_text = report_text
    report_text = _normalize_critic_report_text(report_text)
    parsed: dict[str, object] = {"raw_text": raw_text}
    errors: list[str] = []

    def _match_scalar(label: str, pattern: str) -> str | None:
        match = re.search(pattern, report_text, re.MULTILINE)
        if not match:
            errors.append(f"Missing or malformed `{label}`.")
            return None
        return match.group(1).strip()

    decision = _match_scalar("Decision", r"^Decision:\s*(accept|repair|re-scope|reject)\s*$")
    scope_label = _match_scalar(
        "Scope label",
        r"^Scope label:\s*(Unconditional|Research-grade conditional|Restricted|Not acceptable)\s*$",
    )

    score_patterns = {
        "target_theorem_fidelity": r"^Target theorem fidelity:\s*([0-5](?:\.\d)?)(?:\s*/\s*5)?\s*$",
        "object_fidelity": r"^Object fidelity:\s*([0-5](?:\.\d)?)(?:\s*/\s*5)?\s*$",
        "burden_discharge": r"^Burden discharge:\s*([0-5](?:\.\d)?)(?:\s*/\s*5)?\s*$",
        "assumption_hygiene": r"^Assumption hygiene:\s*([0-5](?:\.\d)?)(?:\s*/\s*5)?\s*$",
        "overall_score": r"^Overall score:\s*(?:min\s*=\s*)?([0-5](?:\.\d)?)(?:\s*/\s*5)?\s*$",
    }

    scores: dict[str, float] = {}
    for name, pattern in score_patterns.items():
        value = _match_scalar(name, pattern)
        if value is None:
            continue
        scores[name] = float(value)

    rejection_conditions_text = _extract_critic_section(report_text, "Rejection conditions applied")
    score_caps_text = _extract_critic_section(report_text, "Score caps applied")
    if not rejection_conditions_text:
        errors.append("Missing `Rejection conditions applied` section.")
    if not score_caps_text:
        errors.append("Missing `Score caps applied` section.")

    parsed.update(
        {
            "decision": decision,
            "scope_label": scope_label,
            **scores,
            "rejection_conditions_text": rejection_conditions_text,
            "score_caps_text": score_caps_text,
            "rejection_conditions_active": bool(rejection_conditions_text) and not _section_is_none(rejection_conditions_text),
            "score_caps_active": bool(score_caps_text) and not _section_is_none(score_caps_text),
        }
    )

    score_values = [
        scores.get("target_theorem_fidelity"),
        scores.get("object_fidelity"),
        scores.get("burden_discharge"),
        scores.get("assumption_hygiene"),
    ]
    if any(value is None for value in score_values):
        computed_min = None
    else:
        computed_min = min(score_values)
        if "overall_score" in scores and abs(scores["overall_score"] - computed_min) > 0.05:
            errors.append("`Overall score` does not match the minimum sub-score.")

    parsed["computed_min_score"] = computed_min
    parsed["parse_errors"] = errors
    parsed["is_valid"] = not errors
    return parsed


def _critic_report_allows_compile_fix(report: dict) -> bool:
    if not report.get("is_valid"):
        return False
    if report.get("rejection_conditions_active"):
        return False
    if report.get("score_caps_active"):
        return False
    if report.get("decision") in {"re-scope", "reject"}:
        return False

    score_names = [
        "target_theorem_fidelity",
        "object_fidelity",
        "burden_discharge",
        "assumption_hygiene",
        "overall_score",
    ]
    try:
        return all(float(report[name]) >= 4.0 for name in score_names)
    except (KeyError, TypeError, ValueError):
        return False


def _critic_report_allows_approval(report: dict) -> bool:
    return _critic_report_allows_compile_fix(report) and report.get("decision") == "accept"


def _critic_panel_text(report_text: str, parse_errors: list[str]) -> str:
    if not parse_errors:
        return report_text
    error_lines = "\n".join(f"- {error}" for error in parse_errors)
    return f"**Critic output parse error**\n\n{error_lines}\n\n---\n\n{report_text}"


def _default_pipeline_progress_count(saved_outputs: list[dict] | None = None) -> int:
    outputs = saved_outputs if saved_outputs is not None else _load_saved_outputs()
    if not isinstance(outputs, list):
        return 0

    if all(
        isinstance(entry, dict)
        and isinstance(entry.get("source_batch_index"), int)
        and isinstance(entry.get("qa_index_in_batch"), int)
        for entry in outputs
    ):
        return len(outputs)

    count = 0
    for entry in outputs:
        if not isinstance(entry, dict):
            break
        prompt = entry.get("prompt")
        output = entry.get("output")
        if not isinstance(prompt, str) or not isinstance(output, str):
            break
        count += 1
    if count and count == len(outputs):
        try:
            structured_data = json.loads(DEFAULT_STRUCTURED_OUTPUT_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return count
        if not isinstance(structured_data, list):
            return count

        flattened_batch_count = 0
        for entry in structured_data:
            if not isinstance(entry, dict):
                return count
            qa_batch = entry.get("qa_batch")
            if not isinstance(qa_batch, list):
                return count
            flattened_batch_count += len(qa_batch)
        return flattened_batch_count
    return count


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


def _reset_proof_source() -> str:
    try:
        DEFAULT_PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PROOF_PATH.write_text("", encoding="utf-8")
        return str(DEFAULT_PROOF_PATH)
    except OSError as exc:
        raise RuntimeError(f"Failed to reset {DEFAULT_PROOF_PATH}: {exc}") from exc


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


def _build_compile_fix_message(
    code: str,
    compiler_output: str,
    iteration: int,
    previous_attempt_diff: str | None = None,
) -> str:
    fidelity_constraint = (
        "Fidelity constraint: the fix must not trigger any rubric §10 rejection condition "
        "(no `sorry`/`admit`/hidden holes, do not weaken the target theorem, do not assume "
        "the central conclusion, do not delete or replace the central object, do not "
        "redefine objects so the theorem becomes trivially true) and must not take an "
        "\"easy fix\" that misaligns the code with the informal claim (no surrogate "
        "definitions, no theorem-shaped hypotheses, no scope downgrade just to compile)."
    )
    parts = [
        COMPILE_FIX_SUFFIX,
        fidelity_constraint,
        f"Compile attempt: {iteration}",
    ]
    if previous_attempt_diff:
        parts.append(
            "Previous fix attempt diff (the previous LLM revision applied this patch and the "
            "code below — the post-patch state — still fails to compile; do not repeat the "
            "previous attempt's approach):\n"
            "```diff\n"
            f"{previous_attempt_diff}\n"
            "```"
        )
    parts.extend([
        f"Compiler output:\n{compiler_output}",
        "Current Lean code:\n"
        "```lean\n"
        f"{code.rstrip()}\n"
        "```",
    ])
    return "\n\n".join(parts)


def _run_critic(
    provider: str,
    model,
    reasoning_effort,
    prompt_text: str,
    proof_code: str,
    *,
    prompt_index: int,
    stage: str,
    previous_proof_code: str | None = None,
) -> tuple[str, dict]:
    rubric_text = _load_rubric_text()
    critic_message = _build_critic_message(
        prompt_text,
        proof_code,
        rubric_text,
        prompt_index=prompt_index,
        stage=stage,
        previous_proof_code=previous_proof_code,
    )
    _delete_critic_decision()
    critic_text, _ = _run_nonstream_turn(
        provider,
        critic_message,
        None,
        model,
        reasoning_effort,
    )
    _write_critic_output(critic_text)
    report = _parse_critic_report(critic_text)

    decision = _read_critic_decision()
    if decision is None:
        decision = _extract_critic_decision_from_text(critic_text)
    if decision is None:
        decision = _critic_report_allows_approval(report)
    _write_critic_decision(decision)
    return critic_text, report


def _build_refinement_message(
    provider: str,
    user_instruction: str,
    prompt_text: str,
    proof_code: str,
    alignment_feedback: str | None,
    prompt_index: int,
) -> str:
    qa_batch = _qa_context_for_prompt(prompt_index, prompt_text)
    rubric_text = _load_rubric_text()
    task_header = (
        "Improve the lean code in `proof.lean` using `critic_output.md`, such that it perfectly "
        "aligns with `autoformalisation_rubric.md` and the QA pair.\n"
    )
    execution_instructions = (
        "Return the complete updated Lean 4 source code as your entire response.\n"
        "Do not use tools.\n"
        "Do not write to any file yourself.\n"
        "Do not wrap the code in markdown fences.\n"
        "Do not include any explanation or commentary before or after the code.\n\n"
    )
    sections = [
        task_header
        + execution_instructions
        + "Hard constraints:\n"
        + "1. Do not introduce any new hypothesis that contains a central conclusion of the informal answer.\n"
        + "2. Do not define an object so that the desired theorem follows by unfolding.\n"
        + "3. Do not replace the informal object with a surrogate; use faithful objects or precise bridge theorems.\n"
        + "4. For every central burden, either prove it in Lean, import it from Mathlib, mark it as explicit input, or state a standard independent external theorem.\n"
        + "5. If a central burden cannot be discharged, downgrade the scope rather than hiding the failure.\n"
        + "6. Include the rubric-required Lean comments for scope, input data, central burden discharge, external theorem independence, object-fidelity notes, and self-audit.\n"
        + "7. Preserve the domain, objects, constants, quantifiers, and conclusion of each target theorem.",
    ]

    if qa_batch is not None:
        sections.append(
            "Current QA batch:\n```json\n"
            + json.dumps(qa_batch, indent=2, ensure_ascii=False)
            + "\n```"
        )

    sections.append("Initial prompt:\n" + (prompt_text or "").strip())

    normalized_feedback = _normalize_review_feedback(
        alignment_feedback,
        CRITIC_WAITING,
        CRITIC_RUNNING,
    )
    if normalized_feedback:
        sections.append("Latest critic output:\n" + normalized_feedback)

    sections.extend([
        "User request:\n" + user_instruction.strip(),
        "Current proof.lean code:\n```lean\n" + f"{proof_code.rstrip()}\n```",
        "Rubric source (`autoformalisation_rubric.md`):\n" + rubric_text,
    ])
    return "\n\n".join(sections)


def _auto_refinement_instruction(stage: str) -> str:
    if stage == "coarse":
        return (
            "Use the latest critic report to repair theorem fidelity, object fidelity, burden "
            "discharge, and assumption hygiene before compile-fix continues. Make the smallest "
            "changes that clear the rubric gate while preserving the QA semantics."
        )
    return (
        "Use the latest critic report to revise the proof for final rubric acceptance. Address "
        "every required repair while preserving fidelity to the QA pair."
    )


def _post_auto_repair_failure_message(stage: str) -> str:
    if stage == "coarse":
        return (
            "An automatic repair session already ran, but the latest coarse critic still did not "
            "clear the rubric gate.\n\n"
            "`critic_output.md` now contains the newest post-repair critic report. "
            "Review that report and repair theorem fidelity, object fidelity, burden discharge, "
            "or assumption hygiene before compiling again."
        )
    return (
        "An automatic repair session already ran, but the latest full critic still did not accept "
        "the proof.\n\n"
        "`critic_output.md` now contains the newest post-repair critic report. "
        "Review that report and address the remaining required repairs before approving."
    )


def _run_refinement_pass(
    prompt_panel,
    output_panel,
    alignment_panel,
    provider,
    session_id,
    model,
    reasoning_effort,
    prompt_index,
    saved_outputs,
    current_prompt,
    instruction: str,
    previous_proof_code: str,
):
    refinement_message = _build_refinement_message(
        provider,
        instruction,
        current_prompt,
        previous_proof_code,
        alignment_panel,
        prompt_index,
    )
    repair_iteration = _read_critic_count() + 1
    refining_label = f"REFINING ({repair_iteration}/{MAX_AUTO_REPAIRS})"
    st = _status(prompt_index, len(saved_outputs), refining_label)

    yield (
        prompt_panel,
        previous_proof_code,
        "Updating `proof.lean` in a fresh repair session...",
        alignment_panel,
        session_id, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, False, False), st,
    )

    try:
        refine_result = ""
        last_sid = session_id
        for event_type, payload in _run_nonstream_turn_with_heartbeats(
            provider,
            refinement_message,
            None,
            model,
            reasoning_effort,
        ):
            if event_type == "heartbeat":
                yield (
                    prompt_panel,
                    _heartbeat_message(payload),
                    "Updating `proof.lean` in a fresh repair session...",
                    alignment_panel,
                    session_id, prompt_index, saved_outputs, current_prompt,
                    *_review_controls(False, False, False),
                    _status(prompt_index, len(saved_outputs), refining_label),
                )
                continue
            refine_result, last_sid = payload
    except Exception as exc:
        traceback.print_exc()
        yield (
            prompt_panel, output_panel, f"Refinement failed:\n{exc}", alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "REFINEMENT FAILED"),
        )
        return

    try:
        persisted_inline = _persist_inline_code_response(
            refine_result,
            overwrite_existing=True,
        )
        if persisted_inline is not None:
            refine_result = persisted_inline
    except RuntimeError as exc:
        yield (
            prompt_panel,
            output_panel,
            f"Failed to persist the refinement response:\n{exc}",
            alignment_panel,
            last_sid, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "FAILED TO WRITE PROOF"),
        )
        return

    try:
        persisted_lean_text = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_panel, output_panel, str(exc), alignment_panel,
            last_sid, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "FAILED TO READ PROOF"),
        )
        return

    if not persisted_lean_text.strip():
        yield (
            prompt_panel,
            refine_result or output_panel,
            "Agent did not update `FSLean/proof.lean`.",
            alignment_panel,
            last_sid, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "AGENT DID NOT WRITE PROOF"),
        )
        return

    _write_critic_count(_read_critic_count() + 1)

    patch_summary = _format_patch_summary(previous_proof_code, persisted_lean_text)
    yield (
        prompt_panel,
        persisted_lean_text,
        patch_summary,
        alignment_panel,
        last_sid, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, False, False),
        _status(prompt_index, len(saved_outputs), "REFINEMENT WRITTEN | PATCH SUMMARY"),
    )

    yield from _compile_then_critic(
        prompt_panel, provider, last_sid, model, reasoning_effort,
        prompt_index, saved_outputs, current_prompt,
        previous_evaluated_code=previous_proof_code,
    )


def capture_msg(message):
    return "", message


# ---------------------------------------------------------------------------
# Outputs tuple order (12 elements):
#   prompt_panel, output_panel, compile_panel, alignment_panel,
#   session_state, prompt_index_state, outputs_state, current_prompt_state,
#   approve_btn, regenerate_btn, skip_btn, status_md
# ---------------------------------------------------------------------------

def _compile_and_align(prompt_display, provider, session_id, model, reasoning_effort,
                       prompt_index, saved_outputs, current_prompt,
                       previous_evaluated_code: str | None = None,
                       max_compile_fix_attempts: int = 25):
    try:
        current_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "COMPILE FAILED"),
        )
        return

    if not current_code.strip():
        yield (
            prompt_display,
            "*Waiting...*",
            "Nothing saved in `FSLean/proof.lean` yet.",
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "NO SAVED PROOF"),
        )
        return

    current_session_id = session_id
    last_compile_text = ""
    previous_input_code: str | None = None

    for iteration in range(1, max_compile_fix_attempts + 1):
        _write_compile_fix_count(_read_compile_fix_count() + 1)
        yield (
            prompt_display,
            current_code,
            f"Iteration: {iteration}\n\nCompiling current Lean output...",
            CRITIC_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), f"COMPILE ITERATION {iteration}"),
        )

        try:
            returncode, stdout, stderr = compile_lean(current_code)
        except Exception as exc:
            traceback.print_exc()
            yield (
                prompt_display,
                current_code,
                f"Compilation runner failed:\n{exc}",
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), "COMPILATION RUNNER FAILED"),
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
                    CRITIC_WAITING,
                    current_session_id, prompt_index, saved_outputs, current_prompt,
                    *_review_controls(False, True, True),
                    _status(prompt_index, len(saved_outputs), "CRITIC BLOCKED"),
                )
                return

            yield (
                prompt_display,
                aligned_code,
                compile_text,
                CRITIC_RUNNING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, False, False),
                _status(prompt_index, len(saved_outputs), f"COMPILED IN {iteration} ITERATION(S) | FULL CRITIC"),
            )

            try:
                critic_text, critic_report = _run_critic(
                    provider,
                    model,
                    reasoning_effort,
                    current_prompt,
                    aligned_code,
                    prompt_index=prompt_index,
                    stage="full",
                    previous_proof_code=previous_evaluated_code,
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
                    _status(prompt_index, len(saved_outputs), "CRITIC FAILED"),
                )
                return

            critic_panel = _critic_panel_text(
                critic_text or CRITIC_WAITING,
                critic_report.get("parse_errors", []),
            )
            approval_ready = (
                _critic_report_allows_approval(critic_report)
                or _read_critic_decision() is True
            )
            current_count = _read_critic_count()
            budget_exhausted = current_count >= MAX_AUTO_REPAIRS

            if approval_ready:
                try:
                    next_saved_outputs = _persist_approved_entry(
                        aligned_code,
                        critic_text,
                        prompt_index,
                        saved_outputs,
                        current_prompt,
                    )
                except RuntimeError as exc:
                    yield (
                        prompt_display,
                        aligned_code,
                        f"{compile_text}\n\nAuto-approve failed:\n{exc}",
                        critic_panel,
                        current_session_id, prompt_index, saved_outputs, current_prompt,
                        *_review_controls(False, True, True),
                        _status(
                            prompt_index, len(saved_outputs),
                            f"COMPILED IN {iteration} ITERATION(S) | AUTO-APPROVE FAILED",
                        ),
                    )
                    return

                yield (
                    prompt_display,
                    aligned_code,
                    compile_text,
                    critic_panel,
                    current_session_id, prompt_index, next_saved_outputs, current_prompt,
                    *_review_controls(False, False, False),
                    _status(
                        prompt_index, len(next_saved_outputs),
                        f"COMPILED IN {iteration} ITERATION(S) | CRITIC ACCEPTED ({current_count}/{MAX_AUTO_REPAIRS} REPAIRS) | AUTO-APPROVED",
                    ),
                )

                yield from send_prompt(
                    provider, None, model, reasoning_effort,
                    prompt_index + 1, next_saved_outputs,
                )
                return

            if budget_exhausted:
                try:
                    next_saved_outputs = _persist_approved_entry(
                        aligned_code,
                        critic_text,
                        prompt_index,
                        saved_outputs,
                        current_prompt,
                    )
                except RuntimeError as exc:
                    yield (
                        prompt_display,
                        aligned_code,
                        f"{compile_text}\n\nAuto-approve failed:\n{exc}",
                        critic_panel,
                        current_session_id, prompt_index, saved_outputs, current_prompt,
                        *_review_controls(False, True, True),
                        _status(
                            prompt_index, len(saved_outputs),
                            f"AUTO REPAIR BUDGET EXHAUSTED ({MAX_AUTO_REPAIRS}/{MAX_AUTO_REPAIRS}) | AUTO-APPROVE FAILED",
                        ),
                    )
                    return

                yield (
                    prompt_display,
                    aligned_code,
                    compile_text,
                    critic_panel,
                    current_session_id, prompt_index, next_saved_outputs, current_prompt,
                    *_review_controls(False, False, False),
                    _status(
                        prompt_index, len(next_saved_outputs),
                        f"AUTO REPAIR BUDGET EXHAUSTED ({MAX_AUTO_REPAIRS}/{MAX_AUTO_REPAIRS}) | AUTO-APPROVED",
                    ),
                )

                yield from send_prompt(
                    provider, None, model, reasoning_effort,
                    prompt_index + 1, next_saved_outputs,
                )
                return

            yield (
                prompt_display,
                aligned_code,
                compile_text,
                critic_panel,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, False, False),
                _status(
                    prompt_index, len(saved_outputs),
                    f"AUTO REPAIR AFTER CRITIC ({current_count + 1}/{MAX_AUTO_REPAIRS})",
                ),
            )
            yield from _run_refinement_pass(
                prompt_display,
                aligned_code,
                critic_panel,
                provider,
                None,
                model,
                reasoning_effort,
                prompt_index,
                saved_outputs,
                current_prompt,
                _auto_refinement_instruction("full"),
                aligned_code,
            )
            return

        if returncode != 1:
            yield (
                prompt_display,
                current_code,
                compile_text,
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), f"UNEXPECTED EXIT CODE {returncode}"),
            )
            return

        yield (
            prompt_display,
            current_code,
            f"{compile_text}\n\nRequesting an LLM revision based on the current compiler error...",
            CRITIC_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), f"REQUESTING FIX {iteration}"),
        )

        previous_attempt_diff = None
        if previous_input_code is not None:
            previous_attempt_diff = _build_brief_patch_diff(
                previous_input_code,
                current_code,
                from_label="before_previous_fix.lean",
                to_label="after_previous_fix.lean",
            ) or None
        fix_message = _build_compile_fix_message(
            current_code, compile_text, iteration, previous_attempt_diff=previous_attempt_diff,
        )
        previous_input_code = current_code

        try:
            _fix_result_text, _ = _run_nonstream_turn(
                provider, fix_message, None, model, reasoning_effort
            )
        except Exception as exc:
            traceback.print_exc()
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nLLM revision request failed:\n{exc}",
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), "LLM FIX FAILED"),
            )
            return

        try:
            persisted_from_response = _persist_inline_code_response(
                _fix_result_text,
                overwrite_existing=True,
            )
            if persisted_from_response is not None:
                current_code = persisted_from_response
        except RuntimeError as exc:
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nFailed to persist the compile-fix response:\n{exc}",
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), "FAILED TO WRITE PROOF"),
            )
            return

        try:
            current_code = _read_proof_source()
        except RuntimeError as exc:
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nFailed to read `FSLean/proof.lean` after LLM fix:\n{exc}",
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), "FAILED TO READ PROOF"),
            )
            return

        if not current_code.strip():
            yield (
                prompt_display,
                current_code,
                f"{compile_text}\n\nAgent did not update `FSLean/proof.lean`.",
                CRITIC_WAITING,
                current_session_id, prompt_index, saved_outputs, current_prompt,
                *_review_controls(False, True, True),
                _status(prompt_index, len(saved_outputs), "AGENT DID NOT WRITE PROOF"),
            )
            return

        yield (
            prompt_display,
            current_code,
            f"{compile_text}\n\nLLM revision applied. Recompiling...",
            CRITIC_WAITING,
            current_session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), f"RECOMPILING AFTER FIX {iteration}"),
        )

    yield (
        prompt_display,
        current_code,
        (
            f"{last_compile_text}\n\n"
            f"Reached the compile-fix safeguard after {max_compile_fix_attempts} iterations.\n\n"
            "The latest generated Lean code is still shown in the output panel."
        ),
        CRITIC_WAITING,
        current_session_id, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, True, True),
        _status(prompt_index, len(saved_outputs), "COMPILE FIX LIMIT REACHED"),
    )


def _compile_then_critic(prompt_display, provider, session_id, model, reasoning_effort,
                         prompt_index, saved_outputs, current_prompt,
                         previous_evaluated_code: str | None = None):
    try:
        proof_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "CRITIC FAILED"),
        )
        return

    if not proof_code.strip():
        yield (
            prompt_display,
            "*Waiting...*",
            "Nothing saved in `FSLean/proof.lean` yet.",
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "NO SAVED PROOF"),
        )
        return

    yield (
        prompt_display,
        proof_code,
        "*Entering compile-fix loop...*",
        CRITIC_WAITING,
        session_id, prompt_index, saved_outputs, current_prompt,
        *_review_controls(False, False, False),
        _status(prompt_index, len(saved_outputs), "ENTERING COMPILE-FIX"),
    )

    yield from _compile_and_align(
        prompt_display, provider, session_id, model, reasoning_effort,
        prompt_index, saved_outputs, current_prompt,
        previous_evaluated_code=previous_evaluated_code,
    )


def send_prompt(provider, session_id, model, reasoning_effort,
                prompt_index, saved_outputs):
    """Send the current default-pipeline prompt, stop after proof.lean is written, then compile."""
    _refresh_prompt_data()
    if TOTAL_PROMPTS == 0:
        setup_message = DATASET_WARNING or DEFAULT_DATASET_SETUP_MESSAGE
        yield (
            setup_message,
            "*Waiting...*",
            "*No compilation run yet.*",
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, "",
            *_review_controls(False, False, False),
            _status(prompt_index, len(saved_outputs), "DEFAULT DATA UNAVAILABLE"),
        )
        return
    if prompt_index >= TOTAL_PROMPTS:
        done_msg = (
            f"All **{TOTAL_PROMPTS}** prompts processed.  \n"
            f"**{len(saved_outputs)}** outputs saved.  \n"
            "Results autosaved to `lean_output_data.json`."
        )
        yield ("*Pipeline complete.*", done_msg, "*No compilation run yet.*", CRITIC_WAITING,
               session_id, prompt_index, saved_outputs, "",
               *_review_controls(False, False, False),
               _status(prompt_index, len(saved_outputs), "DONE"))
        return
    prompt_text = PROMPTS[prompt_index]
    header = f"### Prompt {prompt_index + 1} / {TOTAL_PROMPTS}\n\n---\n\n"
    prompt_display = header + prompt_text

    try:
        _reset_proof_source()
        _delete_critic_output()
        _delete_critic_decision()
        _delete_critic_count()
        _write_compile_fix_count(0)
        _ensure_pipeline_start_time(_pipeline_task_key(prompt_index, prompt_text))
    except RuntimeError as exc:
        yield (
            prompt_display,
            "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "FAILED TO RESET PROOF"),
        )
        return

    message = _build_model_message(prompt_text)
    st = _status(prompt_index, len(saved_outputs), "WRITING PROOF")

    yield (
        prompt_display, "*Writing `FSLean/proof.lean`...*", "*No compilation run yet.*", CRITIC_WAITING,
        session_id, prompt_index, saved_outputs, prompt_text,
        *_review_controls(False, False, False), st
    )

    try:
        agent_result = ""
        last_sid = session_id
        for event_type, payload in _run_nonstream_turn_with_heartbeats(
            provider,
            message,
            session_id,
            model,
            reasoning_effort,
        ):
            if event_type == "heartbeat":
                yield (
                    prompt_display,
                    _heartbeat_message(payload),
                    "*No compilation run yet.*",
                    CRITIC_WAITING,
                    session_id, prompt_index, saved_outputs, prompt_text,
                    *_review_controls(False, False, False),
                    _status(prompt_index, len(saved_outputs), "WRITING PROOF"),
                )
                continue
            agent_result, last_sid = payload
    except Exception as exc:
        traceback.print_exc()
        yield (
            prompt_display,
            "*Waiting...*",
            f"Initial write request failed:\n{exc}",
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "INITIAL WRITE FAILED"),
        )
        return

    try:
        persisted_inline = _persist_inline_code_response(agent_result)
        if persisted_inline is not None:
            agent_result = persisted_inline
    except RuntimeError as exc:
        yield (
            prompt_display,
            agent_result or "*Waiting...*",
            f"Failed to persist the response into `FSLean/proof.lean`:\n{exc}",
            CRITIC_WAITING,
            last_sid, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "FAILED TO WRITE PROOF"),
        )
        return

    try:
        persisted_lean_text = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_display,
            agent_result or "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            last_sid, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "FAILED TO READ PROOF"),
        )
        return

    if not persisted_lean_text.strip():
        yield (
            prompt_display,
            agent_result or "*Waiting...*",
            "Model did not return Lean source.",
            CRITIC_WAITING,
            last_sid, prompt_index, saved_outputs, prompt_text,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "MODEL DID NOT WRITE PROOF"),
        )
        return

    yield from _compile_then_critic(
        prompt_display, provider, last_sid, model, reasoning_effort,
        prompt_index, saved_outputs, prompt_text,
    )


def _persist_approved_entry(proof_code, critic_output, prompt_index, saved_outputs,
                            current_prompt):
    """Build the saved-output entry and autosave artifacts. Returns next saved_outputs.

    Used by both interactive `on_approve` (after the critic gate passes) and the
    auto-approve path when the auto-repair budget is exhausted.
    """
    critic_count = _read_critic_count()
    compile_fix_total = _read_compile_fix_count()
    start_time = _read_pipeline_start_time()
    elapsed_seconds = (time.time() - start_time) if start_time is not None else 0.0
    critic_calls = critic_count + 1
    avg_compile_fix_per_critic_call = compile_fix_total / critic_calls

    entry = {
        "prompt": current_prompt or "",
        "output": proof_code,
        "critic_output": critic_output,
        "critic_count": critic_count,
        "critic_decision": _read_critic_decision(),
        "elapsed_seconds": elapsed_seconds,
        "avg_compile_fix_per_critic_call": avg_compile_fix_per_critic_call,
    }
    if 0 <= prompt_index < len(PROMPT_QA_PAIRS):
        prompt_entry = PROMPT_QA_PAIRS[prompt_index]
        entry.update(
            {
                "source_batch_index": prompt_entry.get("source_batch_index"),
                "qa_index_in_batch": prompt_entry.get("qa_index_in_batch"),
                "source_batch_size": prompt_entry.get("source_batch_size"),
            }
        )
    next_saved_outputs = saved_outputs + [entry]

    next_structured_proofs, next_approved_formal_batches = _rebuild_batch_artifacts_from_saved_outputs(
        next_saved_outputs
    )
    _autosave_outputs(next_saved_outputs)
    _autosave_structured_proofs(next_structured_proofs)
    _autosave_approved_formal_batches(next_approved_formal_batches)

    return next_saved_outputs


def on_approve(output_panel, provider, session_id, model, reasoning_effort,
               prompt_index, saved_outputs, current_prompt):
    """Save the current output and advance to the next prompt."""
    try:
        proof_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "APPROVE FAILED"),
        )
        return

    if not proof_code.strip():
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            "Cannot approve because `FSLean/proof.lean` is empty or missing.",
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "APPROVE BLOCKED"),
        )
        return

    try:
        critic_output = _read_critic_output()
    except RuntimeError as exc:
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "APPROVE FAILED"),
        )
        return

    try:
        next_saved_outputs = _persist_approved_entry(
            proof_code, critic_output, prompt_index, saved_outputs, current_prompt,
        )
    except RuntimeError as exc:
        yield (
            current_prompt or "*Waiting to start...*",
            output_panel or "*Waiting...*",
            str(exc),
            CRITIC_WAITING,
            session_id, prompt_index, saved_outputs, current_prompt or "",
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "APPROVE FAILED"),
        )
        return

    saved_outputs = next_saved_outputs

    prompt_index += 1
    yield from send_prompt(
        provider, None, model, reasoning_effort,
        prompt_index, saved_outputs,
    )


def on_regenerate(prompt_panel, provider, session_id, model, reasoning_effort,
                  prompt_index, saved_outputs, current_prompt):
    """Regenerate the current prompt output."""
    yield from send_prompt(
        provider, None, model, reasoning_effort,
        prompt_index, saved_outputs,
    )


def on_skip(provider, session_id, model, reasoning_effort,
            prompt_index, saved_outputs):
    """Skip this prompt without saving and advance."""
    prompt_index += 1
    yield from send_prompt(
        provider, None, model, reasoning_effort,
        prompt_index, saved_outputs,
    )


def user_refine_submit(message, prompt_panel, output_panel, compile_panel, alignment_panel,
                       provider, session_id, model, reasoning_effort,
                       prompt_index, saved_outputs, current_prompt):
    if not message or not message.strip():
        yield (
            prompt_panel, output_panel, compile_panel, alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(True, True, True),
            _status(prompt_index, len(saved_outputs)),
        )
        return

    try:
        proof_code = _read_proof_source()
    except RuntimeError as exc:
        yield (
            prompt_panel, output_panel, str(exc), alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "REFINEMENT FAILED"),
        )
        return

    if not proof_code.strip():
        yield (
            prompt_panel, output_panel, "Nothing saved in `FSLean/proof.lean` yet.", alignment_panel,
            session_id, prompt_index, saved_outputs, current_prompt,
            *_review_controls(False, True, True),
            _status(prompt_index, len(saved_outputs), "NO SAVED PROOF"),
        )
        return

    yield from _run_refinement_pass(
        prompt_panel,
        output_panel,
        alignment_panel,
        provider,
        session_id,
        model,
        reasoning_effort,
        prompt_index,
        saved_outputs,
        current_prompt,
        message,
        proof_code,
    )


def clear_session():
    _refresh_prompt_data()
    try:
        _delete_critic_output()
        _delete_critic_decision()
        _delete_critic_count()
        _delete_compile_fix_count()
        _delete_pipeline_timer()
    except RuntimeError:
        pass
    saved = _load_saved_outputs()
    start_index = _default_pipeline_progress_count(saved)
    status_extra = []
    if saved:
        status_extra.append("LOADED FROM DISK")
    if DATASET_WARNING:
        status_extra.append("DEFAULT DATA UNAVAILABLE")
    return (
        "*Waiting to start...*", "*Waiting...*", "*No compilation run yet.*", CRITIC_WAITING,
        None, start_index, saved, "",
        *_review_controls(False, False, False),
        _status(start_index, len(saved), " | ".join(status_extra)),
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
_INITIAL_PROMPT_INDEX = _default_pipeline_progress_count(_INITIAL_SAVED)
_INITIAL_STATUS = (
    _status(_INITIAL_PROMPT_INDEX, len(_INITIAL_SAVED),
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
        "Feed prompts from `lean_prompt_data.json` to an LLM one at a time.  \n"
        "Review each generated Lean output against the rubric critic before approving."
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
            gr.Markdown("CRITIC OUTPUT", elem_classes=["panel-label"])
            alignment_panel = gr.Markdown(
                CRITIC_WAITING,
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
        placeholder="After critic review, request repairs such as `remove theorem-shaped assumptions`...",
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
        approve_btn, regenerate_btn, skip_btn, status_md,
    ]

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
                prompt_index_state, outputs_state],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, current_prompt_state],
        outputs=panel_outputs,
    )

    regenerate_btn.click(
        on_regenerate,
        inputs=[prompt_panel, provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state, current_prompt_state],
        outputs=panel_outputs,
    )

    skip_btn.click(
        on_skip,
        inputs=[provider_dropdown, session_state, model_dropdown, reasoning_effort_dropdown,
                prompt_index_state, outputs_state],
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
                prompt_index_state, outputs_state, current_prompt_state],
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
    )


def create_workbench_demo(initial_view: str = "lean"):
    import app as app_module

    combined_css = "\n".join([app_module.CSS, CSS])
    with gr.Blocks(title="Formal Science Workbench", css=combined_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["review-row"]):
            qa_nav_btn = gr.Button("QA Dataset Builder", variant="secondary", min_width=180)
            lean_nav_btn = gr.Button("Lean Code Generator", variant="primary", min_width=180)

        with gr.Column(visible=(initial_view == "qa")) as qa_view:
            app_module.render_qa_builder_ui()

        with gr.Column(visible=(initial_view == "lean")) as lean_view:
            render_lean_builder_ui()

        qa_nav_btn.click(lambda: _switch_workspace_view("qa"), outputs=[qa_view, lean_view])
        lean_nav_btn.click(lambda: _switch_workspace_view("lean"), outputs=[qa_view, lean_view])

    return demo


if __name__ == "__main__":
    create_workbench_demo(initial_view="lean").launch()
