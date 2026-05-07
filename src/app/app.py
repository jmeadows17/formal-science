"""
Gradio QA Dataset Builder powered by Claude or GPT CLI.

Side-by-side layout: input reasoning (left) vs generated QA pairs (right),
with an automatic Likert-scale alignment evaluation below.
Human review (Approve / Reject) at every step.

Run: python src/app/app.py
"""

import sys
import json
import re
import tempfile
import traceback
from pathlib import Path

import gradio as gr

_SRC = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC / "llm"))
sys.path.insert(0, str(_SRC / "qa"))

from claude_cli import ClaudeSession, CLAUDE_REASONING_EFFORTS
from gpt_cli import GPTSession, VALID_REASONING_EFFORTS
from lean_prompts import build_lean_prompt_dataset
from qa_postprocessing import postprocess_raw_dataset

_DEFAULT_PROMPT_BODY_MARKER = "Now, the following **equation-only** derivations"
_VALID_JSON_STRING_ESCAPES = frozenset('"\\/bfnrt')
_LATEX_COMMANDS_WITH_JSON_ESCAPE_PREFIX = frozenset({
    "bar",
    "begin",
    "beta",
    "bigl",
    "bigr",
    "biggl",
    "biggr",
    "boxed",
    "forall",
    "frac",
    "nabla",
    "neq",
    "notin",
    "nu",
    "rangle",
    "rceil",
    "rfloor",
    "rho",
    "right",
    "rvert",
    "rVert",
    "tau",
    "text",
    "textbf",
    "textit",
    "tfrac",
    "theta",
    "times",
    "to",
})

PIPELINE_MODES = ("Core", "Deeper")
_PAPER_DATA_FILE_RE = re.compile(r"^(core|deeper)_(\d+)\.json$")

LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

_EVAL_PREAMBLE = (
    "You are an expert evaluator. Given the INPUT REASONING and the GENERATED QA PAIRS below, "
    "determine how well the outputted QA pairs align with the input reasoning on a 5-point Likert score to 1dp.\n\n"
    "Use any score in 0.1 increments from 1.0 to 5.0 when needed; do not default to whole-number scores.\n\n"
    "Use this scale:\n"
    "  1 — Poor: QA pairs are largely unrelated or misrepresent the input reasoning.\n"
    "  2 — Below Average: Some connection, but significant gaps or inaccuracies.\n"
    "  3 — Average: Reasonable alignment, but missing key aspects of the reasoning.\n"
    "  4 — Good: Strong alignment with only minor gaps or imprecisions.\n"
    "  5 — Excellent: QA pairs fully and accurately capture the input reasoning.\n\n"
    "If a change summary is provided, begin with a brief **Change Summary** note summarizing "
    "what changed and whether the change is substantive; explicitly say if there was no meaningful "
    "change.\n"
    "Return only a raw JSON array with exactly one object per QA pair, in order. "
    "Each object must contain exactly two keys: "
    '`"alignment_comment"` (string) and `"likert_score"` (number). '
    "Each likert_score must be on the 5-point scale to 1dp. "
    "Because the response must be valid JSON, escape every backslash inside string values. "
    r"For example, write `\\(` not `\(`, and `\\frac` not `\frac`. "
    "Do not include any extra keys, markdown, or overall summary entry. Be concise.\n\n---\n\n"
)

_REASONING_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh")
_DEFAULT_CLAUDE_REASONING_EFFORT = "medium"
_ALIGNMENT_PASS_THRESHOLD = 4.5
_MAX_AUTO_IMPROVEMENT_ROUNDS = 5
_QA_RUBRIC_PATH = _SRC / "qa" / "qa_rubric.md"

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
    approved_label = "files approved" if mode in PIPELINE_MODES else "items approved"
    parts = [f"**{n}** {approved_label}"]
    if mode in PIPELINE_MODES:
        records = _PIPELINE_RECORDS.get(mode, [])
        if records:
            if prompt_index >= len(records):
                parts.append(f"{mode} dataset complete")
            else:
                record = records[prompt_index]
                parts.append(
                    f"File **{record['file_position']} / {record['dataset_file_total']}** | "
                    f"{record['file_name']} | **{record['example_count']}** reasoning examples"
                )
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def _paper_data_files(mode):
    prefix = mode.lower()
    paper_data_dir = _ROOT / "data" / "paper_data"
    indexed_files = []
    for path in paper_data_dir.glob(f"{prefix}_*.json"):
        match = _PAPER_DATA_FILE_RE.match(path.name)
        if not match or match.group(1) != prefix:
            continue
        indexed_files.append((int(match.group(2)), path))
    indexed_files.sort(key=lambda item: item[0])
    return indexed_files


def _reasoning_sort_key(key):
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else sys.maxsize


def _format_deeper_reasoning_entry(entry):
    if not isinstance(entry, dict):
        return str(entry or "").strip()

    def _first_nonempty(*values):
        for value in values:
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
            elif value:
                return value
        return ""

    def _format_structured_lines(value, *, ordered=False):
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            lines = []
            for idx, item in enumerate(value, start=1):
                formatted = _format_structured_lines(item)
                if not formatted:
                    continue
                prefix = f"{idx}. " if ordered else "- "
                item_lines = formatted.splitlines()
                lines.append(prefix + item_lines[0])
                lines.extend(f"   {line}" for line in item_lines[1:])
            return "\n".join(lines).strip()

        if isinstance(value, dict):
            lines = []
            for key, item in value.items():
                formatted = _format_structured_lines(item)
                if not formatted:
                    continue
                label = str(key).replace("_", " ").strip()
                item_lines = formatted.splitlines()
                if len(item_lines) == 1:
                    lines.append(f"- {label}: {item_lines[0]}")
                else:
                    lines.append(f"- {label}:")
                    lines.extend(f"  {line}" for line in item_lines)
            return "\n".join(lines).strip()

        return str(value or "").strip()

    sections = []
    title = (entry.get("title") or "").strip()
    if title:
        sections.append(f"**{title}**")

    theorem = _first_nonempty(entry.get("theorem_TD"), entry.get("theorem_statement"))
    if theorem:
        sections.append("### Theorem\n\n" + theorem)

    formal_statement = _format_structured_lines(entry.get("formal_statement"))
    if formal_statement:
        sections.append("### Formal Statement\n\n" + formal_statement)

    premises = _format_structured_lines(entry.get("premises_GammaD") or entry.get("premises"))
    if premises:
        sections.append("### Premises\n\n" + premises)

    derivation = _format_structured_lines(entry.get("derivation") or entry.get("proof_sketch"), ordered=True)
    if derivation:
        sections.append("### Derivation\n\n" + derivation)

    final_result = (entry.get("final_result") or "").strip()
    if final_result:
        sections.append("### Final Result\n\n" + final_result)

    goal_alignment = (entry.get("goal_alignment") or "").strip()
    if goal_alignment:
        sections.append("### Goal Alignment\n\n" + goal_alignment)

    scientific_payoff = _first_nonempty(
        entry.get("scientific_payoff"),
        (entry.get("high_value_audit") or {}).get("scientific_payoff"),
    )
    if scientific_payoff:
        sections.append("### Scientific Payoff\n\n" + scientific_payoff)

    return "\n\n".join(section for section in sections if section).strip()


def _build_batch_generation_prompt(examples):
    count = len(examples)
    header = [
        f"Convert the following {count} reasoning examples into exactly {count} question-answer pairs.",
        "Return a valid JSON array whose top-level value has exactly "
        f"{count} objects, in the same order as the reasoning examples.",
        'Each object must have exactly two string keys: "question" and "answer".',
        "Each question must be fully self-contained and faithful to its source reasoning.",
        "Each answer must show the reasoning clearly and use standard mathematical or scientific notation.",
        "Preserve LaTeX math in the string values.",
        "Because the response must be valid JSON, escape every backslash inside string values.",
        r'For example, write `\\(` not `\(`, and `\\frac` not `\frac`.',
        "Return only the raw JSON array. Do not wrap it in code fences.",
    ]

    body = []
    for idx, example in enumerate(examples, start=1):
        example_id = example.get("item_id", f"reasoning_{idx}")
        reasoning_text = example.get("reasoning_text", "").strip()
        body.append(f"REASONING EXAMPLE {idx} ({example_id}):\n{reasoning_text}")

    return "\n\n".join(header + body).strip()


def _format_file_display(record):
    if not record:
        return "*Waiting to start…*"

    header_parts = [
        f"### {record['mode']} Input",
        f"**Source File:** `{record['file_name']}`",
        f"**File Position:** {record['file_position']} / {record['dataset_file_total']}",
        f"**Reasoning Examples:** {record['example_count']}",
    ]
    if record.get("paper_title"):
        header_parts.append(f"**Paper:** {record['paper_title']}")

    sections = []
    for idx, example in enumerate(record.get("examples", []), start=1):
        sections.append(
            f"#### Example {idx}: `{example['item_id']}`\n\n{example['reasoning_text']}"
        )

    header = "\n\n".join(header_parts)
    body = "\n\n---\n\n".join(sections) if sections else "*No reasoning examples found.*"
    return header + "\n\n---\n\n" + body


def _load_pipeline_records():
    datasets = {mode: [] for mode in PIPELINE_MODES}

    for mode in PIPELINE_MODES:
        prefix = mode.lower()
        records = []
        indexed_files = _paper_data_files(mode)
        total_files = len(indexed_files)
        for file_position, (file_number, path) in enumerate(indexed_files, start=1):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            paper_title = data.get("paper_title", "")
            examples = []

            if prefix == "core":
                reasoning_keys = sorted(
                    [
                        key for key, value in data.items()
                        if key.startswith("core_reasoning_") and isinstance(value, str) and value.strip()
                    ],
                    key=_reasoning_sort_key,
                )
                for key in reasoning_keys:
                    examples.append({
                        "item_id": key,
                        "reasoning_text": data[key].strip(),
                    })
            else:
                reasoning_keys = sorted(
                    [
                        key for key, value in data.items()
                        if key.startswith("deeper_reasoning_") and isinstance(value, str) and value.strip()
                    ],
                    key=_reasoning_sort_key,
                )
                for key in reasoning_keys:
                    examples.append({
                        "item_id": key,
                        "reasoning_text": data[key].strip(),
                    })

            if not examples:
                continue

            record = {
                "mode": mode,
                "file_number": file_number,
                "file_position": file_position,
                "dataset_file_total": total_files,
                "file_name": path.name,
                "file_path": str(path),
                "paper_title": paper_title,
                "example_count": len(examples),
                "examples": examples,
            }
            record["prompt_text"] = _build_batch_generation_prompt(examples)
            record["display_text"] = _format_file_display(record)
            records.append(record)

        datasets[mode] = records

    return datasets


def _get_pipeline_record(mode, record_index):
    records = _PIPELINE_RECORDS.get(mode, [])
    if 0 <= record_index < len(records):
        return records[record_index]
    return None


def _dataset_mode_info(mode):
    records = _PIPELINE_RECORDS.get(mode, [])
    total_examples = sum(record.get("example_count", 0) for record in records)
    return {
        "record_count": len(records),
        "file_count": len(records),
        "example_count": total_examples,
    }


def _format_record_display(record):
    if not record:
        return "*Waiting to start…*"
    return record.get("display_text", "*Waiting to start…*")


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


def _temp_qa_path(record):
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record["file_name"]).stem
    return app_data_dir / f"{stem}_qa_temp.json"


def _temp_alignment_path(record):
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record["file_name"]).stem
    return app_data_dir / f"{stem}_alignment_temp.json"


def _load_qa_rubric_text():
    try:
        return _QA_RUBRIC_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_initial_reasoning_json_text(record):
    if not record:
        return ""
    try:
        return Path(record["file_path"]).read_text(encoding="utf-8").strip()
    except (KeyError, OSError):
        return ""


def _delete_temp_qa_file(record):
    if not record:
        return
    _temp_qa_path(record).unlink(missing_ok=True)
    _temp_alignment_path(record).unlink(missing_ok=True)


def _delete_all_temp_qa_files():
    app_data_dir = _SRC / "app_data"
    if not app_data_dir.exists():
        return
    for path in app_data_dir.glob("*_qa_temp.json"):
        path.unlink(missing_ok=True)
    for path in app_data_dir.glob("*_alignment_temp.json"):
        path.unlink(missing_ok=True)


def _batch_to_json_text(batch):
    return json.dumps(batch, indent=2, ensure_ascii=False)


def _save_temp_qa_batch(record, batch):
    path = _temp_qa_path(record)
    json_text = _batch_to_json_text(batch)
    path.write_text(json_text, encoding="utf-8")
    return path, json_text


def _load_temp_qa_batch(record):
    if not record:
        return None
    path = _temp_qa_path(record)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, list) else None


def _load_temp_qa_json_text(record):
    if not record:
        return ""
    path = _temp_qa_path(record)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _save_temp_alignment_entries(record, entries):
    path = _temp_alignment_path(record)
    json_text = json.dumps(entries, indent=2, ensure_ascii=False)
    path.write_text(json_text, encoding="utf-8")
    return path, json_text


def _load_temp_alignment_entries(record):
    if not record:
        return None
    path = _temp_alignment_path(record)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, list) else None


def _load_temp_alignment_json_text(record):
    if not record:
        return ""
    path = _temp_alignment_path(record)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _render_batch_markdown(batch):
    if not isinstance(batch, list) or not batch:
        return "*Waiting…*"

    sections = []
    for idx, pair in enumerate(batch, start=1):
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        sections.append(
            f"### QA Pair {idx}\n\n"
            f"**Question**\n\n{question}\n\n"
            f"**Answer**\n\n{answer}"
        )
    return "\n\n---\n\n".join(sections)


def _render_alignment_markdown(entries):
    if not isinstance(entries, list) or not entries:
        return ""

    sections = []
    change_summary = ""
    if isinstance(entries[0], dict):
        change_summary = str(
            entries[0].get("change_summary", "") or entries[0].get("patch_difference", "")
        ).strip()
    if change_summary:
        sections.append(f"### Change Summary\n\n{change_summary}")

    for idx, entry in enumerate(entries, start=1):
        comment = str(entry.get("alignment_comment", "")).strip()
        score = entry.get("likert_score")
        score_text = f"{float(score):.1f}/5.0" if isinstance(score, (int, float)) else str(score).strip()
        sections.append(
            f"### QA Pair {idx}\n\n"
            f"**Likert Score**\n\n{score_text}\n\n"
            f"**Alignment Comment**\n\n{comment}"
        )
    return "\n\n---\n\n".join(sections)


def _validate_qa_batch(batch, expected_count=None):
    if not isinstance(batch, list) or not batch:
        return False, "Empty or invalid JSON batch"

    if expected_count is not None and len(batch) != expected_count:
        return False, f"Expected {expected_count} QA pairs, got {len(batch)}"

    for idx, pair in enumerate(batch, start=1):
        if not isinstance(pair, dict):
            return False, f"Pair {idx} is not a dict"
        if set(pair.keys()) != {"question", "answer"}:
            return False, f"Pair {idx} must contain exactly question/answer keys"
        if not isinstance(pair["question"], str) or not isinstance(pair["answer"], str):
            return False, f"Pair {idx} has non-string question/answer values"
        if not pair["question"].strip() or not pair["answer"].strip():
            return False, f"Pair {idx} has an empty question or answer"
    return True, f"{len(batch)} QA pairs validated"


def _validate_alignment_entries(entries, expected_count=None):
    if not isinstance(entries, list) or not entries:
        return False, "Empty or invalid alignment JSON batch"

    if expected_count is not None and len(entries) != expected_count:
        return False, f"Expected {expected_count} alignment entries, got {len(entries)}"

    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            return False, f"Alignment entry {idx} is not a dict"
        if set(entry.keys()) != {"alignment_comment", "likert_score"}:
            return False, f"Alignment entry {idx} must contain exactly alignment_comment/likert_score keys"
        if not isinstance(entry["alignment_comment"], str) or not entry["alignment_comment"].strip():
            return False, f"Alignment entry {idx} has an empty alignment_comment"
        score = entry["likert_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return False, f"Alignment entry {idx} has a non-numeric likert_score"
        if not (1.0 <= float(score) <= 5.0):
            return False, f"Alignment entry {idx} likert_score must be between 1.0 and 5.0"
        if round(float(score), 1) != float(score):
            return False, f"Alignment entry {idx} likert_score must be rounded to 1 decimal place"
    return True, f"{len(entries)} alignment entries validated"


def _attach_change_summary(entries, change_summary):
    return [
        {
            "alignment_comment": entry["alignment_comment"],
            "likert_score": float(entry["likert_score"]),
            "change_summary": change_summary,
        }
        for entry in entries
    ]


def _is_valid_json_unicode_escape(text, slash_index):
    if slash_index + 5 >= len(text) or text[slash_index + 1] != "u":
        return False
    digits = text[slash_index + 2:slash_index + 6]
    return all(ch in "0123456789abcdefABCDEF" for ch in digits)


def _is_probable_latex_command(text, slash_index):
    index = slash_index + 1
    while index < len(text) and text[index].isalpha():
        index += 1
    command = text[slash_index + 1:index]
    return command in _LATEX_COMMANDS_WITH_JSON_ESCAPE_PREFIX


def _repair_invalid_json_string_escapes(text):
    repaired = []
    in_string = False
    index = 0

    while index < len(text):
        ch = text[index]

        if not in_string:
            repaired.append(ch)
            if ch == '"':
                in_string = True
            index += 1
            continue

        if ch == '"':
            repaired.append(ch)
            in_string = False
            index += 1
            continue

        if ch != "\\":
            repaired.append(ch)
            index += 1
            continue

        if index + 1 >= len(text):
            repaired.append("\\\\")
            index += 1
            continue

        next_ch = text[index + 1]
        if next_ch in _VALID_JSON_STRING_ESCAPES or _is_valid_json_unicode_escape(text, index):
            if _is_probable_latex_command(text, index):
                repaired.append("\\\\")
                repaired.append(next_ch)
                index += 2
                continue
            repaired.append("\\")
            repaired.append(next_ch)
        else:
            repaired.append("\\\\")
            repaired.append(next_ch)
        index += 2

    return "".join(repaired)


def _parse_generated_qa_json(raw_text, expected_count=None):
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    if not cleaned:
        return None, "", "Model returned empty output"

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_invalid_json_string_escapes(cleaned)
        if repaired != cleaned:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as repaired_exc:
                exc = repaired_exc
            else:
                ok, message = _validate_qa_batch(parsed, expected_count=expected_count)
                if not ok:
                    return None, repaired, message
                return parsed, _batch_to_json_text(parsed), message
        return None, cleaned, (
            f"Model output was not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        )

    ok, message = _validate_qa_batch(parsed, expected_count=expected_count)
    if not ok:
        return None, cleaned, message
    return parsed, _batch_to_json_text(parsed), message


def _parse_alignment_json(raw_text, expected_count=None):
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    if not cleaned:
        return None, "", "Model returned empty alignment output"

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_invalid_json_string_escapes(cleaned)
        if repaired != cleaned:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as repaired_exc:
                exc = repaired_exc
            else:
                ok, message = _validate_alignment_entries(parsed, expected_count=expected_count)
                if not ok:
                    return None, repaired, message
                return parsed, json.dumps(parsed, indent=2, ensure_ascii=False), message
        return None, cleaned, (
            f"Model output was not valid alignment JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        )

    ok, message = _validate_alignment_entries(parsed, expected_count=expected_count)
    if not ok:
        return None, cleaned, message
    return parsed, json.dumps(parsed, indent=2, ensure_ascii=False), message


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


def _pipeline_progress_path():
    return _SRC / "app_data" / "pipeline_progress.json"


def _load_pipeline_progress():
    path = _pipeline_progress_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    progress = {}
    for mode in PIPELINE_MODES:
        value = data.get(mode)
        if isinstance(value, int) and value >= 0:
            progress[mode] = value
    return progress


def _save_pipeline_progress(progress):
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        mode: max(0, int(progress.get(mode, 0)))
        for mode in PIPELINE_MODES
        if isinstance(progress.get(mode), int)
    }
    _pipeline_progress_path().write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _set_pipeline_progress(mode, record_index):
    if mode not in PIPELINE_MODES:
        return
    progress = _load_pipeline_progress()
    progress[mode] = max(0, int(record_index))
    _save_pipeline_progress(progress)


def _clear_pipeline_progress():
    _pipeline_progress_path().unlink(missing_ok=True)


def _resume_record_index(mode, approved_pairs=None):
    if mode not in PIPELINE_MODES:
        return 0

    record_count = len(_PIPELINE_RECORDS.get(mode, []))
    saved_progress = _load_pipeline_progress()
    if mode in saved_progress:
        return min(saved_progress[mode], record_count)

    approved_count = len(approved_pairs) if isinstance(approved_pairs, list) else 0
    return min(approved_count, record_count)


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


def _build_change_summary(previous_text: str | None, current_text: str) -> str:
    if previous_text is None:
        return ""

    previous_text = previous_text or ""
    current_text = current_text or ""
    if previous_text == current_text:
        return "No meaningful changes."

    try:
        previous_batch = json.loads(previous_text)
        current_batch = json.loads(current_text)
    except json.JSONDecodeError:
        previous_batch = None
        current_batch = None

    if isinstance(previous_batch, list) and isinstance(current_batch, list):
        summary_parts = []
        if len(previous_batch) != len(current_batch):
            summary_parts.append(
                f"QA pair count changed from {len(previous_batch)} to {len(current_batch)}."
            )

        pair_changes = []
        for idx, (previous_pair, current_pair) in enumerate(zip(previous_batch, current_batch), start=1):
            if not isinstance(previous_pair, dict) or not isinstance(current_pair, dict):
                pair_changes.append(f"QA pair {idx}: structure changed.")
                continue

            changed_fields = []
            if previous_pair.get("question") != current_pair.get("question"):
                changed_fields.append("question")
            if previous_pair.get("answer") != current_pair.get("answer"):
                changed_fields.append("answer")
            if changed_fields:
                pair_changes.append(
                    f"QA pair {idx}: revised {' and '.join(changed_fields)}."
                )

        if pair_changes:
            summary_parts.append(f"Revised {len(pair_changes)} QA pair(s).")
            summary_parts.extend(pair_changes[:3])
            if len(pair_changes) > 3:
                summary_parts.append(f"... and {len(pair_changes) - 3} more revised pair(s).")

        if len(current_batch) > len(previous_batch):
            summary_parts.append(
                f"Added {len(current_batch) - len(previous_batch)} new QA pair(s)."
            )
        elif len(previous_batch) > len(current_batch):
            summary_parts.append(
                f"Removed {len(previous_batch) - len(current_batch)} QA pair(s)."
            )

        if summary_parts:
            return " ".join(summary_parts)
        return "Minor text-only edits with no structural change."

    previous_line_count = len(previous_text.splitlines())
    current_line_count = len(current_text.splitlines())
    return (
        "Text updated. "
        f"Previous version had {previous_line_count} line(s); current version has {current_line_count} line(s)."
    )


def _build_refinement_prompt(input_text, current_output_json, user_instruction, evaluation_feedback=None):
    sections = [
        "Revise the CURRENT TEMP QA JSON so it aligns better with the INPUT REASONING.",
        "Keep the task domain and content grounded in the input reasoning.\n"
        "Make every generated question fully self-contained; do not rely on previous questions, "
        "previous results, or unstated context.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        'Return only a revised raw JSON array with exactly two string keys per object: "question" and "answer".',
        "INPUT REASONING:\n" + (input_text or ""),
        "CURRENT TEMP QA JSON:\n" + (current_output_json or ""),
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
        "The previous temp QA JSON was rejected. Generate a new version that better follows the "
        "input reasoning.",
        "Keep the task domain and content grounded in the input reasoning.\n"
        "Make every generated question fully self-contained; do not rely on previous questions, "
        "previous results, or unstated context.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        'Return only a regenerated raw JSON array with exactly two string keys per object: "question" and "answer".',
        "INPUT REASONING:\n" + (input_text or ""),
        "PREVIOUS TEMP QA JSON:\n" + (rejected_output or ""),
    ]

    normalized_feedback = _normalize_review_feedback(
        evaluation_feedback,
        EVAL_WAITING,
        EVAL_RUNNING,
    )
    if normalized_feedback:
        sections.append("LATEST ALIGNMENT EVALUATION:\n" + normalized_feedback)

    return "\n\n".join(sections)


def _build_auto_improvement_prompt(record, current_output_json, current_alignment_json):
    temp_json_name = _temp_qa_path(record).name if record else "temp_qa.json"
    reasoning_name = Path(record["file_path"]).name if record and record.get("file_path") else "initial_reasoning.json"
    rubric_text = _load_qa_rubric_text()
    initial_reasoning_json = _load_initial_reasoning_json_text(record)

    sections = [
        f"Improve ```{temp_json_name}``` alignment with ```qa_rubric.md``` and ```{reasoning_name}```.",
        "Revise the current temp QA JSON so it better satisfies the rubric and remains faithful to the initial reasoning JSON.",
        "Keep the task domain and content grounded in the source reasoning JSON.\n"
        "Make every generated question fully self-contained; do not rely on previous questions, "
        "previous results, or unstated context.\n"
        "Do not switch to coding, UI, CSS, or repository-editing tasks.\n"
        'Return only a revised raw JSON array with exactly two string keys per object: "question" and "answer".',
        "Preserve LaTeX math in the string values.",
        "Because the response must be valid JSON, escape every backslash inside string values.",
        r'For example, write `\\(` not `\(`, and `\\frac` not `\frac`.',
    ]

    if rubric_text:
        sections.append("QA RUBRIC (qa_rubric.md):\n" + rubric_text)
    if initial_reasoning_json:
        sections.append(f"INITIAL REASONING JSON ({reasoning_name}):\n" + initial_reasoning_json)
    sections.append(f"CURRENT TEMP QA JSON ({temp_json_name}):\n" + (current_output_json or ""))
    if current_alignment_json:
        sections.append("CURRENT TEMP ALIGNMENT JSON:\n" + current_alignment_json)

    return "\n\n".join(sections)


def _build_alignment_prompt(source_reasoning_text, current_output_json, previous_output_json=None):
    rubric_text = _load_qa_rubric_text()

    sections = [_EVAL_PREAMBLE]
    if rubric_text:
        sections.append("QA RUBRIC (qa_rubric.md):\n" + rubric_text)
    if source_reasoning_text:
        sections.append("SOURCE REASONING PANEL TEXT:\n" + source_reasoning_text)
    sections.append("CURRENT TEMP QA JSON:\n" + (current_output_json or ""))

    change_summary = _build_change_summary(previous_output_json, current_output_json)
    if change_summary:
        sections.append("CHANGE SUMMARY FROM PREVIOUS OUTPUT:\n" + change_summary)

    return "\n\n---\n\n".join(sections)


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


def _eval_stream(provider, source_reasoning_text, output_text, model, reasoning_effort, previous_output=None, session_id=None):
    """Yield (text_so_far, session_id) as the selected provider streams the evaluation."""
    eval_prompt = _build_alignment_prompt(
        source_reasoning_text,
        output_text,
        previous_output_json=previous_output,
    )
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
    "as a JSON array.\n"
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
    "JSON array.\n"
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

    Checks: schema, count, verbatim substring, ordering, prompt contamination.
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
                return False, f"Question {i+1} matches prompt content"
            if pair["answer"] in prompt_text:
                return False, f"Answer {i+1} matches prompt content"

    return True, f"{len(batch)} pairs extracted and verified"


def _extract_qa_structured(provider, session_id, model, reasoning_effort,
                           raw_output, current_input):
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

    prompt_text = current_input

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
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def _alignment_checkbox_update(entries):
    if not isinstance(entries, list) or not entries:
        return gr.update(value=False)
    scores = []
    for entry in entries:
        if not isinstance(entry, dict):
            return gr.update(value=False)
        score = entry.get("likert_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return gr.update(value=False)
        scores.append(float(score))
    return gr.update(value=all(score >= _ALIGNMENT_PASS_THRESHOLD for score in scores))


def _alignment_entries_pass(entries):
    if not isinstance(entries, list) or not entries:
        return False
    return all(
        isinstance(entry, dict)
        and isinstance(entry.get("likert_score"), (int, float))
        and not isinstance(entry.get("likert_score"), bool)
        and float(entry["likert_score"]) >= _ALIGNMENT_PASS_THRESHOLD
        for entry in entries
    )


def _approve_current_temp_batch(
    *,
    output_panel,
    provider,
    author_session_id,
    eval_session_id,
    model,
    reasoning_effort,
    record_index,
    approved_pairs,
    current_input,
    mode,
    alignment_loop_count,
):
    record = _get_pipeline_record(mode, record_index) if mode in PIPELINE_MODES else None
    temp_batch = _load_temp_qa_batch(record)
    if temp_batch is None:
        yield (
            gr.update(),
            gr.update(),
            "**Approval Failed**\n\nNo valid temp QA JSON exists for the current input file.",
            False,
            alignment_loop_count,
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, True),
            _status(record_index, approved_pairs, mode, "NO TEMP QA JSON"),
        )
        return

    ok, validate_msg = _validate_qa_batch(
        temp_batch,
        expected_count=(record.get("example_count") if record else None),
    )
    if not ok:
        yield (
            gr.update(),
            gr.update(),
            f"**Approval Failed**\n\n{validate_msg}",
            False,
            alignment_loop_count,
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, True),
            _status(record_index, approved_pairs, mode, "TEMP QA JSON INVALID"),
        )
        return

    yield (
        gr.update(),
        gr.update(),
        gr.update(),
        False,
        alignment_loop_count,
        author_session_id,
        eval_session_id,
        record_index,
        approved_pairs,
        current_input,
        False,
        *_controls(False, False),
        _status(record_index, approved_pairs, mode, "APPENDING TEMP QA JSON…"),
    )
    entry = {"batch": temp_batch}
    if mode in PIPELINE_MODES and record:
        entry.update({
            "mode": mode,
            "source_file": record["file_name"],
            "source_path": record["file_path"],
            "source_item_ids": [example["item_id"] for example in record.get("examples", [])],
            "example_count": record.get("example_count", 0),
            "paper_title": record.get("paper_title", ""),
        })
    approved_pairs = approved_pairs + [entry]
    _autosave_datasets(approved_pairs)
    _delete_temp_qa_file(record)

    verified_status = f"VERIFIED & SAVED — {validate_msg}"
    if mode in PIPELINE_MODES:
        next_record_index = record_index + 1
        _set_pipeline_progress(mode, next_record_index)
        yield (
            gr.update(),
            gr.update(),
            f"**Approval Saved:** {validate_msg}",
            False,
            0,
            None,
            None,
            next_record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, False),
            _status(next_record_index, approved_pairs, mode, verified_status),
        )
        yield from send_pipeline_record(
            provider, None, None, model, reasoning_effort,
            next_record_index, approved_pairs, False, mode,
        )
        return

    yield (
        "*Paste your next reasoning input below.*",
        "*Waiting…*",
        f"**Approval Saved:** {validate_msg}",
        False,
        0,
        None,
        None,
        record_index,
        approved_pairs,
        "",
        False,
        *_controls(False, False),
        _status(record_index, approved_pairs, mode, verified_status),
    )


EVAL_WAITING = "*Alignment evaluation will appear here after generation completes…*"
EVAL_RUNNING = "*Evaluating alignment…*"

_GPT_MODEL_METADATA = _load_gpt_model_metadata()
_CODEX_DEFAULT_REASONING_EFFORT = _load_codex_reasoning_effort()

PROVIDER_MODELS = {
    "Claude": ["sonnet", "opus", "haiku"],
    "GPT": _load_gpt_models(),
}

_PIPELINE_RECORDS = _load_pipeline_records()


# ---------------------------------------------------------------------------
# Outputs tuple order (17 elements):
#   input_panel, output_panel, eval_panel, alignment_checkbox, alignment_loop_counter,
#   author_session_state, eval_session_state, record_index_state,
#   approved_state, current_input_state, edit_mode_state,
#   approve_btn, edit_btn, reject_btn, back_btn, cancel_edit_btn, status_md
# ---------------------------------------------------------------------------

def _message_box_update(mode):
    return gr.update(
        value="",
        placeholder=f"{mode} mode: use Start {mode} Mode  ·  Automatic improvement runs below the alignment threshold.",
    )


def _start_button_update(mode):
    if mode in PIPELINE_MODES:
        return gr.update(value=f"Start {mode} Mode", visible=True)
    return gr.update(value="Start Selected Mode", visible=False)


def _reset_ui_state(mode, record_index=None):
    saved_approved = _load_saved_progress()
    if record_index is None:
        record_index = _resume_record_index(mode, saved_approved)
    if mode in PIPELINE_MODES:
        dataset_info = _dataset_mode_info(mode)
        if dataset_info["record_count"]:
            input_text = "*Waiting to start…*"
            status_extra = (
                f"{dataset_info['file_count']} files | {dataset_info['example_count']} reasoning examples"
            )
            if saved_approved:
                status_extra = "LOADED FROM DISK | " + status_extra
        else:
            input_text = f"*No {mode.lower()} input files found in `data/paper_data`.*"
            status_extra = "NO INPUT FILES FOUND"
    else:
        input_text = "*Waiting to start…*"
        status_extra = "LOADED FROM DISK" if saved_approved else ""

    status = _status(
        record_index,
        saved_approved,
        mode,
        status_extra,
    )
    return (
        input_text,
        "*Waiting…*",
        "",
        False,
        0,
        None, None, record_index, saved_approved, "", False,
        *_controls(False, False), status,
    )


def _run_alignment_cycle(
    *,
    provider,
    model,
    reasoning_effort,
    record,
    input_display,
    current_input,
    approved_pairs,
    record_index,
    mode,
    author_session_id,
    eval_session_id,
    current_output_json,
    rendered_output,
    status_text,
):
    previous_output_json = None
    current_author_sid = author_session_id
    current_eval_sid = eval_session_id
    status_prefix = status_text
    alignment_loop_count = 0

    for round_index in range(_MAX_AUTO_IMPROVEMENT_ROUNDS + 1):
        yield (
            input_display,
            rendered_output,
            EVAL_RUNNING,
            False,
            alignment_loop_count,
            current_author_sid,
            current_eval_sid,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, True),
            status_prefix,
        )

        raw_eval_text = ""
        last_eval_sid = current_eval_sid
        for eval_text, sid in _eval_stream(
            provider,
            current_input,
            current_output_json,
            model,
            reasoning_effort,
            previous_output=previous_output_json,
            session_id=current_eval_sid,
        ):
            raw_eval_text = eval_text
            last_eval_sid = sid
            yield (
                input_display,
                rendered_output,
                EVAL_RUNNING,
                False,
                alignment_loop_count,
                current_author_sid,
                last_eval_sid,
                record_index,
                approved_pairs,
                current_input,
                False,
                *_controls(False, True),
                status_prefix,
            )

        current_eval_sid = last_eval_sid
        alignment_entries, _, alignment_msg = _parse_alignment_json(
            raw_eval_text,
            expected_count=(record.get("example_count") if record else None),
        )
        if alignment_entries is None:
            yield (
                input_display,
                rendered_output,
                f"**Temp Alignment JSON Error**\n\n{alignment_msg}",
                False,
                alignment_loop_count,
                current_author_sid,
                current_eval_sid,
                record_index,
                approved_pairs,
                current_input,
                False,
                *_controls(False, True),
                _status(record_index, approved_pairs, mode, "TEMP ALIGNMENT JSON INVALID"),
            )
            return

        change_summary = "None" if previous_output_json is None else _build_change_summary(
            previous_output_json,
            current_output_json,
        )
        saved_alignment_entries = _attach_change_summary(alignment_entries, change_summary or "None")
        _, saved_alignment_json = _save_temp_alignment_entries(record, saved_alignment_entries)
        rendered_eval = _render_alignment_markdown(saved_alignment_entries)
        alignment_ok = _alignment_entries_pass(saved_alignment_entries)
        final_status = _status(record_index, approved_pairs, mode, alignment_msg)
        yield (
            input_display,
            rendered_output,
            rendered_eval,
            _alignment_checkbox_update(saved_alignment_entries),
            alignment_loop_count,
            current_author_sid,
            current_eval_sid,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, True),
            final_status,
        )

        if alignment_ok and alignment_loop_count >= 1:
            yield from _approve_current_temp_batch(
                output_panel=rendered_output,
                provider=provider,
                author_session_id=current_author_sid,
                eval_session_id=current_eval_sid,
                model=model,
                reasoning_effort=reasoning_effort,
                record_index=record_index,
                approved_pairs=approved_pairs,
                current_input=current_input,
                mode=mode,
                alignment_loop_count=alignment_loop_count,
            )
            return

        if round_index >= _MAX_AUTO_IMPROVEMENT_ROUNDS:
            yield (
                input_display,
                rendered_output,
                rendered_eval,
                _alignment_checkbox_update(saved_alignment_entries),
                alignment_loop_count,
                current_author_sid,
                current_eval_sid,
                record_index,
                approved_pairs,
                current_input,
                False,
                *_controls(False, True),
                _status(
                    record_index,
                    approved_pairs,
                    mode,
                    f"{alignment_msg} | AUTO-IMPROVEMENT LIMIT REACHED",
                ),
            )
            return

        improvement_prompt = _build_auto_improvement_prompt(record, current_output_json, saved_alignment_json)
        alignment_loop_count += 1
        status_prefix = _status(
            record_index,
            approved_pairs,
            mode,
            f"{alignment_msg} | AUTO-IMPROVING ROUND {round_index + 1}",
        )
        yield (
            input_display,
            rendered_output,
            rendered_eval,
            False,
            alignment_loop_count,
            current_author_sid,
            current_eval_sid,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, False),
            status_prefix,
        )

        raw_text = ""
        last_author_sid = current_author_sid
        for text, sid in _stream(provider, improvement_prompt, current_author_sid, model, reasoning_effort):
            raw_text = text
            last_author_sid = sid
            yield (
                input_display,
                "*Auto-improving temp QA JSON…*",
                rendered_eval,
                False,
                alignment_loop_count,
                sid,
                current_eval_sid,
                record_index,
                approved_pairs,
                current_input,
                False,
                *_controls(False, False),
                status_prefix,
            )

        batch, batch_json_text, parse_msg = _parse_generated_qa_json(
            raw_text,
            expected_count=(record.get("example_count") if record else None),
        )
        if batch is None:
            yield (
                input_display,
                f"```json\n{batch_json_text or raw_text}\n```",
                rendered_eval,
                False,
                alignment_loop_count,
                last_author_sid,
                current_eval_sid,
                record_index,
                approved_pairs,
                current_input,
                False,
                *_controls(False, True),
                _status(record_index, approved_pairs, mode, "TEMP JSON INVALID"),
            )
            return

        previous_output_json = current_output_json
        _, current_output_json = _save_temp_qa_batch(record, batch)
        rendered_output = _render_batch_markdown(batch)
        current_author_sid = last_author_sid


def start_selected_mode(provider, model, reasoning_effort,
                        author_session_id, eval_session_id, record_index, approved_pairs, edit_mode, mode):
    _delete_all_temp_qa_files()
    yield from send_pipeline_record(
        provider, None, None, model, reasoning_effort,
        record_index, approved_pairs, edit_mode, mode,
    )


def on_mode_change(mode):
    _delete_all_temp_qa_files()
    return (
        *_reset_ui_state(mode),
        _start_button_update(mode),
        _message_box_update(mode),
    )


def send_pipeline_record(provider, author_session_id, eval_session_id, model, reasoning_effort,
                         record_index, approved_pairs, edit_mode, mode):
    """Send the current Core/Deeper source file, stream QA, then stream evaluation."""
    records = _PIPELINE_RECORDS.get(mode, [])
    if not records:
        yield (
            f"*No {mode.lower()} input files found in `data/paper_data`.*",
            "*Waiting…*",
            "",
            False,
            0,
            None, None, record_index, approved_pairs, "", False,
            *_controls(False, False),
            _status(record_index, approved_pairs, mode, "NO INPUT FILES FOUND"),
        )
        return

    if record_index >= len(records):
        done = (
            f"All **{len(records)}** source files processed.  \n"
            f"**{len(approved_pairs)}** files approved.  \n"
            "Approved outputs have been autosaved."
        )
        yield (
            "*Pipeline complete.*",
            done,
            "",
            False,
            0,
            None, None, record_index, approved_pairs, "", False,
            *_controls(False, False),
            _status(record_index, approved_pairs, mode, "DONE | AUTOSAVED"),
        )
        return

    record = records[record_index]
    llm_message = record["prompt_text"]
    input_display = _format_record_display(record)
    st = _status(record_index, approved_pairs, mode)

    # Phase 1: stream QA generation
    yield (
        input_display,
        "*Generating…*",
        EVAL_WAITING,
        False,
        0,
        author_session_id,
        eval_session_id,
        record_index,
        approved_pairs,
        input_display,
        False,
        *_controls(False, False),
        st,
    )

    raw_text = ""
    last_author_sid = author_session_id
    for text, sid in _stream(provider, llm_message, author_session_id, model, reasoning_effort):
        raw_text = text
        last_author_sid = sid
        yield (
            input_display,
            "*Generating…*",
            EVAL_WAITING,
            False,
            0,
            sid,
            eval_session_id,
            record_index,
            approved_pairs,
            input_display,
            edit_mode,
            *_controls(edit_mode, True),
            st,
        )

    batch, batch_json_text, parse_msg = _parse_generated_qa_json(
        raw_text,
        expected_count=record.get("example_count"),
    )
    if batch is None:
        yield (
            input_display,
            f"```json\n{batch_json_text or raw_text}\n```",
            f"**Temp QA JSON Error**\n\n{parse_msg}\n\nRegenerate or revise the output so it is valid JSON.",
            False,
            0,
            last_author_sid,
            eval_session_id,
            record_index,
            approved_pairs,
            input_display,
            edit_mode,
            *_controls(edit_mode, True),
            _status(record_index, approved_pairs, mode, "TEMP JSON INVALID"),
        )
        return

    _, saved_json_text = _save_temp_qa_batch(record, batch)
    rendered_output = _render_batch_markdown(batch)

    yield from _run_alignment_cycle(
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        record=record,
        input_display=input_display,
        current_input=input_display,
        approved_pairs=approved_pairs,
        record_index=record_index,
        mode=mode,
        author_session_id=last_author_sid,
        eval_session_id=eval_session_id,
        current_output_json=saved_json_text,
        rendered_output=rendered_output,
        status_text=st,
    )


def on_approve(output_panel, provider, author_session_id, eval_session_id, model, reasoning_effort,
               record_index, approved_pairs, current_input, mode):
    """Append the current temp QA JSON batch to qa_data.json, then auto-advance."""
    yield from _approve_current_temp_batch(
        output_panel=output_panel,
        provider=provider,
        author_session_id=author_session_id,
        eval_session_id=eval_session_id,
        model=model,
        reasoning_effort=reasoning_effort,
        record_index=record_index,
        approved_pairs=approved_pairs,
        current_input=current_input,
        mode=mode,
        alignment_loop_count=0,
    )


def on_reject(input_panel, provider, author_session_id, eval_session_id, model, reasoning_effort,
              record_index, approved_pairs, current_input,
              mode, edit_mode, output_panel, eval_panel):
    """Reject, retry, then re-evaluate."""
    record = _get_pipeline_record(mode, record_index) if mode in PIPELINE_MODES else None
    current_temp_json = _load_temp_qa_json_text(record)
    if not current_temp_json:
        yield (
            input_panel,
            output_panel,
            "**Reject Failed**\n\nNo temp QA JSON is available to regenerate from.",
            False,
            0,
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            False,
            *_controls(False, True),
            _status(record_index, approved_pairs, mode, "NO TEMP QA JSON"),
        )
        return

    st = _status(record_index, approved_pairs, mode, "MANUAL AUTO-IMPROVEMENT RESTART")
    yield from _run_alignment_cycle(
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        record=record,
        input_display=input_panel,
        current_input=current_input,
        approved_pairs=approved_pairs,
        record_index=record_index,
        mode=mode,
        author_session_id=author_session_id,
        eval_session_id=eval_session_id,
        current_output_json=current_temp_json,
        rendered_output=output_panel,
        status_text=st,
    )


def on_edit(record_index, approved_pairs, mode):
    """Enter edit mode and show a way back to review controls."""
    return (
        True,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        _status(record_index, approved_pairs, mode, "EDITING — type a refinement below"),
    )


def on_back_from_edit(record_index, approved_pairs, mode):
    """Return from edit mode to the normal review controls."""
    return (
        False,
        *_controls(False, True),
        _status(record_index, approved_pairs, mode),
    )


def capture_msg(message):
    """Clear the textbox and stash the message for the next handler."""
    return "", message


def user_chat_submit(message, input_panel, output_panel, eval_panel,
                     provider, author_session_id, eval_session_id, model, reasoning_effort,
                     record_index, approved_pairs,
                     current_input, mode, edit_mode):
    """Handle typed edit refinements, then evaluate."""
    record = _get_pipeline_record(mode, record_index) if mode in PIPELINE_MODES else None
    if not message or not message.strip():
        yield (
            input_panel,
            gr.update(),
            gr.update(),
            False,
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, False),
            _status(record_index, approved_pairs, mode),
        )
        return

    if not edit_mode:
        yield (
            input_panel,
            output_panel,
            eval_panel,
            _alignment_checkbox_update(_load_temp_alignment_entries(record)),
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            _status(record_index, approved_pairs, mode, "Click Edit before sending a refinement."),
        )
        return

    st = _status(record_index, approved_pairs, mode)
    current_temp_json = _load_temp_qa_json_text(record)
    if not current_temp_json:
        yield (
            input_panel,
            output_panel,
            "**Edit Failed**\n\nNo temp QA JSON is available for refinement.",
            False,
            author_session_id,
            eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            _status(record_index, approved_pairs, mode, "NO TEMP QA JSON"),
        )
        return

    input_display = input_panel
    llm_message = _build_refinement_prompt(current_input, current_temp_json, message, eval_panel)
    stream_session_id = author_session_id
    next_eval_session_id = eval_session_id

    yield (
        input_display,
        "*Refining…*",
        EVAL_WAITING,
        False,
        stream_session_id,
        next_eval_session_id,
        record_index,
        approved_pairs,
        current_input,
        edit_mode,
        *_controls(edit_mode, False),
        st,
    )

    raw_text = ""
    last_author_sid = stream_session_id
    for text, sid in _stream(provider, llm_message, stream_session_id, model, reasoning_effort):
        raw_text = text
        last_author_sid = sid
        yield (
            input_display,
            "*Refining…*",
            EVAL_WAITING,
            False,
            sid,
            next_eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            st,
        )

    batch, batch_json_text, parse_msg = _parse_generated_qa_json(
        raw_text,
        expected_count=(record.get("example_count") if record else None),
    )
    if batch is None:
        yield (
            input_display,
            f"```json\n{batch_json_text or raw_text}\n```",
            f"**Temp QA JSON Error**\n\n{parse_msg}\n\nThe previous temp JSON has been kept unchanged.",
            False,
            last_author_sid,
            next_eval_session_id,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            _status(record_index, approved_pairs, mode, "TEMP JSON INVALID"),
        )
        return

    _, saved_json_text = _save_temp_qa_batch(record, batch)
    rendered_output = _render_batch_markdown(batch)

    # Evaluate using the temp JSON file
    yield (
        input_display,
        rendered_output,
        EVAL_RUNNING,
        False,
        last_author_sid,
        next_eval_session_id,
        record_index,
        approved_pairs,
        current_input,
        edit_mode,
        *_controls(edit_mode, True),
        st,
    )

    raw_eval_text = ""
    last_eval_sid = next_eval_session_id
    for eval_text, sid in _eval_stream(
        provider,
        current_input,
        saved_json_text,
        model,
        reasoning_effort,
        previous_output=current_temp_json,
        session_id=next_eval_session_id,
    ):
        raw_eval_text = eval_text
        last_eval_sid = sid
        yield (
            input_display,
            rendered_output,
            EVAL_RUNNING,
            False,
            last_author_sid,
            last_eval_sid,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            st,
        )

    alignment_entries, _, alignment_msg = _parse_alignment_json(
        raw_eval_text,
        expected_count=(record.get("example_count") if record else None),
    )
    if alignment_entries is None:
        yield (
            input_display,
            rendered_output,
            f"**Temp Alignment JSON Error**\n\n{alignment_msg}\n\nThe previous temp alignment JSON has been kept unchanged.",
            False,
            last_author_sid,
            last_eval_sid,
            record_index,
            approved_pairs,
            current_input,
            edit_mode,
            *_controls(edit_mode, True),
            _status(record_index, approved_pairs, mode, "TEMP ALIGNMENT JSON INVALID"),
        )
        return

    _save_temp_alignment_entries(record, alignment_entries)
    rendered_eval = _render_alignment_markdown(alignment_entries)
    yield (
        input_display,
        rendered_output,
        rendered_eval,
        _alignment_checkbox_update(alignment_entries),
        last_author_sid,
        last_eval_sid,
        record_index,
        approved_pairs,
        current_input,
        edit_mode,
        *_controls(edit_mode, True),
        _status(record_index, approved_pairs, mode, alignment_msg),
    )


def clear_session(mode):
    _delete_all_temp_qa_files()
    return (
        *_reset_ui_state(mode),
        _start_button_update(mode),
        _message_box_update(mode),
    )


def reset_saved_qa_data(mode):
    app_data_dir = _SRC / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    _delete_all_temp_qa_files()
    _clear_pipeline_progress()

    qa_path = app_data_dir / "qa_data.json"
    lean_prompt_path = app_data_dir / "lean_prompt_data.json"

    qa_path.write_text("[]\n", encoding="utf-8")
    lean_prompt_path.write_text("[]\n", encoding="utf-8")

    reset_outputs = list(_reset_ui_state(mode, 0))
    reset_outputs[-1] = _status(0, [], mode, "SAVED QA DATA RESET")
    return (*reset_outputs, _start_button_update(mode), _message_box_update(mode))


_INITIAL_APPROVED = _load_saved_progress()
_INITIAL_MODE = "Core"
_INITIAL_RECORD_INDEX = _resume_record_index(_INITIAL_MODE, _INITIAL_APPROVED)
_INITIAL_STATUS = (
    _status(_INITIAL_RECORD_INDEX, _INITIAL_APPROVED, _INITIAL_MODE, "LOADED FROM DISK")
    if _INITIAL_APPROVED
    else ""
)

def on_provider_change(provider, mode, current_effort):
    _delete_all_temp_qa_files()
    model_choices = PROVIDER_MODELS[provider]
    model = model_choices[0]
    return (
        gr.update(choices=model_choices, value=model),
        update_reasoning_effort_dropdown(provider, model, current_effort),
        *_reset_ui_state(mode),
        _start_button_update(mode),
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
    author_session_state = gr.State(None)
    eval_session_state = gr.State(None)
    record_index_state = gr.State(_INITIAL_RECORD_INDEX)
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
            ["Core", "Deeper"],
            value=_INITIAL_MODE,
            label="Mode",
            info="Load reasoning from data/paper_data in ascending file order.",
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
    # --- Side-by-side comparison ---
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("SOURCE REASONING", elem_classes=["panel-label"])
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
    alignment_checkbox = gr.Checkbox(
        value=False,
        label="All Individual Likert Scores Above 4.5",
        interactive=False,
    )
    alignment_loop_counter = gr.Number(
        value=0,
        label="Alignment Improvement Loops",
        precision=0,
        interactive=False,
    )

    # --- Review buttons ---
    with gr.Row(elem_classes=["review-row"]):
        approve_btn = gr.Button("Approve", variant="primary", visible=False, min_width=140)
        edit_btn = gr.Button("Edit", variant="secondary", visible=False, min_width=140)
        reject_btn = gr.Button("Reject", variant="stop", visible=False, min_width=140)
        back_btn = gr.Button("Back to Review", visible=False, min_width=160)

    status_md = gr.Markdown(_INITIAL_STATUS, elem_classes=["status-bar"])

    # --- Edit input ---
    msg = gr.Textbox(
        label="Message",
        placeholder="Automatic improvement is enabled when alignment scores fall below threshold.",
        autofocus=True,
        visible=False,
    )
    cancel_edit_btn = gr.Button("Back to Review", variant="secondary", visible=False)

    # --- Action buttons ---
    with gr.Row():
        start_btn = gr.Button("Start Core Mode", variant="primary")
        clear_btn = gr.Button("New Session")
        reset_saved_btn = gr.Button("Reset Saved QA Data", variant="stop")

    # --- Shared output list ---
    panel_outputs = [
        input_panel, output_panel, eval_panel, alignment_checkbox, alignment_loop_counter,
        author_session_state, eval_session_state, record_index_state,
        approved_state, current_input_state, edit_mode_state,
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
        start_selected_mode,
        inputs=[provider_dropdown, model_dropdown, reasoning_effort_dropdown,
                author_session_state, eval_session_state, record_index_state, approved_state, edit_mode_state,
                mode_radio],
        outputs=panel_outputs,
    )

    approve_btn.click(
        on_approve,
        inputs=[output_panel, provider_dropdown, author_session_state, eval_session_state,
                model_dropdown, reasoning_effort_dropdown,
                record_index_state, approved_state, current_input_state, mode_radio],
        outputs=panel_outputs,
    )

    reject_btn.click(
        on_reject,
        inputs=[input_panel, provider_dropdown, author_session_state, eval_session_state,
                model_dropdown, reasoning_effort_dropdown,
                record_index_state, approved_state, current_input_state, mode_radio, edit_mode_state,
                output_panel, eval_panel],
        outputs=panel_outputs,
    )

    clear_btn.click(clear_session, inputs=[mode_radio], outputs=[*panel_outputs, start_btn, msg])
    reset_saved_btn.click(
        reset_saved_qa_data,
        inputs=[mode_radio],
        outputs=[*panel_outputs, start_btn, msg],
    )


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
