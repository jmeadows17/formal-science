"""Validated postprocessing pipeline for structured Lean proofs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from compile_lean import compile_lean


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_STRUCTURED_PROOFS_PATH = ROOT / "app_data" / "structured_proofs.json"
DEFAULT_FORMAL_QA_OUTPUT_PATH = ROOT / "app_data" / "formal_qa_data.json"
DEFAULT_AUDIT_OUTPUT_PATH = ROOT / "app_data" / "postprocessed_batches.json"

_FINAL_BATCH_STATUSES = {"approved", "skipped"}
_THEOREM_RE = re.compile(r"^\s*theorem\s+(C\d+(?:_[^\s:(]+)?)(?=\s|[:({])")
_LABEL_RE = re.compile(r"^(C\d+)")
_DECL_LINE_PATTERNS = [
    (re.compile(r"^\s*theorem\s+([^\s:(]+)"), "theorem"),
    (re.compile(r"^\s*lemma\s+([^\s:(]+)"), "lemma"),
    (re.compile(r"^\s*def\s+([^\s:(]+)"), "def"),
    (re.compile(r"^\s*noncomputable\s+def\s+([^\s:(]+)"), "def"),
    (re.compile(r"^\s*abbrev\s+([^\s:(]+)"), "abbrev"),
]
_DOCSTRING_START_RE = re.compile(r"^\s*/-(?:!|-)?")
_ENV_LINE_PATTERNS = [
    re.compile(r"^\s*namespace\s+\S+"),
    re.compile(r"^\s*open\s+scoped\s+\S+"),
    re.compile(r"^\s*open\s+\S+"),
    re.compile(r"^\s*set_option\s+\S+"),
    re.compile(r"^\s*section(?:\s+\S+)?"),
    re.compile(r"^\s*noncomputable\s+section(?:\s+\S+)?"),
    re.compile(r"^\s*variable(?:\s|\()"),
    re.compile(r"^\s*attribute(?:\s|\[)"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int_field(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"Block field `{key}` must be an integer, got {value!r}.")
    return value


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc


def load_structured_proofs(path: Path = DEFAULT_STRUCTURED_PROOFS_PATH) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} must contain a top-level JSON list.")
    return data


def load_postprocessing_audit(path: Path = DEFAULT_AUDIT_OUTPUT_PATH) -> list[dict[str, Any]]:
    try:
        data = _load_json(path)
    except RuntimeError as exc:
        if path.exists():
            raise
        return []
    if not isinstance(data, list):
        raise RuntimeError(f"{path} must contain a top-level JSON list.")
    return data


def load_formal_qa_data(path: Path = DEFAULT_FORMAL_QA_OUTPUT_PATH) -> list[dict[str, str]]:
    try:
        data = _load_json(path)
    except RuntimeError:
        if path.exists():
            raise
        return []
    if not isinstance(data, list):
        raise RuntimeError(f"{path} must contain a top-level JSON list.")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"{path} entry {index} must be an object.")
        question = item.get("question")
        answer = item.get("answer")
        formal_answer = item.get("formal_answer")
        if not isinstance(question, str) or not isinstance(answer, str) or not isinstance(formal_answer, str):
            raise RuntimeError(
                f"{path} entry {index} must contain string `question`, `answer`, and `formal_answer` values."
            )
        normalized.append(
            {
                "question": question,
                "answer": answer,
                "formal_answer": formal_answer,
            }
        )
    return normalized


def save_postprocessing_audit(
    entries: list[dict[str, Any]],
    path: Path = DEFAULT_AUDIT_OUTPUT_PATH,
) -> None:
    _write_json(path, entries)


def _validate_batch_entry(entry: dict[str, Any], *, batch_index: int) -> tuple[list[dict[str, str]], str]:
    qa_batch = entry.get("qa_batch")
    formal_proofs = entry.get("formal_proofs")

    if not isinstance(qa_batch, list):
        raise RuntimeError(f"Batch {batch_index} must contain a list-valued `qa_batch`.")
    if not isinstance(formal_proofs, str):
        raise RuntimeError(f"Batch {batch_index} must contain a string-valued `formal_proofs`.")

    normalized_batch: list[dict[str, str]] = []
    for qa_index, qa_item in enumerate(qa_batch, start=1):
        if not isinstance(qa_item, dict):
            raise RuntimeError(
                f"Batch {batch_index} QA item {qa_index} must be an object with "
                "`question` and `answer`."
            )
        question = qa_item.get("question")
        answer = qa_item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise RuntimeError(
                f"Batch {batch_index} QA item {qa_index} must contain string-valued "
                "`question` and `answer` fields."
            )
        normalized_batch.append({"question": question, "answer": answer})

    return normalized_batch, formal_proofs


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def source_line_count(text: str) -> int:
    return len(source_lines(text))


def render_numbered_source(text: str) -> str:
    lines = source_lines(text)
    width = max(4, len(str(len(lines) or 1)))
    return "".join(f"{idx:0{width}d}| {line}" for idx, line in enumerate(lines, start=1))


def detect_target_theorems(formal_proofs: str) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_lines(formal_proofs), start=1):
        match = _THEOREM_RE.match(line)
        if not match:
            continue

        theorem_name = match.group(1)
        label_match = _LABEL_RE.match(theorem_name)
        if not label_match:
            continue

        targets.append(
            {
                "label": label_match.group(1),
                "theorem_name": theorem_name,
                "line_number": line_number,
                "line_text": line.rstrip("\n"),
            }
        )
    return targets


def _match_declaration_header(line: str) -> tuple[str, str] | None:
    for pattern, kind in _DECL_LINE_PATTERNS:
        match = pattern.match(line)
        if match:
            return kind, match.group(1)
    return None


def _block_start_kind(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped:
        return None
    if line == stripped and stripped.startswith("import "):
        return "import"
    if line == stripped and any(pattern.match(stripped) for pattern in _ENV_LINE_PATTERNS):
        return "env"
    if line == stripped and _DOCSTRING_START_RE.match(stripped):
        return "doc"
    if line == stripped and _match_declaration_header(stripped):
        return "decl"
    return None


def _merge_doc_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    pending_docs: list[dict[str, Any]] = []

    for block in blocks:
        if block["kind"] == "doc":
            pending_docs.append(block)
            continue

        if pending_docs and block["kind"] == "decl":
            start_line = pending_docs[0]["start_line"]
            text = "".join(doc["text"] for doc in pending_docs) + block["text"]
            merged.append(
                {
                    **block,
                    "decl_start_line": block["start_line"],
                    "full_start_line": start_line,
                    "full_text": text,
                    "start_line": start_line,
                    "text": text,
                }
            )
            pending_docs = []
            continue

        merged.extend(pending_docs)
        pending_docs = []
        merged.append(block)

    merged.extend(pending_docs)
    return merged


def top_level_blocks(formal_proofs: str) -> list[dict[str, Any]]:
    lines = source_lines(formal_proofs)
    starts: list[tuple[int, str]] = []

    for line_number, line in enumerate(lines, start=1):
        kind = _block_start_kind(line)
        if kind:
            starts.append((line_number, kind))

    if not starts:
        return []

    blocks: list[dict[str, Any]] = []
    for index, (start_line, kind) in enumerate(starts):
        end_line = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        text, _, _ = slice_exact_lines(formal_proofs, start_line, end_line)
        header_line = lines[start_line - 1].strip()
        decl = _match_declaration_header(header_line) if kind == "decl" else None
        decl_kind = decl[0] if decl else None
        decl_name = decl[1] if decl else None
        label = None
        if decl_kind == "theorem" and decl_name:
            label_match = _LABEL_RE.match(decl_name)
            if label_match:
                label = label_match.group(1)

        blocks.append(
            {
                "kind": kind,
                "decl_kind": decl_kind,
                "decl_name": decl_name,
                "label": label,
                "decl_start_line": start_line if kind == "decl" else None,
                "full_start_line": start_line,
                "start_line": start_line,
                "end_line": end_line,
                "full_text": text,
                "text": text,
            }
        )

    return _merge_doc_blocks(blocks)


def target_theorem_block(formal_proofs: str, theorem_name: str) -> dict[str, Any] | None:
    for block in top_level_blocks(formal_proofs):
        if block["kind"] != "decl":
            continue
        if block.get("decl_kind") != "theorem":
            continue
        if block.get("decl_name") == theorem_name:
            return block
    return None


def shared_preamble_line_count(formal_proofs: str) -> int:
    blocks = top_level_blocks(formal_proofs)
    target_blocks = [
        block
        for block in blocks
        if block["kind"] == "decl" and block.get("decl_kind") == "theorem" and block.get("label")
    ]
    if not target_blocks:
        return source_line_count(formal_proofs)

    first_target_full_start = int(target_blocks[0]["full_start_line"])
    preamble_blocks = [
        block
        for block in blocks
        if _int_field(block, "end_line") < first_target_full_start and block["kind"] in {"import", "env"}
    ]
    if not preamble_blocks:
        return 0
    return int(preamble_blocks[-1]["end_line"])


def shared_preamble_text(formal_proofs: str) -> tuple[str, list[dict[str, int | str]]]:
    preamble_end_line = shared_preamble_line_count(formal_proofs)
    if preamble_end_line <= 0:
        return "", []

    preamble_text, _, _ = slice_exact_lines(formal_proofs, 1, preamble_end_line)
    return preamble_text, [
        {
            "kind": "preamble",
            "start_line": 1,
            "end_line": preamble_end_line,
        }
    ]


def build_fragment_assembled_answer(
    formal_proofs: str,
    start_line: int,
    end_line: int,
    helper_blocks: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, int | str]]]:
    fragments: list[dict[str, int | str]] = []
    pieces: list[str] = []

    preamble_text, preamble_fragments = shared_preamble_text(formal_proofs)
    if preamble_text:
        pieces.append(preamble_text)
        fragments.extend(preamble_fragments)

    for helper_block in helper_blocks or []:
        pieces.append(helper_block["full_text"])
        fragments.append(
            {
                "kind": "helper",
                "start_line": helper_block["full_start_line"],
                "end_line": helper_block["end_line"],
            }
        )

    body_text, _, _ = slice_exact_lines(formal_proofs, start_line, end_line)
    pieces.append(body_text)
    fragments.append(
        {
            "kind": "body",
            "start_line": start_line,
            "end_line": end_line,
        }
    )

    return "".join(pieces), fragments


def referenced_helper_blocks(
    formal_proofs: str,
    target_start_line: int,
    target_end_line: int,
) -> list[dict[str, Any]]:
    blocks = top_level_blocks(formal_proofs)
    if not blocks:
        return []

    target_text, _, _ = slice_exact_lines(formal_proofs, target_start_line, target_end_line)
    search_text = target_text
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()

    helper_candidates = [
        block
        for block in blocks
        if block["kind"] == "decl"
        and block["decl_name"]
        and block["label"] is None
        and int(block["full_start_line"]) < target_start_line
    ]

    changed = True
    while changed:
        changed = False
        for block in helper_candidates:
            decl_name = block["decl_name"]
            if not decl_name or decl_name in selected_names:
                continue
            if not re.search(rf"\b{re.escape(decl_name)}\b", search_text):
                continue
            selected.append(block)
            selected_names.add(decl_name)
            search_text += "\n" + block["text"]
            changed = True

    selected.sort(key=lambda block: _int_field(block, "full_start_line"))
    return selected


def validate_extracted_formal_answer(
    formal_answer: str,
    *,
    qa_index: int,
    target_theorem_name: str,
) -> list[str]:
    errors: list[str] = []
    extracted_targets = detect_target_theorems(formal_answer)

    if len(extracted_targets) != 1:
        errors.append(
            f"QA item {qa_index} assembled answer must contain exactly one C-theorem, found {len(extracted_targets)}."
        )
        return errors

    extracted_target = extracted_targets[0]
    expected_label = f"C{qa_index}"
    if extracted_target["label"] != expected_label:
        errors.append(
            f"QA item {qa_index} assembled answer contains label `{extracted_target['label']}`, expected `{expected_label}`."
        )
    if extracted_target["theorem_name"] != target_theorem_name:
        errors.append(
            f"QA item {qa_index} assembled answer contains theorem `{extracted_target['theorem_name']}`, "
            f"expected `{target_theorem_name}`."
        )

    return errors


def validate_llm_extracted_formal_answer(
    formal_answer: str,
    *,
    qa_index: int,
    target_theorem_name: str,
) -> list[str]:
    errors: list[str] = []
    extracted_targets = detect_target_theorems(formal_answer)
    if not extracted_targets:
        return [f"QA item {qa_index} extracted answer contains no detectable C-theorem."]

    matching_targets = [
        target for target in extracted_targets if target["theorem_name"] == target_theorem_name
    ]
    if len(matching_targets) != 1:
        return [
            f"QA item {qa_index} extracted answer must contain target theorem `{target_theorem_name}` exactly once; "
            f"found {len(matching_targets)} occurrence(s)."
        ]

    target = matching_targets[0]
    expected_label = f"C{qa_index}"
    if target["label"] != expected_label:
        errors.append(
            f"QA item {qa_index} extracted answer contains target label `{target['label']}`, expected `{expected_label}`."
        )

    if extracted_targets[-1]["theorem_name"] != target_theorem_name:
        errors.append(
            f"QA item {qa_index} extracted answer must end with target theorem `{target_theorem_name}`."
        )

    for extra_target in extracted_targets:
        if extra_target["theorem_name"] == target_theorem_name:
            continue
        extra_label_number = _label_number(extra_target["label"])
        if extra_label_number is None or extra_label_number >= qa_index:
            errors.append(
                f"QA item {qa_index} extracted answer includes non-target theorem "
                f"`{extra_target['theorem_name']}` with label `{extra_target['label']}`."
            )

    return errors


def slice_exact_lines(text: str, start_line: int, end_line: int) -> tuple[str, int, int]:
    lines = source_lines(text)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise ValueError(
            f"Invalid line span {start_line}-{end_line} for source with {len(lines)} lines."
        )

    start_offset = sum(len(line) for line in lines[: start_line - 1])
    end_offset = sum(len(line) for line in lines[:end_line])
    chunk = text[start_offset:end_offset]
    return chunk, start_offset, end_offset


def build_boundary_prompt(
    entry: dict[str, Any],
    *,
    batch_index: int,
    extra_instruction: str | None = None,
    previous_attempt: dict[str, Any] | None = None,
) -> str:
    qa_batch, formal_proofs = _validate_batch_entry(entry, batch_index=batch_index)
    candidates = detect_target_theorems(formal_proofs)

    if not candidates:
        raise RuntimeError(
            f"Batch {batch_index} has no detectable `theorem Ck...` candidates in `formal_proofs`."
        )

    qa_lines = []
    for qa_index, qa_item in enumerate(qa_batch, start=1):
        qa_lines.append(f"Q{qa_index}: {qa_item['question'].strip()}")
        qa_lines.append(f"A{qa_index}: {qa_item['answer'].strip()}")

    candidate_lines = [
        f"- line {candidate['line_number']}: {candidate['label']} -> {candidate['theorem_name']}"
        for candidate in candidates
    ]

    sections = [
        "You are aligning QA items to proof targets inside a single Lean source file.",
        (
            "Return strict JSON only with shape "
            '{"items":[{"qa_index":1,"target_label":"C1","target_theorem_name":"C1_name",'
            '"start_line":1,"end_line":42,"confidence":"high","justification":"..."}]}.'
        ),
        (
            "Rules:\n"
            "- Choose boundaries only from the provided source lines.\n"
            "- Do not rewrite, summarize, normalize, or emit Lean code.\n"
            "- The app will assemble each answer from exact copied source fragments only.\n"
            "- The shared preamble before the first C-theorem is automatically prepended during verification.\n"
            "- Earlier non-C helper declarations that are referenced by your selected span are automatically prepended during verification.\n"
            "- Therefore your selected span should be the minimal target-local code region needed after that shared preamble and helper stitching.\n"
            "- The chosen span must contain the target theorem for the QA item.\n"
            "- The chosen span must not contain any other C-theorem declaration besides the target theorem.\n"
            "- The chosen span must stop before the next QA item's target proof begins.\n"
            "- Do not include earlier C-theorem declarations for context; helper defs/lemmas are handled separately."
        ),
        f"Batch index: {batch_index}",
        "QA batch:\n" + "\n".join(qa_lines),
        "Detected theorem candidates:\n" + "\n".join(candidate_lines),
        "Lean source with line numbers:\n" + render_numbered_source(formal_proofs),
    ]

    if previous_attempt:
        previous_errors = previous_attempt.get("validation_errors") or []
        previous_boundaries = previous_attempt.get("boundary_items") or []
        previous_extracted_items = previous_attempt.get("extracted_items") or []
        if previous_errors:
            sections.append(
                "Previous attempt failed with these validation errors:\n"
                + "\n".join(f"- {error}" for error in previous_errors)
            )
        if previous_boundaries:
            sections.append(
                "Previous normalized boundary items:\n"
                + json.dumps(previous_boundaries, indent=2, ensure_ascii=False)
            )
        compile_failures = []
        for item in previous_extracted_items:
            if not isinstance(item, dict) or item.get("compile_ok"):
                continue
            qa_index = item.get("qa_index", "?")
            theorem_name = item.get("target_theorem_name", "")
            line_range = f"{item.get('start_line', '?')}-{item.get('end_line', '?')}"
            compile_summary = str(item.get("compile_summary", "")).strip() or "No compiler output."
            compile_failures.append(
                f"QA item {qa_index} / {theorem_name} / lines {line_range} failed to compile:\n{compile_summary}"
            )
        if compile_failures:
            sections.append(
                "Compiler feedback from the previous exact-slice attempt:\n"
                + "\n\n".join(compile_failures)
            )
            sections.append(
                "Repair instruction:\n"
                "Adjust the line boundaries only. Do not rewrite Lean code. "
                "Choose a different exact copied fragment span from `formal_proofs` that preserves semantics "
                "but compiles cleanly without including irrelevant C-theorems."
            )
        elif previous_errors:
            sections.append(
                "Repair instruction:\n"
                "The previous response was structurally invalid. Return exactly one JSON item per QA pair, "
                "with `qa_index` values 1 through the batch size in order. Adjust boundaries only. "
                "Do not rewrite Lean code or include irrelevant C-theorems."
            )

    if extra_instruction and extra_instruction.strip():
        sections.append("Additional user instruction:\n" + extra_instruction.strip())

    return "\n\n".join(sections)


def build_extraction_prompt(
    entry: dict[str, Any],
    *,
    batch_index: int,
    extra_instruction: str | None = None,
    previous_attempt: dict[str, Any] | None = None,
) -> str:
    qa_batch, formal_proofs = _validate_batch_entry(entry, batch_index=batch_index)
    candidates = detect_target_theorems(formal_proofs)

    if not candidates:
        raise RuntimeError(
            f"Batch {batch_index} has no detectable `theorem Ck...` candidates in `formal_proofs`."
        )

    qa_lines = []
    for qa_index, qa_item in enumerate(qa_batch, start=1):
        qa_lines.append(f"Q{qa_index}: {qa_item['question'].strip()}")
        qa_lines.append(f"A{qa_index}: {qa_item['answer'].strip()}")

    candidate_lines = [
        f"- {candidate['label']} -> {candidate['theorem_name']} (line {candidate['line_number']})"
        for candidate in candidates
    ]

    sections = [
        "You are extracting per-QA Lean answers from a single batch-level Lean source file.",
        (
            "Return strict JSON only with shape "
            '{"items":[{"qa_index":1,"target_label":"C1","target_theorem_name":"C1_name",'
            '"formal_answer":"<Lean code>","confidence":"high","justification":"..."}]}.'
        ),
        (
            "Rules:\n"
            "- You must output one `formal_answer` per QA item.\n"
            "- Each `formal_answer` must be valid Lean code that compiles on its own in the project.\n"
            "- Extract code manually from the provided `formal_proofs`; do not summarize.\n"
            "- Include every import, namespace/open statement, helper declaration, and prior theorem dependency needed for compilation.\n"
            "- The target theorem for `Qk/Ak` must be the final C-theorem in that `formal_answer`.\n"
            "- Earlier C-theorems may be included only if they are required dependencies of the target theorem.\n"
            "- Do not include later unrelated C-theorems after the target theorem.\n"
            "- Preserve alignment: `qa_index = k` must end with theorem label `Ck` and the correct target theorem name from the source file.\n"
            "- Do not invent new Lean code outside what is needed to extract and compile the relevant proof content."
        ),
        f"Batch index: {batch_index}",
        "QA batch:\n" + "\n".join(qa_lines),
        "Detected theorem candidates:\n" + "\n".join(candidate_lines),
        "Lean source:\n```lean\n" + formal_proofs + "\n```",
    ]

    if previous_attempt:
        previous_errors = previous_attempt.get("validation_errors") or []
        previous_items = previous_attempt.get("extraction_items") or previous_attempt.get("boundary_items") or []
        previous_extracted_items = previous_attempt.get("extracted_items") or []
        if previous_errors:
            sections.append(
                "Previous attempt failed with these validation errors:\n"
                + "\n".join(f"- {error}" for error in previous_errors)
            )
        if previous_items:
            sections.append(
                "Previous normalized extraction items:\n"
                + json.dumps({"items": previous_items}, indent=2, ensure_ascii=False)
            )
        compile_failures = []
        for item in previous_extracted_items:
            if not isinstance(item, dict) or item.get("compile_ok"):
                continue
            qa_index = item.get("qa_index", "?")
            theorem_name = item.get("target_theorem_name", "")
            compile_summary = str(item.get("compile_summary", "")).strip() or "No compiler output."
            compile_failures.append(
                f"QA item {qa_index} / {theorem_name} failed to compile:\n{compile_summary}"
            )
        if compile_failures:
            sections.append(
                "Compiler feedback from the previous extraction attempt:\n"
                + "\n\n".join(compile_failures)
            )
            sections.append(
                "Repair instruction:\n"
                "Return revised `formal_answer` code for the failing QA items. "
                "You may include additional extracted dependencies from `formal_proofs` if needed for compilation and alignment."
            )

    if extra_instruction and extra_instruction.strip():
        sections.append("Additional user instruction:\n" + extra_instruction.strip())

    return "\n\n".join(sections)


def _decode_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response was empty.")

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    decoder = json.JSONDecoder()
    if stripped and stripped[0] in "{[":
        try:
            return decoder.decode(stripped)
        except json.JSONDecodeError:
            pass

    for index, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            return decoder.raw_decode(stripped[index:])[0]
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not locate valid JSON in the model response.")


def parse_boundary_response(response_text: str) -> list[dict[str, Any]]:
    payload = _decode_json_from_text(response_text)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("items")
    else:
        raise ValueError("Boundary response JSON must be an object or list.")

    if not isinstance(items, list):
        raise ValueError("Boundary response JSON must contain a list-valued `items` field.")

    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each boundary item must be a JSON object.")

        raw_qa_index = item.get("qa_index")
        raw_start_line = item.get("start_line", 1)
        raw_end_line = item.get("end_line")

        try:
            qa_index = int(raw_qa_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qa_index in boundary item: {raw_qa_index!r}") from exc

        try:
            start_line = int(raw_start_line)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid start_line in boundary item: {raw_start_line!r}") from exc

        try:
            end_line = int(raw_end_line)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid end_line in boundary item: {raw_end_line!r}") from exc

        theorem_name = item.get("target_theorem_name")
        if theorem_name is not None and not isinstance(theorem_name, str):
            raise ValueError("`target_theorem_name` must be a string when present.")

        target_label = item.get("target_label")
        if target_label is not None and not isinstance(target_label, str):
            raise ValueError("`target_label` must be a string when present.")

        normalized_items.append(
            {
                "qa_index": qa_index,
                "target_label": (target_label or "").strip(),
                "target_theorem_name": (theorem_name or "").strip(),
                "start_line": start_line,
                "end_line": end_line,
                "confidence": str(item.get("confidence", "")).strip(),
                "justification": str(item.get("justification", "")).strip(),
            }
        )

    return normalized_items


def parse_extraction_response(response_text: str) -> list[dict[str, Any]]:
    payload = _decode_json_from_text(response_text)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("items")
    else:
        raise ValueError("Extraction response JSON must be an object or list.")

    if not isinstance(items, list):
        raise ValueError("Extraction response JSON must contain a list-valued `items` field.")

    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each extraction item must be a JSON object.")

        raw_qa_index = item.get("qa_index")
        try:
            qa_index = int(raw_qa_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qa_index in extraction item: {raw_qa_index!r}") from exc

        theorem_name = item.get("target_theorem_name")
        if theorem_name is not None and not isinstance(theorem_name, str):
            raise ValueError("`target_theorem_name` must be a string when present.")

        target_label = item.get("target_label")
        if target_label is not None and not isinstance(target_label, str):
            raise ValueError("`target_label` must be a string when present.")

        formal_answer = item.get("formal_answer")
        if not isinstance(formal_answer, str) or not formal_answer.strip():
            raise ValueError("Each extraction item must contain a non-empty string `formal_answer`.")

        normalized_items.append(
            {
                "qa_index": qa_index,
                "target_label": (target_label or "").strip(),
                "target_theorem_name": (theorem_name or "").strip(),
                "formal_answer": formal_answer,
                "confidence": str(item.get("confidence", "")).strip(),
                "justification": str(item.get("justification", "")).strip(),
            }
        )

    return normalized_items


def _label_number(label: str) -> int | None:
    match = re.match(r"^C(\d+)$", label.strip())
    if not match:
        return None
    return int(match.group(1))


def _candidate_for_item(
    boundary_item: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    theorem_name = boundary_item.get("target_theorem_name", "").strip()
    label = boundary_item.get("target_label", "").strip()
    start_line = boundary_item["start_line"]
    end_line = boundary_item["end_line"]

    if theorem_name:
        matches = [candidate for candidate in candidates if candidate["theorem_name"] == theorem_name]
        if not matches:
            errors.append(f"No theorem candidate matches `{theorem_name}`.")
            return None, errors
        candidate = matches[0]
        if label and candidate["label"] != label:
            errors.append(
                f"Theorem `{theorem_name}` has label `{candidate['label']}`, not `{label}`."
            )
        return candidate, errors

    if not label:
        errors.append("Boundary item is missing both `target_label` and `target_theorem_name`.")
        return None, errors

    matches = [
        candidate
        for candidate in candidates
        if candidate["label"] == label and start_line <= candidate["line_number"] <= end_line
    ]
    if len(matches) == 1:
        return matches[0], errors
    if not matches:
        errors.append(
            f"No theorem candidate with label `{label}` appears inside lines {start_line}-{end_line}."
        )
    else:
        errors.append(
            f"Multiple theorem candidates with label `{label}` appear inside lines {start_line}-{end_line}; "
            "the response must specify `target_theorem_name`."
        )
    return None, errors


def _candidate_for_extraction_item(
    extraction_item: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    theorem_name = extraction_item.get("target_theorem_name", "").strip()
    label = extraction_item.get("target_label", "").strip()

    if theorem_name:
        matches = [candidate for candidate in candidates if candidate["theorem_name"] == theorem_name]
        if not matches:
            errors.append(f"No theorem candidate matches `{theorem_name}`.")
            return None, errors
        candidate = matches[0]
        if label and candidate["label"] != label:
            errors.append(
                f"Theorem `{theorem_name}` has label `{candidate['label']}`, not `{label}`."
            )
        return candidate, errors

    if not label:
        errors.append("Extraction item is missing both `target_label` and `target_theorem_name`.")
        return None, errors

    matches = [candidate for candidate in candidates if candidate["label"] == label]
    if len(matches) == 1:
        return matches[0], errors
    if not matches:
        errors.append(f"No theorem candidate with label `{label}` was detected.")
    else:
        errors.append(
            f"Multiple theorem candidates with label `{label}` were detected; "
            "the response must specify `target_theorem_name`."
        )
    return None, errors


def _summarize_compile_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append("STDOUT:\n" + stdout.strip())
    if stderr.strip():
        parts.append("STDERR:\n" + stderr.strip())
    return "\n\n".join(parts).strip()


def evaluate_boundary_response(
    entry: dict[str, Any],
    *,
    batch_index: int,
    response_text: str,
    compile_fn: Callable[[str], tuple[int, str, str]] = compile_lean,
) -> dict[str, Any]:
    qa_batch, formal_proofs = _validate_batch_entry(entry, batch_index=batch_index)
    line_total = source_line_count(formal_proofs)
    candidates = detect_target_theorems(formal_proofs)
    timestamp = _utc_now_iso()

    result: dict[str, Any] = {
        "batch_index": batch_index,
        "status": "failed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "lean_prompt": entry.get("lean_prompt", ""),
        "qa_batch": qa_batch,
        "formal_proofs": formal_proofs,
        "formal_proofs_sha256": _hash_text(formal_proofs),
        "line_count": line_total,
        "target_candidates": candidates,
        "llm_response": response_text,
        "boundary_items": [],
        "validation_errors": [],
        "flattened_records": [],
        "extracted_items": [],
    }

    errors: list[str] = []
    try:
        boundary_items = parse_boundary_response(response_text)
    except ValueError as exc:
        result["validation_errors"] = [str(exc)]
        return result

    if len(boundary_items) != len(qa_batch):
        errors.append(
            f"Model returned {len(boundary_items)} boundary items for {len(qa_batch)} QA pairs."
        )

    boundary_items = sorted(boundary_items, key=lambda item: item["qa_index"])
    result["boundary_items"] = boundary_items
    expected_indices = list(range(1, len(qa_batch) + 1))
    actual_indices = [item["qa_index"] for item in boundary_items]
    if actual_indices != expected_indices:
        errors.append(
            f"Boundary item indices must be exactly {expected_indices}, got {actual_indices}."
        )

    extracted_items: list[dict[str, Any]] = []
    for item in boundary_items:
        qa_index = item["qa_index"]
        start_line = item["start_line"]
        end_line = item["end_line"]

        if start_line < 1:
            errors.append(f"QA item {qa_index} has invalid start_line {start_line}.")
            continue
        if end_line < start_line:
            errors.append(
                f"QA item {qa_index} has line range {start_line}-{end_line} with end before start."
            )
            continue
        if end_line > line_total:
            errors.append(
                f"QA item {qa_index} has end_line {end_line}, beyond source line count {line_total}."
            )
            continue

        candidate, candidate_errors = _candidate_for_item(item, candidates)
        if candidate_errors:
            errors.extend(f"QA item {qa_index}: {error}" for error in candidate_errors)
            continue
        if candidate is None:
            continue

        target_line = candidate["line_number"]
        if not start_line <= target_line <= end_line:
            errors.append(
                f"QA item {qa_index} span {start_line}-{end_line} does not include target theorem "
                f"`{candidate['theorem_name']}` on line {target_line}."
            )
            continue

        label = item["target_label"] or candidate["label"]
        label_number = _label_number(label)
        if label_number is None:
            errors.append(f"QA item {qa_index} has invalid target label `{label}`.")
            continue
        if label_number != qa_index:
            errors.append(
                f"QA item {qa_index} should target label `C{qa_index}`, got `{label}`."
            )

        target_block = target_theorem_block(formal_proofs, candidate["theorem_name"])
        if target_block is None:
            errors.append(
                f"QA item {qa_index} target theorem block for `{candidate['theorem_name']}` could not be located."
            )
            continue

        canonical_start_line = int(target_block["start_line"])
        canonical_end_line = int(target_block["end_line"])
        try:
            local_chunk, start_offset, end_offset = slice_exact_lines(
                formal_proofs,
                canonical_start_line,
                canonical_end_line,
            )
        except ValueError as exc:
            errors.append(f"QA item {qa_index}: {exc}")
            continue

        helper_blocks = referenced_helper_blocks(
            formal_proofs,
            canonical_start_line,
            canonical_end_line,
        )
        contaminated_helpers = [
            block.get("decl_name", "")
            for block in helper_blocks
            if isinstance(block, dict) and block.get("label") is not None
        ]
        if contaminated_helpers:
            errors.append(
                f"QA item {qa_index} helper stitching included target-style theorem blocks: {contaminated_helpers}."
            )
            continue
        formal_answer, fragments = build_fragment_assembled_answer(
            formal_proofs,
            canonical_start_line,
            canonical_end_line,
            helper_blocks,
        )
        assembly_errors = validate_extracted_formal_answer(
            formal_answer,
            qa_index=qa_index,
            target_theorem_name=candidate["theorem_name"],
        )
        errors.extend(assembly_errors)

        try:
            compile_returncode, compile_stdout, compile_stderr = compile_fn(formal_answer)
        except Exception as exc:  # pragma: no cover - defensive path for runtime tool failure
            compile_returncode, compile_stdout, compile_stderr = 1, "", str(exc)
            errors.append(f"QA item {qa_index} compile runner failed: {exc}")
            compile_ok = False
        else:
            compile_ok = compile_returncode == 0
            if not compile_ok:
                errors.append(
                    f"QA item {qa_index} exact-fragment assembled answer did not compile "
                    f"(exit code {compile_returncode})."
                )

        qa_item = qa_batch[qa_index - 1]
        extracted_items.append(
            {
                "qa_index": qa_index,
                "target_label": label,
                "target_theorem_name": candidate["theorem_name"],
                "target_line_number": target_line,
                "requested_start_line": start_line,
                "requested_end_line": end_line,
                "start_line": canonical_start_line,
                "end_line": canonical_end_line,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "local_chunk_sha256": _hash_text(local_chunk),
                "formal_answer_sha256": _hash_text(formal_answer),
                "source_fragments": fragments,
                "helper_decl_names": [block["decl_name"] for block in helper_blocks if block.get("decl_name")],
                "question": qa_item["question"],
                "answer": qa_item["answer"],
                "formal_answer": formal_answer,
                "compile_ok": compile_ok,
                "compile_returncode": compile_returncode,
                "compile_stdout": compile_stdout,
                "compile_stderr": compile_stderr,
                "compile_summary": _summarize_compile_output(compile_stdout, compile_stderr),
                "confidence": item.get("confidence", ""),
                "justification": item.get("justification", ""),
            }
        )
    for prev_item, next_item in zip(extracted_items, extracted_items[1:]):
        if prev_item["start_line"] > next_item["start_line"]:
            errors.append(
                f"QA item {prev_item['qa_index']} starts after QA item {next_item['qa_index']}."
            )
        if prev_item["target_line_number"] >= next_item["target_line_number"]:
            errors.append(
                f"QA item {prev_item['qa_index']} target theorem is not before QA item "
                f"{next_item['qa_index']}."
            )
        if prev_item["end_line"] >= next_item["target_line_number"]:
            errors.append(
                f"QA item {prev_item['qa_index']} leaks into the next target theorem starting on "
                f"line {next_item['target_line_number']}."
            )

    if len(extracted_items) == len(qa_batch):
        flattened_records = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "formal_answer": item["formal_answer"],
            }
            for item in extracted_items
        ]
        result["flattened_records"] = flattened_records
    else:
        result["flattened_records"] = []

    result["extracted_items"] = extracted_items
    result["validation_errors"] = errors
    if not errors and len(extracted_items) == len(qa_batch):
        result["status"] = "validated"

    return result


def evaluate_extraction_response(
    entry: dict[str, Any],
    *,
    batch_index: int,
    response_text: str,
    compile_fn: Callable[[str], tuple[int, str, str]] = compile_lean,
) -> dict[str, Any]:
    qa_batch, formal_proofs = _validate_batch_entry(entry, batch_index=batch_index)
    candidates = detect_target_theorems(formal_proofs)
    line_total = source_line_count(formal_proofs)
    timestamp = _utc_now_iso()

    result: dict[str, Any] = {
        "batch_index": batch_index,
        "status": "failed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "lean_prompt": entry.get("lean_prompt", ""),
        "qa_batch": qa_batch,
        "formal_proofs": formal_proofs,
        "formal_proofs_sha256": _hash_text(formal_proofs),
        "line_count": line_total,
        "target_candidates": candidates,
        "llm_response": response_text,
        "extraction_items": [],
        "boundary_items": [],
        "validation_errors": [],
        "flattened_records": [],
        "extracted_items": [],
    }

    errors: list[str] = []
    try:
        extraction_items = parse_extraction_response(response_text)
    except ValueError as exc:
        result["validation_errors"] = [str(exc)]
        return result

    if len(extraction_items) != len(qa_batch):
        errors.append(
            f"Model returned {len(extraction_items)} extraction items for {len(qa_batch)} QA pairs."
        )

    extraction_items = sorted(extraction_items, key=lambda item: item["qa_index"])
    result["extraction_items"] = extraction_items
    result["boundary_items"] = copy.deepcopy(extraction_items)
    expected_indices = list(range(1, len(qa_batch) + 1))
    actual_indices = [item["qa_index"] for item in extraction_items]
    if actual_indices != expected_indices:
        errors.append(
            f"Extraction item indices must be exactly {expected_indices}, got {actual_indices}."
        )

    extracted_items: list[dict[str, Any]] = []
    for item in extraction_items:
        qa_index = item["qa_index"]
        candidate, candidate_errors = _candidate_for_extraction_item(item, candidates)
        if candidate_errors:
            errors.extend(f"QA item {qa_index}: {error}" for error in candidate_errors)
            continue
        if candidate is None:
            continue

        label = item["target_label"] or candidate["label"]
        label_number = _label_number(label)
        if label_number is None:
            errors.append(f"QA item {qa_index} has invalid target label `{label}`.")
            continue
        if label_number != qa_index:
            errors.append(f"QA item {qa_index} should target label `C{qa_index}`, got `{label}`.")

        target_block = target_theorem_block(formal_proofs, candidate["theorem_name"])
        if target_block is None:
            errors.append(
                f"QA item {qa_index} target theorem block for `{candidate['theorem_name']}` could not be located."
            )
            continue

        canonical_start_line = _int_field(target_block, "start_line")
        canonical_end_line = _int_field(target_block, "end_line")
        local_chunk, start_offset, end_offset = slice_exact_lines(
            formal_proofs,
            canonical_start_line,
            canonical_end_line,
        )

        formal_answer = item["formal_answer"]
        assembly_errors = validate_llm_extracted_formal_answer(
            formal_answer,
            qa_index=qa_index,
            target_theorem_name=candidate["theorem_name"],
        )
        errors.extend(assembly_errors)

        try:
            compile_returncode, compile_stdout, compile_stderr = compile_fn(formal_answer)
        except Exception as exc:  # pragma: no cover
            compile_returncode, compile_stdout, compile_stderr = 1, "", str(exc)
            errors.append(f"QA item {qa_index} compile runner failed: {exc}")
            compile_ok = False
        else:
            compile_ok = compile_returncode == 0
            if not compile_ok:
                errors.append(
                    f"QA item {qa_index} exact-fragment assembled answer did not compile "
                    f"(exit code {compile_returncode})."
                )

        qa_item = qa_batch[qa_index - 1]
        extracted_items.append(
            {
                "qa_index": qa_index,
                "target_label": label,
                "target_theorem_name": candidate["theorem_name"],
                "target_line_number": candidate["line_number"],
                "requested_start_line": canonical_start_line,
                "requested_end_line": canonical_end_line,
                "start_line": canonical_start_line,
                "end_line": canonical_end_line,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "local_chunk_sha256": _hash_text(local_chunk),
                "formal_answer_sha256": _hash_text(formal_answer),
                "source_fragments": [],
                "helper_decl_names": [],
                "question": qa_item["question"],
                "answer": qa_item["answer"],
                "formal_answer": formal_answer,
                "compile_ok": compile_ok,
                "compile_returncode": compile_returncode,
                "compile_stdout": compile_stdout,
                "compile_stderr": compile_stderr,
                "compile_summary": _summarize_compile_output(compile_stdout, compile_stderr),
                "confidence": item.get("confidence", ""),
                "justification": item.get("justification", ""),
            }
        )

    if len(extracted_items) == len(qa_batch):
        result["flattened_records"] = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "formal_answer": item["formal_answer"],
            }
            for item in extracted_items
        ]
    else:
        result["flattened_records"] = []

    result["extracted_items"] = extracted_items
    result["validation_errors"] = errors
    if not errors and len(extracted_items) == len(qa_batch):
        result["status"] = "validated"

    return result


def set_attempt_status(attempt: dict[str, Any], status: str) -> dict[str, Any]:
    updated = copy.deepcopy(attempt)
    updated["status"] = status
    updated["updated_at"] = _utc_now_iso()
    updated.setdefault("created_at", updated["updated_at"])
    return updated


def upsert_audit_entry(
    entries: list[dict[str, Any]],
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    updated_entries = [copy.deepcopy(existing) for existing in entries]
    batch_index = entry.get("batch_index")
    if not isinstance(batch_index, int):
        raise RuntimeError("Audit entries must contain an integer `batch_index`.")

    for idx, existing in enumerate(updated_entries):
        if existing.get("batch_index") == batch_index:
            created_at = existing.get("created_at")
            replacement = copy.deepcopy(entry)
            if created_at and not replacement.get("created_at"):
                replacement["created_at"] = created_at
            updated_entries[idx] = replacement
            return updated_entries

    updated_entries.append(copy.deepcopy(entry))
    updated_entries.sort(key=lambda item: item.get("batch_index", 0))
    return updated_entries


def approved_batch_count(entries: list[dict[str, Any]]) -> int:
    return sum(1 for entry in entries if entry.get("status") == "approved")


def next_batch_index(entries: list[dict[str, Any]], total_batches: int) -> int:
    status_by_batch = {
        entry.get("batch_index"): entry.get("status")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("batch_index"), int)
    }
    for batch_index in range(1, total_batches + 1):
        status = status_by_batch.get(batch_index)
        if status not in _FINAL_BATCH_STATUSES:
            return batch_index
    return total_batches + 1


def approved_batch_partitions_from_formal_qa(
    structured_entries: list[dict[str, Any]],
    formal_qa_entries: list[dict[str, str]],
) -> list[tuple[int, list[dict[str, str]]]]:
    partitions: list[tuple[int, list[dict[str, str]]]] = []
    offset = 0
    total_records = len(formal_qa_entries)

    for batch_index, entry in enumerate(structured_entries, start=1):
        qa_batch, _ = _validate_batch_entry(entry, batch_index=batch_index)
        batch_size = len(qa_batch)
        if offset + batch_size > total_records:
            break

        batch_records = formal_qa_entries[offset: offset + batch_size]
        aligned = True
        for qa_item, record in zip(qa_batch, batch_records):
            if qa_item["question"] != record["question"] or qa_item["answer"] != record["answer"]:
                aligned = False
                break

        if not aligned:
            break

        partitions.append((batch_index, copy.deepcopy(batch_records)))
        offset += batch_size

    return partitions


def reconcile_audit_with_formal_qa(
    structured_entries: list[dict[str, Any]],
    audit_entries: list[dict[str, Any]],
    formal_qa_entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    partitions = approved_batch_partitions_from_formal_qa(structured_entries, formal_qa_entries)
    if not partitions:
        return []

    audit_by_batch = {
        entry.get("batch_index"): entry
        for entry in audit_entries
        if isinstance(entry, dict) and isinstance(entry.get("batch_index"), int)
    }

    reconciled: list[dict[str, Any]] = []
    now = _utc_now_iso()
    for batch_index, records in partitions:
        structured_entry = structured_entries[batch_index - 1]
        existing = audit_by_batch.get(batch_index, {})
        reconciled_entry: dict[str, Any] = {
            "batch_index": batch_index,
            "status": "approved",
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "lean_prompt": structured_entry.get("lean_prompt", ""),
            "qa_batch": copy.deepcopy(structured_entry.get("qa_batch", [])),
            "formal_proofs": structured_entry.get("formal_proofs", ""),
            "flattened_records": records,
            "validation_errors": [],
        }

        for key in [
            "llm_response",
            "boundary_items",
            "extracted_items",
            "target_candidates",
            "formal_proofs_sha256",
            "line_count",
        ]:
            if key in existing:
                reconciled_entry[key] = copy.deepcopy(existing[key])

        reconciled.append(reconciled_entry)

    return reconciled


def flatten_approved_audit_entries(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    approved_entries = sorted(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("status") == "approved"
        ),
        key=lambda entry: entry["batch_index"],
    )

    flattened: list[dict[str, str]] = []
    for entry in approved_entries:
        records = entry.get("flattened_records")
        if not isinstance(records, list):
            records = [
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "formal_answer": item.get("formal_answer", ""),
                }
                for item in entry.get("extracted_items", [])
                if isinstance(item, dict)
            ]

        for record in records:
            if not isinstance(record, dict):
                continue
            question = record.get("question")
            answer = record.get("answer")
            formal_answer = record.get("formal_answer")
            if not isinstance(question, str) or not isinstance(answer, str) or not isinstance(formal_answer, str):
                continue
            flattened.append(
                {
                    "question": question,
                    "answer": answer,
                    "formal_answer": formal_answer,
                }
            )

    return flattened


def write_formal_qa_data_from_audit(
    entries: list[dict[str, Any]],
    output_path: Path = DEFAULT_FORMAL_QA_OUTPUT_PATH,
) -> list[dict[str, str]]:
    flattened = flatten_approved_audit_entries(entries)
    _write_json(output_path, flattened)
    return flattened
