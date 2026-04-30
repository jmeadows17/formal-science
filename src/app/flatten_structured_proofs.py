"""Flatten structured Lean proofs into aligned question/answer/proof records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "app_data" / "structured_proofs.json"
DEFAULT_OUTPUT_PATH = ROOT / "app_data" / "formal_qa_data.json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _is_bad_comment(comment: str, marker_count: int) -> bool:
    markers = [f"C{i}" for i in range(1, marker_count + 1)]
    counts = [comment.count(marker) for marker in markers]
    present_markers = len(markers) - counts.count(0)
    return present_markers > 1


def _extract_comments(example: str, marker_count: int) -> list[str]:
    comments: list[str] = []
    splits = example.split("/-")
    for split in splits[1:]:
        comment = "/-" + split.split("-/")[0] + "-/"
        if _is_bad_comment(comment, marker_count):
            comments.append(comment)

    for line in example.splitlines(keepends=True):
        if _is_bad_comment(line, marker_count):
            comments.append(line)

    return comments


def _get_imports(example: str) -> str:
    lines = example.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if "C1" in line:
            return "".join(lines[:index])
    raise ValueError("Could not locate the C1 marker in the formal proof output.")


def _remove_unterminated_block_open_tokens(text: str) -> str:
    """Remove unmatched `/-` or `/-!` opening tokens while keeping closed blocks."""
    stack: list[tuple[int, int]] = []
    index = 0
    length = len(text)

    while index < length:
        if text.startswith("/-", index):
            token_length = 3 if index + 2 < length and text[index + 2] == "!" else 2
            stack.append((index, token_length))
            index += token_length
            continue
        if text.startswith("-/", index):
            if stack:
                stack.pop()
            index += 2
            continue
        index += 1

    if not stack:
        return text

    to_delete = set()
    for position, token_length in stack:
        to_delete.update(range(position, position + token_length))
    return "".join(char for idx, char in enumerate(text) if idx not in to_delete)


def _find_marker_indices(lines: list[str], expected_count: int) -> dict[int, int]:
    marker_indices: dict[int, int] = {}
    for marker in range(2, expected_count + 1):
        marker_text = f"C{marker}"
        for index, line in enumerate(lines):
            if marker_text in line:
                marker_indices[marker] = index
                break
        else:
            raise ValueError(f"Could not locate the {marker_text} marker in the formal proof output.")
    return marker_indices


def _split_cleaned_example(example: str, expected_count: int) -> list[str]:
    lines = example.splitlines(keepends=True)
    _get_imports(example)
    marker_indices = _find_marker_indices(lines, expected_count)

    chunks: list[str] = []
    for marker in range(2, expected_count + 1):
        stop = marker_indices[marker]
        # Keep each chunk self-contained by making later chunks cumulative.
        # This preserves earlier theorem statements and helper lemmas when a later
        # proof refers back to them (for example `C3` using `C2`, or `C5` using
        # a lemma defined after `C4` but before `C5`).
        chunks.append(_remove_unterminated_block_open_tokens("".join(lines[:stop])).strip())

    chunks.append(_remove_unterminated_block_open_tokens("".join(lines)).strip())
    return chunks


def split_formal_proofs(formal_proofs: str, expected_count: int) -> list[str]:
    """Split one Lean source blob into per-question proof chunks."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")

    cleaned_example = formal_proofs
    for comment in _extract_comments(cleaned_example, expected_count):
        cleaned_example = cleaned_example.replace(comment, "")

    try:
        return _split_cleaned_example(cleaned_example, expected_count)
    except ValueError:
        # Some generated docstrings mention earlier proof labels (for example `C1` and `C2`
        # inside the prose for `C3`). In that case the notebook cleanup can erase the very
        # marker lines we need, so fall back to splitting the raw file and keep the same
        # per-chunk cleanup pass.
        return _split_cleaned_example(formal_proofs, expected_count)


def flatten_structured_proofs(data: Any) -> list[dict[str, str]]:
    if not isinstance(data, list):
        raise RuntimeError("structured_proofs.json must contain a top-level JSON list.")

    flattened: list[dict[str, str]] = []
    for batch_index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Entry {batch_index} must be a JSON object.")

        qa_batch = entry.get("qa_batch")
        formal_proofs = entry.get("formal_proofs")
        if not isinstance(qa_batch, list):
            raise RuntimeError(f"Entry {batch_index} must contain a list-valued `qa_batch`.")
        if not isinstance(formal_proofs, str):
            raise RuntimeError(f"Entry {batch_index} must contain a string-valued `formal_proofs`.")

        proof_chunks = split_formal_proofs(formal_proofs, len(qa_batch))
        if len(proof_chunks) != len(qa_batch):
            raise RuntimeError(
                f"Entry {batch_index} produced {len(proof_chunks)} proof chunks for "
                f"{len(qa_batch)} QA pairs."
            )

        for qa_index, (qa_item, formal_answer) in enumerate(zip(qa_batch, proof_chunks), start=1):
            if not isinstance(qa_item, dict):
                raise RuntimeError(
                    f"Entry {batch_index} QA item {qa_index} must be an object with "
                    "`question` and `answer`."
                )

            question = qa_item.get("question")
            answer = qa_item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise RuntimeError(
                    f"Entry {batch_index} QA item {qa_index} must contain string-valued "
                    "`question` and `answer` fields."
                )

            flattened.append(
                {
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "formal_answer": formal_answer,
                }
            )

    return flattened


def write_flattened_structured_proofs(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, str]]:
    flattened = flatten_structured_proofs(_load_json(input_path))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(flattened, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write {output_path}: {exc}") from exc
    return flattened


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split each batch-level `formal_proofs` Lean file into aligned per-question "
            "records and write a flattened JSON dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to structured proofs JSON (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the flattened output JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    flattened = write_flattened_structured_proofs(args.input, args.output)
    print(f"Wrote {len(flattened)} flattened records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
