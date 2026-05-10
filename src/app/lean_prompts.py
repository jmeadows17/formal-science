"""Utilities for building one-QA-at-a-time Lean prompts from batched QA data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Pair = dict[str, str]
Batch = list[Pair]

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_DATA_PATH = _ROOT / "src" / "app_data" / "qa_data.json"

def load_qa_batches(path: str | Path = DEFAULT_QA_DATA_PATH) -> list[Batch]:
    """Load the batched QA dataset stored in ``qa_data.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_batches(data)


def get_qa_batch(batch_index: int, path: str | Path = DEFAULT_QA_DATA_PATH) -> Batch:
    """Return a single validated QA batch by index."""
    batches = load_qa_batches(path)
    return batches[batch_index]


def build_lean_prompt(
    pair: Pair,
    *,
    qa_index: int = 1,
    batch_size: int = 1,
) -> str:
    """Build a minimal Lean-generation prompt for a single QA pair."""
    validated_pair = _validate_pair(pair)
    qa_json = json.dumps([validated_pair], indent=2, ensure_ascii=False)

    return (
        "Write Lean 4 + Mathlib code to `FSLean/proof.lean` for the QA batch below.\n\n"
        "Use `src/app/autoformalisation_rubric.md` as the alignment standard. "
        "The final Lean file must match the QA items semantically, satisfy the rubric, "
        "avoid surrogate objects and theorem-shaped assumptions, and compile in the local project.\n\n"
        "This prompt covers exactly one QA pair.\n\n"
        "QA batch:\n"
        f"```json\n{qa_json}\n```\n\n"
        "Produce exactly one aligned target theorem for this QA pair, and include the rubric-required "
        "comments needed for auditability."
    )


def build_lean_prompt_from_dataset(
    batch_index: int,
    path: str | Path = DEFAULT_QA_DATA_PATH,
) -> list[str]:
    """Load one batch from disk and build one prompt per QA pair in that batch."""
    batch = get_qa_batch(batch_index, path)
    return [
        build_lean_prompt(pair, qa_index=qa_index, batch_size=len(batch))
        for qa_index, pair in enumerate(batch, start=1)
    ]


def build_lean_prompt_dataset(batches: list[Batch]) -> list[str]:
    """Build one Lean prompt for each QA pair across all batches."""
    validated_batches = _validate_batches(batches)
    prompts: list[str] = []
    for batch in validated_batches:
        for qa_index, pair in enumerate(batch, start=1):
            prompts.append(build_lean_prompt(pair, qa_index=qa_index, batch_size=len(batch)))
    return prompts


def build_lean_prompt_dataset_from_file(
    path: str | Path = DEFAULT_QA_DATA_PATH,
) -> list[str]:
    """Load all QA batches from disk and build one prompt per QA pair."""
    return build_lean_prompt_dataset(load_qa_batches(path))


def _validate_batches(data: Any) -> list[Batch]:
    if not isinstance(data, list):
        raise TypeError("qa_data.json must contain a list of QA batches")
    return [_validate_batch(batch) for batch in data]


def _validate_batch(batch: Any) -> Batch:
    if not isinstance(batch, list) or not batch:
        raise TypeError("each QA batch must be a non-empty list")

    validated: Batch = []
    for pair in batch:
        validated.append(_validate_pair(pair))
    return validated


def _validate_pair(pair: Any) -> Pair:
    if not isinstance(pair, dict):
        raise TypeError("each QA pair must be a dictionary")
    question = pair.get("question")
    answer = pair.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError("each QA pair must contain string 'question' and 'answer' fields")
    return {"question": question, "answer": answer}
