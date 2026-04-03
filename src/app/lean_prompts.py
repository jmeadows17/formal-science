"""Utilities for building Lean autoformalisation prompts from QA batches."""

from __future__ import annotations

import json
import textwrap
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


def build_lean_prompt(batch: Batch) -> str:
    """
    Build the Lean prompt for a single QA batch.

    The prompt numbering is local to the batch:
    questions are labelled ``Q1`` to ``Qn``, answers ``A1`` to ``An``,
    and requested Lean theorems ``C1`` to ``Cn``.
    """
    validated_batch = _validate_batch(batch)
    count = len(validated_batch)
    q_range = _label_range("Q", count)
    a_range = _label_range("A", count)
    c_range = _label_range("C", count)
    derivation_noun = "mini-derivation" if count == 1 else "mini-derivations"
    range_verb = "is" if count == 1 else "are"

    header = textwrap.dedent(
        f"""\
        # Task
        Autoformalise {count} physics/math {derivation_noun} ({q_range}) into **compilable Lean 4 + Mathlib** proofs that *derive* the stated results from explicit premises. Do **not** introduce new axioms or hand-wave steps; use Mathlib lemmas wherever appropriate.

        # Inputs
        Use the following questions and high-level derivation sketches as guidance (you may restate them formally).
        {q_range} {range_verb} exactly as below; derive the same endpoints, but do it properly in Lean:
        """
    )

    qa_text = _build_qa_block(validated_batch)

    requirements = textwrap.dedent(
        """\
        # Requirements
        1. **No new axioms.** Do not use `axiom`, `sorry`, or postulated equalities.
        2. **Use Mathlib theorems** for calculus and functional equality etc., and import dot-style submodules e.g. ```import Mathlib.Tactic```.
        3. **Physics modelling must be explicit and minimal.**
        4. **Compilation:** Provide a **single Lean file** that compiles with current Mathlib. Use only standard imports (e.g., `Mathlib.Data.Complex.Basic`).
        5. **Clarity:** For each Ci (corresponding to Qi), include:
           - A brief `/-! ... -/` docstring mapping the formal statement to the physics/math.
           - A `theorem` (not `example`) with clear names, explicit hypotheses, and a readable proof.
        6. **No hidden magic:** Prefer `calc` blocks and small helper lemmas over massive `simp` bundles. You may use `ring`, `linarith`, `simp`, and `field_simp` where appropriate.
        7. **Determinism:** Avoid brittle rewrites. Keep proofs robust across Mathlib updates.
        """
    )

    deliverables = textwrap.dedent(
        f"""\
        # Deliverables
        - One Lean file saved to ```FSLean/proof.lean``` relative to the repository root (the current workspace root). Do **not** create a nested ```formal-science/``` directory. The file must contain **{c_range}** exactly in this order.
        """
    )

    acceptance = textwrap.dedent(
        """\
        # Acceptance criteria
        - File compiles on a stock Mathlib toolchain.
        - No axioms/sorries; proofs are readable and well-commented.
        """
    )

    closing = textwrap.dedent(
        f"""\
        Now, using {q_range} and {a_range} above, write **compilable Lean 4 code ({c_range})** that solves each question `Q` according to the corresponding answer `A`, adhering strictly to the Requirements, Deliverables, and Acceptance criteria.
        """
    )

    return "\n\n".join([header, qa_text, requirements, deliverables, acceptance, closing])


def build_lean_prompt_from_dataset(
    batch_index: int,
    path: str | Path = DEFAULT_QA_DATA_PATH,
) -> str:
    """Load a batch from disk and build its Lean prompt."""
    return build_lean_prompt(get_qa_batch(batch_index, path))


def build_lean_prompt_dataset(batches: list[Batch]) -> list[str]:
    """Build one Lean prompt for each QA batch."""
    validated_batches = _validate_batches(batches)
    return [build_lean_prompt(batch) for batch in validated_batches]


def build_lean_prompt_dataset_from_file(
    path: str | Path = DEFAULT_QA_DATA_PATH,
) -> list[str]:
    """Load all QA batches from disk and build their Lean prompts."""
    return build_lean_prompt_dataset(load_qa_batches(path))


def _build_qa_block(batch: Batch) -> str:
    entries = []
    for index, pair in enumerate(batch, start=1):
        question = pair["question"].strip()
        answer = pair["answer"].strip()
        entries.append(f"- **Q{index}:** {question}\n\n  **A{index}:** {answer}\n")
    return "\n".join(entries)


def _label_range(prefix: str, count: int) -> str:
    if count < 1:
        raise ValueError("count must be positive")
    if count == 1:
        return f"{prefix}1"
    return f"{prefix}1-{prefix}{count}"


def _validate_batches(data: Any) -> list[Batch]:
    if not isinstance(data, list):
        raise TypeError("qa_data.json must contain a list of QA batches")
    return [_validate_batch(batch) for batch in data]


def _validate_batch(batch: Any) -> Batch:
    if not isinstance(batch, list) or not batch:
        raise TypeError("each QA batch must be a non-empty list")

    validated: Batch = []
    for pair in batch:
        if not isinstance(pair, dict):
            raise TypeError("each QA pair must be a dictionary")
        question = pair.get("question")
        answer = pair.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("each QA pair must contain string 'question' and 'answer' fields")
        validated.append({"question": question, "answer": answer})
    return validated
