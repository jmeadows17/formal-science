# formal-science

`formal-science` is an experimental pipeline for turning informal physics and math derivations into:

1. reviewed natural-language question/answer pairs,
2. Lean 4 autoformalisation prompts,
3. batch-level Lean proofs, and
4. flattened `(question, answer, formal_answer)` training data.

The repository includes both the code for running that pipeline and a checked-in snapshot of generated datasets under [`src/app_data`](./src/app_data).

This implementation is intended as an improved version of the pipeline analyzed in the *FormalScience* paper at ACL 2026. In the comparison notes retained in [`legacy/formalphysics_comparison.md`](./legacy/formalphysics_comparison.md), the dataset referred to as `FormalPhysics_v2.json` corresponds to this repo's [`src/app_data/formal_qa_data.json`](./src/app_data/formal_qa_data.json).

## To-Do

- remove irrelevant code
- revamp custom dataset input

## External Links

- FormalScience paper (ACL 2026 / arXiv): [arXiv:2604.23002](https://arxiv.org/abs/2604.23002)
- Updated FormalPhysics dataset: [`jmeadows17/FormalPhysics`](https://huggingface.co/datasets/jmeadows17/FormalPhysics)

## What The Repo Does

The workflow is organized as a staged human-in-the-loop pipeline:

1. `src/app/app.py` generates QA batches from source derivations using a local Claude or Codex/GPT CLI session.
2. `src/qa/qa_postprocessing.py` cleans model output into structured QA pairs.
3. `src/app/lean_prompts.py` converts each QA batch into a Lean 4 + Mathlib autoformalisation prompt.
4. `src/app/lean_app.py` generates batch-level Lean proofs and checks them with the local Lean project in [`FSLean`](./FSLean).
5. `src/app/postprocessing_app.py` validates and splits each batch-level Lean file into per-question formal answers.
6. `src/app/flatten_structured_proofs.py` writes the final flattened dataset in [`src/app_data/formal_qa_data.json`](./src/app_data/formal_qa_data.json).

The current checked-in data snapshot includes intermediate artifacts for the full pipeline, so the repo is usable both as code and as a dataset snapshot.

## Relation To The ACL 2026 FormalScience Version

This repository is not just a copy of the earlier FormalScience workflow. It is a revised implementation with stronger validation and postprocessing around the same broad task: converting informal derivations into aligned formal artifacts.

Relative to the earlier version analyzed in the ACL 2026 paper, this pipeline adds or tightens:

- human review at multiple stages,
- structured QA postprocessing,
- Lean compilation checks inside a local Mathlib project,
- theorem-boundary validation for extracted formal answers,
- an explicit audit trail in [`src/app_data/postprocessed_batches.json`](./src/app_data/postprocessed_batches.json).

The comparison summary in [`legacy/formalphysics_comparison.md`](./legacy/formalphysics_comparison.md) describes the resulting `FormalPhysics_v2` dataset as stronger than the original `FormalPhysics` version on semantic fidelity, logical preservation, and consistency.

### Comparison Snapshot

From the comparison note:

| Metric | FormalPhysics | FormalPhysics_v2 |
|---|---:|---:|
| Examples | 200 | 215 |
| Mean Score | 3.84 / 5 | **4.50 / 5** |
| Score >= 4 (%) | 69% | **~90%+** |
| Severe Failures | Present | Rare |
| Semantic Fidelity | Moderate, inconsistent | **High, consistent** |
| Logical Preservation | Mixed | **Strong** |
| Depth Preservation | Often reduced via assumptions | **Generally preserved** |
| Main Weakness | Semantic drift, tautologies, mismatches | Premise-loading (assumed theorems) |

In the naming used by this repository:

- earlier dataset version: `FormalPhysics`, corresponding to the original dataset file at [`data/FormalPhysics.json`](https://github.com/jmeadows17/formal-science/blob/main/data/FormalPhysics.json)
- improved dataset version (`FormalPhysics_v2` in the comparison note): [`src/app_data/formal_qa_data.json`](./src/app_data/formal_qa_data.json)

## Repository Layout

- [`src/app`](./src/app): Gradio apps, Lean compilation helpers, and proof postprocessing utilities.
- [`src/qa`](./src/qa): QA prompt generation and QA-output cleanup.
- [`src/llm`](./src/llm): wrappers around local `claude` and `codex` CLIs.
- [`src/app_data`](./src/app_data): generated datasets and pipeline artifacts.
- [`FSLean`](./FSLean): local Lean 4 / Mathlib project used to compile generated proofs.
- [`data`](./data): source dataset files and notebooks.
- [`legacy`](./legacy): earlier notebooks, source materials, and experiments retained for reference.

## Key Data Files

The main generated JSON artifacts are:

- [`src/app_data/qa_data.json`](./src/app_data/qa_data.json): cleaned QA batches.
- [`src/app_data/lean_prompt_data.json`](./src/app_data/lean_prompt_data.json): one Lean prompt per QA batch.
- [`src/app_data/lean_output_data.json`](./src/app_data/lean_output_data.json): raw batch-level Lean generation outputs.
- [`src/app_data/structured_proofs.json`](./src/app_data/structured_proofs.json): reviewed batch records containing QA plus approved Lean source.
- [`src/app_data/postprocessed_batches.json`](./src/app_data/postprocessed_batches.json): audit trail for proof-boundary extraction and validation.
- [`src/app_data/formal_qa_data.json`](./src/app_data/formal_qa_data.json): flattened final dataset with `question`, `answer`, and `formal_answer`; this is the dataset referred to as `FormalPhysics_v2` in [`legacy/formalphysics_comparison.md`](./legacy/formalphysics_comparison.md).

## Requirements

### Python

The repository is not packaged yet, so install the small set of runtime tools manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio pytest
```

### Lean

Generated proofs are checked against the Lean project in [`FSLean`](./FSLean), which currently uses:

- Lean toolchain: `leanprover/lean4:v4.24.0-rc1`
- Mathlib via `lake`

After cloning the repo:

```bash
cd FSLean
lake build
cd ..
```

### Local LLM CLI Access

The interactive apps do not call hosted APIs directly. They expect local CLI tooling:

- `src/llm/claude_cli.py` expects a working `claude` CLI or a discoverable Claude Code VS Code extension binary.
- `src/llm/gpt_cli.py` expects a working `codex` CLI or a discoverable ChatGPT/Codex VS Code extension binary.

If you only want to inspect the checked-in datasets or run pure postprocessing/tests, you do not need those CLIs.

## Running The Pipeline

### 1. Build or review QA batches

```bash
python src/app/app.py
```

This launches a Gradio app for generating and reviewing QA pairs from the source derivations. Approved outputs are saved to `src/app_data/qa_data.json`, and Lean prompts are autosaved to `src/app_data/lean_prompt_data.json`.

### 2. Generate Lean proofs

```bash
python src/app/lean_app.py
```

This launches the Lean generation app. It feeds prompts from `lean_prompt_data.json`, compiles candidate Lean code via `lake env lean`, and saves reviewed outputs to `src/app_data/lean_output_data.json` and `src/app_data/structured_proofs.json`.

### 3. Validate and split proofs into per-QA formal answers

```bash
python src/app/postprocessing_app.py
```

This app aligns each QA item with the correct theorem block in the batch-level Lean source, validates the extracted fragments, and writes the final flattened data to `src/app_data/formal_qa_data.json`.

## Useful Scripts

- `python src/qa/qa_postprocessing.py <input.json> [output.json]`
  Cleans raw model QA output into batched QA pairs.
- `python src/app/flatten_structured_proofs.py`
  Rebuilds `formal_qa_data.json` from `structured_proofs.json`.
- `python src/app/claude_generate_compile_loop.py --index 0 --prompt-file src/app_data/lean_prompt_data.json`
  Runs a standalone Claude generate/compile/fix loop against `FSLean/proof.lean`.

## Testing

Run the current unit tests with:

```bash
pytest src/qa/tests src/app/tests
```

These tests cover:

- QA markdown parsing and cleanup behavior,
- batch-size inference and malformed-output rejection,
- proof-boundary parsing and extraction validation,
- helper/preamble assembly for extracted Lean fragments.

## Notes

- The source material and prompt templates are currently tailored to the physics-style derivations included in this repo, rather than being a general-purpose formalisation framework.
- The main residual weakness called out in the comparison note is premise-loading: some formalizations stay semantically aligned but assume intermediate theorems rather than deriving every step in the strongest possible way.
- `legacy/` contains notebooks and earlier assets that informed the current code but are not part of the main runtime path.
- `FSLean/proof.lean` is a working output target for generated Lean code; the pipeline may overwrite it during generation experiments.
