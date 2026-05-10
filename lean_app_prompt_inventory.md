# Lean App Prompt Inventory

This document inventories the prompt-related text, requirements, and instructions used by the Lean app stack in:

- [src/app/lean_app.py](/home/jmeadows17/formal-science/src/app/lean_app.py)
- [src/app/lean_prompts.py](/home/jmeadows17/formal-science/src/app/lean_prompts.py)

It distinguishes:

- model-facing prompt text sent to the LLM
- default-pipeline prompt construction
- user-facing UI/help text that describes prompt behavior

## 1. Prompt Source Flow

The default pipeline in `lean_app.py` does not hardcode the substantive math/physics task prompt. Instead:

1. It loads `lean_prompt_data.json` and `qa_data.json` in `_load_prompt_qa_pairs()` ([lean_app.py:56](/home/jmeadows17/formal-science/src/app/lean_app.py:56)).
2. If `lean_prompt_data.json` is missing or stale, it rebuilds it from `qa_data.json` with `build_lean_prompt_dataset_from_file()` from `lean_prompts.py`.
3. `qa_data.json` stays batched, but `lean_prompt_data.json` is flattened to one prompt per QA pair in batch order.
4. In default mode, the prompt actually sent to the model is `PROMPTS[prompt_index]`.
5. In custom mode, the prompt actually sent to the model is the stripped contents of the custom prompt textbox.
6. In both modes, `lean_app.py` appends additional operational instructions telling the model to write to `FSLean/proof.lean`.

## 2. Model-Facing Prompt Templates in `lean_app.py`

### 2.1 Initial generation suffix

Defined as `CODE_ONLY_SUFFIX` in [lean_app.py:295](/home/jmeadows17/formal-science/src/app/lean_app.py:295) and appended by `_build_model_message()` in [lean_app.py:510](/home/jmeadows17/formal-science/src/app/lean_app.py:510):

```text
# Task
Write the complete Lean 4 source code directly to `FSLean/proof.lean`.
Do not write to any other file.
Do not output the code as part of your response.
After writing `FSLean/proof.lean`, stop immediately.
Do not read the file back.
Do not provide any explanation beyond a brief confirmation that the file was written.
```

This is the most important operational instruction in the app. It forces file output rather than chat output.

### 2.2 Compile-fix suffix

Defined as `COMPILE_FIX_SUFFIX` in [lean_app.py:306](/home/jmeadows17/formal-science/src/app/lean_app.py:306) and used by `_build_compile_fix_message()` in [lean_app.py:561](/home/jmeadows17/formal-science/src/app/lean_app.py:561):

```text
The Lean file did not compile.

Fix `FSLean/proof.lean` in place to resolve the compiler errors below.
Do not write to any other file.
After updating `FSLean/proof.lean`, stop immediately.
Do not read the file back.
Do not provide any explanation beyond a brief confirmation that the file was updated.
```

`_build_compile_fix_message()` then appends:

- `Compile attempt: {iteration}`
- `Compiler output:`
- the formatted compiler output text
- `Current Lean code:` followed by a fenced Lean code block

See [lean_app.py:561](/home/jmeadows17/formal-science/src/app/lean_app.py:561).

### 2.3 Alignment / evaluation prompt

Defined as `ALIGNMENT_PROMPT` in [lean_app.py:316](/home/jmeadows17/formal-science/src/app/lean_app.py:316):

```text
Using a 5-point Likert scale to 1dp, determine how well each Lean code proof Ci successfully proves the target results from Qi and Ai, and aligns with the Requirements. Use any score in 0.1 increments from 1.0 to 5.0 when needed; do not default to whole-number scores. If a patch difference is provided, begin with a brief **Patch Difference** note summarizing what changed and whether the change is substantive; explicitly say if there was no meaningful change.
```

`_build_alignment_message()` constructs the full evaluation prompt as:

1. `Initial prompt:\n{prompt_text}`
2. `Current proof.lean code:` plus a fenced Lean block
3. optionally `Patch difference from previous evaluated proof:` plus a fenced diff block
4. the `ALIGNMENT_PROMPT` text

See [lean_app.py:573](/home/jmeadows17/formal-science/src/app/lean_app.py:573).

### 2.4 Refinement prompt

`_build_refinement_message()` in [lean_app.py:596](/home/jmeadows17/formal-science/src/app/lean_app.py:596) sends this fixed instruction block:

```text
Revise `FSLean/proof.lean` so it better satisfies the initial prompt and addresses the latest alignment feedback.
Update the file directly. Do not write to any other file.
After updating `FSLean/proof.lean`, stop immediately.
Do not read the file back.
Do not provide any explanation beyond a brief confirmation that the file was updated.
```

It then appends:

1. `Initial prompt:\n{prompt_text}`
2. optionally `Latest alignment feedback:\n{normalized_feedback}`
3. `User request:\n{user_instruction}`
4. `Current proof.lean code:` plus a fenced Lean block

The normalization step strips placeholder texts such as “Alignment evaluation will appear here…” and “Evaluating alignment…” before passing feedback back to the model ([lean_app.py:473](/home/jmeadows17/formal-science/src/app/lean_app.py:473), [lean_app.py:612](/home/jmeadows17/formal-science/src/app/lean_app.py:612)).

## 3. Default-Pipeline Prompt Template in `lean_prompts.py`

When `lean_prompt_data.json` is absent, `lean_app.py` rebuilds prompt text from `qa_data.json` using `lean_prompts.py` ([lean_app.py:67](/home/jmeadows17/formal-science/src/app/lean_app.py:67)).

### 3.1 One-pair-at-a-time semantics

`build_lean_prompt_dataset()` now flattens the original batched `qa_data.json` into one prompt per QA pair while preserving batch order.

Each prompt:

- includes exactly one QA pair serialized as JSON
- tells the model to align with `src/app/autoformalisation_rubric.md`
- tells the model the theorem label it must use, e.g. `C2`, so approved single-pair proofs can later be rebuilt into their original batch

### 3.2 Prompt template

The prompt is now intentionally minimal. In substance it says:

```text
Write Lean 4 + Mathlib code to `FSLean/proof.lean` for the QA batch below.
Use `src/app/autoformalisation_rubric.md` as the alignment standard.
This prompt is for a single QA pair from a larger batch.
The target theorem for this pair must be named `Ck`.
QA batch:
```json
[{"question":"...","answer":"..."}]
```
Produce exactly one aligned target theorem for this QA pair, and include the rubric-required comments needed for auditability.
```

## 4. Validation Constraints Relevant to Prompt Construction

The prompt builder only accepts batched QA data matching the expected schema:

- top-level value must be a list of batches
- each batch must be a non-empty list
- each QA pair must be a dictionary
- each pair must contain string `question` and `answer` fields

`lean_app.py` performs an additional compatibility check:

- `lean_prompt_data.json` and `qa_data.json` must both be top-level lists
- the prompt file must contain the same number of entries as the flattened QA-pair count

See [lean_app.py:81](/home/jmeadows17/formal-science/src/app/lean_app.py:81).

## 5. User-Facing Prompt / UI Text in `lean_app.py`

These are not model-facing instructions, but they describe prompt behavior to the user.

### 5.1 Module docstring

Defined at the top of the file ([lean_app.py:1](/home/jmeadows17/formal-science/src/app/lean_app.py:1)):

```text
Gradio Lean Code Generator powered by Claude or GPT CLI.

Feeds prompts from ``lean_prompt_data.json`` one at a time to an LLM and
displays the generated Lean code.  Also supports custom prompts.

Run: python src/app/lean_app.py
```

### 5.2 Missing-dataset setup message

Defined as `DEFAULT_DATASET_SETUP_MESSAGE` ([lean_app.py:38](/home/jmeadows17/formal-science/src/app/lean_app.py:38)):

```text
Default pipeline data is missing. Populate `src/app_data/qa_data.json` first, or run the QA builder (`python src/app/app.py`) to generate both `qa_data.json` and `lean_prompt_data.json`. Custom mode is still available.
```

### 5.3 Main UI header

Rendered in `render_lean_builder_ui()` ([lean_app.py:1219](/home/jmeadows17/formal-science/src/app/lean_app.py:1219)):

```text
# Lean Code Generator
Feed prompts from `lean_prompt_data.json` to an LLM one at a time, or enter your own custom prompt.
Review each generated Lean output before approving.
```

### 5.4 Mode help text

Defined on the mode selector ([lean_app.py:1244](/home/jmeadows17/formal-science/src/app/lean_app.py:1244)):

```text
Default: iterates through lean_prompt_data.json. Custom: enter your own prompt.
```

### 5.5 Reasoning effort help text

Defined on the reasoning dropdown ([lean_app.py:1257](/home/jmeadows17/formal-science/src/app/lean_app.py:1257)):

```text
Claude: low/medium/high/max. GPT choices depend on the selected model.
```

### 5.6 Custom prompt placeholder

Defined on the custom prompt box ([lean_app.py:1267](/home/jmeadows17/formal-science/src/app/lean_app.py:1267)):

```text
Enter your Lean generation prompt here...
```

### 5.7 Refinement message placeholder

Defined on the message box ([lean_app.py:1315](/home/jmeadows17/formal-science/src/app/lean_app.py:1315)):

```text
After alignment, request changes such as `improve alignment`...
```

## 6. Session-Level Constraints That Reinforce Prompt Behavior

These are not prompt text, but they affect how the model can comply.

- Claude sessions are created with `max_turns = 10` and tools `["Bash", "Edit", "Read", "Write", "Replace"]` ([lean_app.py:419](/home/jmeadows17/formal-science/src/app/lean_app.py:419)).
- GPT sessions are restricted to the proof directory root via `tools = [str(DEFAULT_PROOF_PATH.parent)]` ([lean_app.py:425](/home/jmeadows17/formal-science/src/app/lean_app.py:425)).

Operationally, that means the prompt instruction to write only `FSLean/proof.lean` is backed by the tool environment.

## 7. Summary

The full prompt stack is layered:

1. `lean_prompts.py` constructs the substantive default task:
   - task statement
   - QA inputs
   - requirements
   - deliverables
   - acceptance criteria
   - final “write Lean code” instruction
2. `lean_app.py` wraps that prompt with operational file-writing instructions:
   - write directly to `FSLean/proof.lean`
   - do not write elsewhere
   - do not echo code in chat
   - stop immediately after writing
   - do not read the file back
3. `lean_app.py` also defines separate prompt templates for:
   - compile-fix turns
   - alignment evaluation turns
   - user-driven refinement turns

If you want, the next useful step is to generate a second markdown file containing the current concrete prompt text from `src/app_data/lean_prompt_data.json`, rather than just the templates.
