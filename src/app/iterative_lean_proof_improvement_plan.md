# Iterative Lean Proof Improvement Plan Using Rubric v5

## Objective

Use the research-grade `formal_answer` rubric v5 to iteratively improve generated Lean proofs against an informal QA pair. The goal is to ensure that the final `target_theorem` faithfully formalizes the informal answer, discharges the central proof burdens, avoids surrogate objects, and contains no theorem-shaped assumptions.

The critic artifact for each iteration is `critic_output.md`. It should be a concrete rubric-scored audit of the current `FSLean/proof.lean` against the input QA pair and `autoformalisation_rubric.md`, not a generic alignment note.

## Core Principle

Do not optimize a single average Likert score. Use a gated, adversarial loop where the overall score is the minimum of four sub-scores:

```text
overall_score = min(
  Target theorem fidelity,
  Object fidelity,
  Burden discharge,
  Assumption hygiene
)
```

A proof is accepted only if every sub-score is at least 4 and no automatic rejection or score cap applies.

## Recommended Three-Agent Loop

### Session Boundary Rule

Use a fresh LLM session for each major stage.

- Writer stage: new session writes `FSLean/proof.lean`.
- Critic stage: new session reads the QA pair, current `FSLean/proof.lean`, and `autoformalisation_rubric.md`, then writes its report to `critic_output.md`.
- Repair stage: new session reads the QA pair, current `FSLean/proof.lean`, and `critic_output.md`, then rewrites `FSLean/proof.lean`.

Do not let the critic share the writer's session, and do not let the repair model share the critic's session. The goal is to reduce anchoring, preserve adversarial pressure, and force each stage to reason from artifacts rather than conversational momentum.

### 1. Critic / Auditor

The critic evaluates the current proof before any repair attempt.

Responsibilities:

- Re-informalize `target_theorem` in plain English, including all assumptions.
- Extract 3–10 central burdens from the informal QA pair.
- Identify surrogate objects, theorem-shaped assumptions, and definitions that encode conclusions.
- Apply hard-zero conditions and score caps.
- Produce a structured failure report with concrete scores rather than only a scalar score.

Why re-informalize:

- Lean can prove a precise theorem that is not actually the theorem asked for by the QA pair.
- Re-informalization exposes silent weakening of quantifiers, domains, conclusions, asymptotic meaning, and hidden theorem-shaped assumptions.
- If the plain-English restatement would not be accepted as an answer to the original question, the proof fails before repair prioritization even begins.

Why extract 3–10 central burdens:

- A single holistic alignment judgment is too lossy for repair.
- Central burdens identify the real mathematical obligations the informal answer depends on.
- They let the critic distinguish between burdens that are proved, imported from mathlib, delegated to a standard external theorem, merely assumed, or hidden in definitions.
- They prevent fake progress where the proof becomes more polished without discharging the main argument.

### 2. Planner

The planner decides what kind of repair is realistic.

Responsibilities:

- Decide whether the proof should target `Unconditional`, `Research-grade conditional`, `Restricted`, or `Not acceptable`.
- Decide which burdens should be proved in-file, imported from mathlib, treated as explicit input, or represented by standard external domain theorems.
- Reject repair paths that merely rename central burdens as external facts.
- Specify the exact changes the repair model must make.

### 3. Lean Repair Model

The repair model rewrites the proof under hard constraints.

Responsibilities:

- Preserve the informal theorem’s domain, objects, constants, quantifiers, and conclusion.
- Remove theorem-shaped assumptions.
- Replace surrogates with faithful objects or precise bridge theorems.
- Add the required rubric comments: scope, input data, central burden discharge, external theorem independence, object-fidelity notes, and self-audit.

## Compile-Fix Policy

Do not begin the expensive compile-fix loop immediately after first draft generation.

Instead:

1. Run a coarse semantic/rubric screen on the initial proof.
2. Start the compile-fix loop only once the proof reaches a coarse rubric score of at least 4.0 on every sub-score, with no hard-zero condition and no active score cap.
3. If any sub-score is below 4.0, repair statement fidelity, object fidelity, burden discharge, and assumption hygiene first.

Rationale:

- A proof below 4.0 on any core rubric dimension is usually targeting the wrong theorem, using surrogates, or assuming central burdens.
- Compile-fixing such a proof wastes effort on syntax and local proof mechanics before semantic adequacy is established.
- Once alignment reaches 4.0, compile-fix becomes worthwhile because the theorem is plausible enough that mechanical repair exposes useful structural details rather than polishing the wrong object.

Operationally, compile-fix is a cleanup and structure-revealing step, not the primary acceptance gate.

## Iteration Stages

Use staged targets rather than asking the model to produce a research-grade proof in one step.

```text
Stage 1: Writer session produces an initial `proof.lean`.
Stage 2: Critic session scores coarse rubric alignment and exposes theorem shape, assumptions, and central burdens.
Stage 3: If any coarse sub-score < 4.0, or any hard-zero/cap applies, repair semantically in a fresh repair session and re-critic.
Stage 4: Once every coarse sub-score ≥ 4.0 and no hard-zero/cap applies, run compile-fix on `proof.lean`.
Stage 5: Critic session runs full rubric review and writes `critic_output.md`.
Stage 6: Fresh repair session rewrites the proof from the QA pair plus `critic_output.md`.
Stage 7: Add successor theorem tests.
Stage 8: Pass final rubric v5 scoring.
```

## Approval Semantics

In the Lean app, `Approve` should mean final acceptance of the current proof for the current QA batch, not merely "save the current draft and move on."

The user should click `Approve` only when the critic has determined that the proof appropriately satisfies rubric v5 for the current batch.

That means all of the following must hold:

- `Decision = accept` in `critic_output.md`
- `Target theorem fidelity ≥ 4.0`
- `Object fidelity ≥ 4.0`
- `Burden discharge ≥ 4.0`
- `Assumption hygiene ≥ 4.0`
- `overall_score = min(...) ≥ 4.0`
- no hard-zero condition applies
- no active score cap applies
- the re-informalization matches the QA pair
- successor tests, if run at this stage, do not expose a semantic mismatch

If those conditions are not met, the correct action is repair, re-scope, regenerate, or skip, not approval.

Operationally:

- `Regenerate` should request a fresh batch proof attempt.
- free-form refinement should revise the current `FSLean/proof.lean` against the latest critic output.
- `Approve` should be reserved for proofs that have already cleared the rubric gate.

## Approved Output Persistence

Approval should update two different artifacts with different roles.

1. Raw approved batch proof artifact.
   Keep the batch-level Lean source for traceability, reproducibility, and later re-extraction.

2. Batch-preserving approved formal dataset.
   Also maintain a new JSON file that mirrors the batch structure of `qa_data.json`, but adds a `formal_answer` field to each QA item once that item's proof has been approved.

Recommended file roles:

- `qa_data.json`: source batched QA pairs
- `structured_proofs.json`: approved batch-level Lean source blobs and prompt metadata
- `approved_formal_batches.json`: new batch-preserving approved dataset mirroring `qa_data.json`
- flattened final dataset: derived later from the batch-preserving approved dataset when needed

Recommended shape for `approved_formal_batches.json`:

```json
[
  [
    {
      "question": "...",
      "answer": "...",
      "formal_answer": "..."
    },
    {
      "question": "...",
      "answer": "...",
      "formal_answer": "..."
    }
  ],
  [
    {
      "question": "...",
      "answer": "...",
      "formal_answer": "..."
    }
  ]
]
```

This file should preserve batch boundaries exactly as in `qa_data.json`.

Approval-time behavior:

- When a batch proof is approved, split the approved Lean source into the per-QA `formal_answer` fragments aligned to that batch.
- Build one approved batch entry that preserves the original QA ordering.
- Append that batch entry to `approved_formal_batches.json`.
- Autosave after each approval so the file incrementally reflects all approved batches so far.

Why store this file:

- It keeps the reviewed formal outputs aligned with the original QA batching rather than only as one large Lean blob.
- It gives downstream steps a stable source of approved `(question, answer, formal_answer)` triplets without destroying batch structure.
- It separates "raw approved proof artifact" from "dataset-ready approved formal answers."

## `critic_output.md` Contract

`critic_output.md` should compare the current `FSLean/proof.lean` directly against:

- the informal `question`
- the informal `answer`
- `autoformalisation_rubric.md`

It should not be a free-form review. It should be a concrete audit the repair stage can act on mechanically.

Scoring rules:

- Use 1 decimal place for every score.
- Use the rubric's four scoring dimensions directly.
- Set `overall_score` to the minimum of the four sub-scores, never the average.
- Allow scores from `0.0` to `5.0`. Do not force the floor to `1.0`, because rubric v5 includes hard-zero conditions.
- If a hard-zero condition applies, state it explicitly and assign `overall_score = 0.0`.
- If a score cap applies, state the cap explicitly and ensure no affected score exceeds the cap.

Required top-level fields in `critic_output.md`:

- `Decision`: `accept` / `repair` / `re-scope` / `reject`
- `Scope label`: `Unconditional` / `Research-grade conditional` / `Restricted` / `Not acceptable`
- `Target theorem fidelity`: `_._ / 5`
- `Object fidelity`: `_._ / 5`
- `Burden discharge`: `_._ / 5`
- `Assumption hygiene`: `_._ / 5`
- `Overall score`: `min = _._ / 5`
- `Hard-zero conditions applied`
- `Score caps applied`
- `Re-informalization`
- `Central burdens`
- `Surrogate objects`
- `Theorem-shaped assumptions`
- `Definitions encoding conclusions`
- `External facts requiring replacement or stronger justification`
- `Required repairs`

The critic should justify each numerical score with short evidence tied to the rubric. The repair stage should be able to read the report and know exactly which theorem statement, object model, burdens, and assumptions must change.

## Structured Failure Report Template

Each repair cycle should start from a report like this:

```text
Decision: accept / repair / re-scope / reject
Scope label: Unconditional / Research-grade conditional / Restricted / Not acceptable

Target theorem fidelity: _._ / 5
Object fidelity: _._ / 5
Burden discharge: _._ / 5
Assumption hygiene: _._ / 5
Overall score: min = _._ / 5

Hard caps or hard-zero conditions applied:
- ...

Re-informalization:
- Plain-English restatement of exactly what `target_theorem` proves, including all nontrivial assumptions.
- Short verdict: would this restatement count as a correct answer to the original QA pair?

Central burdens:
1. ...
   Current status: proved_in_file / imported_from_mathlib / external_domain_fact / explicit_input / not_formalized
   Failure: ...
   Required repair: ...

Surrogate objects:
- ...

Theorem-shaped assumptions:
- ...

Definitions encoding conclusions:
- ...

External facts requiring replacement or stronger justification:
- ...

Required repairs:
1. ...
2. ...
3. ...
```

Avoid prompts such as “Your proof scored 2/5; improve it.” They are too lossy and encourage superficial changes.

## Repair Prompt Template

Use this prompt for the Lean repair model:

```text
Revise the Lean proof under rubric v5.

Hard constraints:
1. Do not introduce any new hypothesis that contains a central conclusion of the informal answer.
2. Do not define an object so that the desired theorem follows by unfolding.
3. Do not replace an informal object with a surrogate. If the informal answer uses an integral, derivative, ODE flow, distribution, path integral, hitting time, or asymptotic bound, formalize that object faithfully or use a precise named external theorem.
4. For every central burden, either prove it in Lean, import it from mathlib, mark it as explicit input from the question, or state a standard independently justified external theorem.
5. If a central burden cannot be discharged, downgrade the scope to `Restricted` or `Not acceptable`; do not label it `Research-grade conditional`.
6. Include a central burden discharge table in Lean comments.
7. Include an external theorem independence note for every external theorem.
8. Ensure `target_theorem` proves the same claim as the informal answer, with no weakened domain, quantifiers, topology, equality, limit, or asymptotic meaning.
```

## Successor Theorem Tests

For each QA pair, create 2–5 small successor statements that should follow from a faithful `target_theorem`, but not from common surrogate versions.

Examples:

- For ODE claims, successor tests should use the actual vector field, not only a basin predicate.
- For distributional identities, successor tests should apply the result to test functions.
- For effective-action claims, successor tests should use the action functional or a precise bridge theorem, not only coefficient equality.
- For boundary-condition claims, successor tests should start from the boundary-condition predicate and derive spectral consequences.

Successor tests act as integration tests for semantic fidelity.

## Adversarial Mutation Tests

After repair, try small mutations:

- Weaken a hypothesis.
- Strengthen a conclusion.
- Replace a surrogate with the faithful object.
- Instantiate degenerate or boundary cases.
- Test whether the theorem still appears valid for nonsensical objects.

If a theorem remains provable under nonsensical abstractions, it is probably too abstract or surrogate-based.

## Triage for the Original Five Proofs

| Pair | Best Target | Recommended Policy |
|---|---|---|
| 1. QFT compactification / BF / monopoles | Conditional or Restricted | Do not chase a fully self-contained proof; require precise external QFT bridge theorems. |
| 2. Product membership in `Σ_QM × Σ_2d` | Assembly-only conditional | Accept only if explicitly scoped as assembly; otherwise reject as packaging. |
| 3. Lotka–Volterra fixed dose | Research-grade conditional | Best candidate for serious repair; use standard LV theorems and faithful CAT dynamics. |
| 4. Intermittent adaptive therapy asymptotics | Conditional with explicit constants | Replace opaque `O(δ)` predicates and theorem-shaped estimates with quantified bounds. |
| 5. Coulomb implies Gauss | Research-grade conditional | Use a distributional formulation and a standard singular-kernel identity. |

## Stopping Criteria

### Accept

Stop and accept when:

- `Approve` would be semantically correct at this point.
- Re-informalization matches the QA pair.
- All central burdens are discharged.
- No surrogate objects remain.
- No theorem-shaped assumptions remain.
- Successor tests pass.
- All four rubric sub-scores meet the acceptance threshold.
- Any compile-fix that was run occurred only after every coarse rubric sub-score reached at least 4.0 and no hard-zero/cap remained active.
- The approved batch has been persisted both as a raw approved batch proof artifact and as a batch-preserving approved formal dataset entry with `question`, `answer`, and `formal_answer`.

### Re-scope

Re-scope when:

- Object fidelity requires unavailable libraries or extensive new infrastructure.
- The result can only be proved using broad external theorems that carry most of the central burden.
- The theorem is realistically only a special case.

### Reject

Reject when:

- The model repeatedly assumes the main result under new names.
- The target theorem remains a packaging lemma while the QA asks for substantive proof.
- A central object cannot be represented faithfully.
- Re-informalization still does not match the informal answer.

## Most Important Diagnostic Question

For every iteration, ask:

> What would still be missing if this theorem were taken at face value?

If the answer includes a central burden from the informal proof, the proof is not yet research-grade.
