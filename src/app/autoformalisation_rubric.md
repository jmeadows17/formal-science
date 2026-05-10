# Minimal Research-Grade Lean `formal_answer` Rubric

A dataset item has the form:

```json
{
  "question": "<informal question>",
  "answer": "<informal answer>",
  "formal_answer": "<Lean code>"
}
```

The `formal_answer` is accepted only if it contains a self-contained Lean development proving the mathematical claim made by the informal answer.

## 1. Required Lean Structure

The Lean code must contain exactly one final theorem named:

```lean
theorem target_theorem ... : ... := by
  ...
```

It may contain imports, definitions, and supporting lemmas.

It must contain no:

```text
sorry
admit
unfilled goals
undeclared axioms
hidden proof holes
```

Only the standard Lean/mathlib axioms are allowed by default:

```text
propext
Classical.choice
Quot.sound
```

Any other external fact must be explicitly named, stated, and justified as an independent theorem.

---

## 2. Statement Fidelity Gate

The statement of `target_theorem` must faithfully formalize the informal answer.

Reject if it silently changes:

```text
domain
objects
constants
quantifiers
assumptions
conclusion
equality/inequality strength
limit/asymptotic meaning
topology, measure, norm, order, or regularity setting
```

Examples of automatic rejection:

```text
“for all” becomes “there exists”
an equality becomes one inequality
a limit becomes a finite approximation
a continuous theorem becomes a finite/discrete surrogate
a sharp bound loses the equality or extremal case
```

After writing `target_theorem`, its plain-English re-informalization, including all assumptions, must still be an acceptable answer to the original question.

---

## 3. Object Fidelity Gate

Every central mathematical object in the informal claim must be represented at the correct level in Lean.

For example:

```text
integral              → actual integral or named theorem about that integral
derivative/Laplacian  → actual operator or faithful weak/distributional version
ODE/flow              → state space, vector field, solution/flow
probability           → probability space or finite weights with normalization
variance/covariance   → explicit definitions or imported standard definitions
asymptotic O(δ)       → explicit constants and small-δ quantifiers
Dirac delta           → measure/distribution-level object
path integral/action  → faithful object or precise bridge theorem
boundary condition    → predicate plus proof of consequences
```

Reject or cap at non-research-grade if a central object is replaced by a placeholder, arbitrary predicate, coefficient record, or definition that already encodes the desired conclusion.

---

## 4. Central Burden Discharge

Before judging the proof, identify the central mathematical burdens: the substantive steps without which the informal answer would not be convincing.

Each central burden must be classified in a Lean comment as exactly one of:

```text
proved_in_file
imported_from_mathlib
external_domain_fact
explicit_input_from_question
```

Reject if any central burden is:

```text
not formalized
encoded in a definition
assumed as a hypothesis
hidden inside a theorem-shaped external fact
```

Required comment shape:

```lean
/-
Scope: Unconditional / Research-grade conditional / Restricted / Not acceptable

Central burden discharge:
1. <burden>.
   Status: proved_in_file / imported_from_mathlib / external_domain_fact / explicit_input_from_question.
   Declarations: <names>.
   Justification: <one sentence>.

External domain facts:
- <name>: <standard theorem, field, hypotheses, and why it is not equivalent to target_theorem>.

Object-fidelity notes:
- <central objects and how they are represented>.

Self-audit:
- Surrogate objects removed: ...
- Theorem-shaped assumptions removed: ...
- Remaining external facts: ...
- Why each external fact is independent of target_theorem: ...
-/
```

---

## 5. No Theorem-Shaped Assumptions

Reject if any hypothesis assumes the result, a stronger result, or a central step the answer is supposed to prove.

A hypothesis is invalid if it contains the main:

```text
equality
inequality
limit
convergence statement
spectrum
basin inclusion
path-integral identity
effective action formula
variance decomposition
hitting-time estimate
asymptotic bound
```

Allowed assumptions are only:

```text
explicit input data from the question
technical conditions such as smoothness, measurability, positivity, nonzero assumptions
mathlib facts
proved supporting lemmas
standard independent external domain theorems
```

Definitions also must not bake in the desired conclusion. If the final theorem is mostly proved by `rfl`, `simp`, or unfolding a definition that encodes the answer, reject.

---

## 6. External Theorem Admissibility

An external theorem is allowed only if it is:

```text
explicitly named
precisely stated
standard in the relevant field
reusable across many problems
stated over the same faithful objects as target_theorem
not equivalent to the target conclusion
documented in a Lean comment
```

Reject external facts that are bespoke to the dataset item, collapse several central burdens into one assumption, or directly assert the final theorem.

---

## 7. Quantitative, Limit, and Sharpness Claims

Any claim involving constants, bounds, rates, limits, asymptotics, or sharpness must be formalized explicitly.

For example, informal:

```text
Eδ(T) ≤ C_T δ for sufficiently small δ
```

should become something like:

```lean
∃ C_T > 0, ∃ δ₀ > 0,
  ∀ δ, 0 < δ → δ ≤ δ₀ → E δ T ≤ C_T * δ
```

Opaque predicates such as `IsBigOClaim`, `HasErrorBound`, or `SharpBound` do not count unless they unfold to explicit quantified content or are connected to a standard imported theorem.

Sharpness requires both:

```text
the bound
an equality case, extremizer, or ε-approximation
```

---

## 8. Scope Labels

Every `formal_answer` must classify the final theorem as exactly one of:

```text
Unconditional
Research-grade conditional
Restricted
Not acceptable
```

Only these count as research-grade:

```text
Unconditional
Research-grade conditional
```

A theorem is `Research-grade conditional` only if all non-mathlib assumptions are named, standard, independent external theorems.

A theorem is `Not acceptable` if it is toy, surrogate, weakened, assumption-loaded, or merely a packaging lemma.

---

## 9. Acceptance Scores

Assign four scores from 0 to 5:

```text
1. Target theorem fidelity
2. Object fidelity
3. Burden discharge
4. Assumption hygiene
```

Accept only if:

```text
each score ≥ 4
no rejection rule applies
overall score = minimum of the four scores
```

Guidance:

```text
5 = fully faithful / fully discharged / clean
4 = minor harmless technical assumptions
3 = plausible but incomplete or somewhat broad
2 = weakened, surrogate, or one central burden assumed
1 = multiple central failures
0 = different theorem or target conclusion assumed
```

Score caps:

```text
central burden assumed                          → max 2
central object replaced by surrogate            → max 2
target theorem silently weakened                → max 2
external fact unnamed or target-shaped          → max 2
proof follows by unfolding answer-encoding defs → max 2
packaging/product-membership lemma only         → max 2
```

---

## 10. Final Rejection Checklist

Reject the `formal_answer` if any answer is yes:

```text
1. Does it use sorry, admit, hidden holes, or undeclared axioms?
2. Does target_theorem fail to match the informal answer?
3. Does re-informalization fail to match the original claim?
4. Are central objects missing or replaced by surrogates?
5. Is any central burden assumed instead of proved/imported/externally justified?
6. Does any hypothesis contain the main conclusion or a central conclusion?
7. Do definitions encode what should have been proved?
8. Are external facts unnamed, nonstandard, bespoke, or equivalent to target_theorem?
9. Are asymptotic, limit, quantitative, or sharpness claims made opaque?
10. Is the proof only a toy model, special case, or packaging lemma without clear restriction?
```

---

## Minimal Acceptance Criterion

A `formal_answer` is research-grade exactly when:

```text
target_theorem states the same claim as the informal answer;
all central objects are faithfully formalized;
all central burdens are proved, imported from mathlib, explicit input, or standard independent external facts;
no theorem-shaped assumptions or answer-encoding definitions are used;
quantitative and limit claims are explicit;
external facts are named, reusable, and not equivalent to the target;
the scope is Unconditional or Research-grade conditional;
all four scores are at least 4.
```

Anything less risks accepting polished Lean code that proves a surrogate theorem rather than the original mathematical claim.
