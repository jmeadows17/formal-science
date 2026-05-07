# Minimal Rubric: Extracting Self-Contained Mathematical Reasoning from Scientific Papers

## Purpose

Given a scientific paper containing mathematical reasoning (derivations, models, inequalities, asymptotics, identities, or formal relationships), produce a UTF-8 JSON object containing:

- `paper_title`
- `core_reasoning_1`, `core_reasoning_2`, ..., `core_reasoning_N`

Each `core_reasoning_k` must be a fully self-contained LaTeX derivation of a non-trivial mathematical claim stated, implied, or directly supported by the paper.

Every derivation must be understandable without consulting:

- the source paper,
- external references,
- or any other `core_reasoning_k`.

The output must remain mathematically rigorous, machine-parseable, and faithful to the paper.

---

## 1. Output Format

The output must be a single UTF-8 JSON object.

The object must contain:

- `paper_title`
- contiguous `core_reasoning_k` fields beginning at `1`

No additional top-level keys are allowed.

### `paper_title`

Format:

```text
"<Title> (<First-author surname>, <year>, <venue>)"
```

### `core_reasoning_k`

Each reasoning entry must:

- be a single JSON string containing escaped LaTeX,
- use `\n` for line breaks,
- compile successfully with only:

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}

\begin{document}
...
\end{document}
```

No additional packages or custom commands may be assumed.

---

## 2. Required Structure of Each `core_reasoning_k`

Each entry must contain the following sections in order.

---

### 2.1 Claim

A precise falsifiable mathematical statement.

The claim must state a concrete result such as:

- an equation,
- inequality,
- asymptotic relation,
- identity,
- necessity/sufficiency condition,
- existence statement,
- uniqueness statement,
- conservation law,
- or formal relationship.

Topical summaries are forbidden.

---

### 2.2 Premises

A labelled LaTeX premises block using contiguous labels:

```latex
(P1), (P2), ...
```

Requirements:

- every premise must be explicitly stated,
- every fact used later in the derivation must either:
  - appear as a premise,
  - or be derived inline,
- standard mathematical identities used later must appear symbolically as premises unless derived explicitly,
- all assumptions, domain restrictions, and limit conditions must be stated explicitly.

---

### 2.3 Derivation

A step-by-step symbolic derivation using `align` environments.

Requirements:

- one logical or algebraic operation per step,
- nearly every substantive step must include an explicit justification,
- justifications must identify the premise, algebraic rule, or transformation used,
- new definitions must use `:=`,
- substitutions and limiting procedures must be shown explicitly,
- branches or cases must be clearly labelled,
- symbolic reasoning must remain explicit throughout.

The derivation must be sufficiently detailed for reconstruction without external references.

---

### 2.4 Final Result

Each derivation must end with exactly one boxed result:

```latex
\boxed{...}
```

The boxed result must directly correspond to the claim statement.

---

### 2.5 Closure Marker

Each derivation must end with:

```latex
\hfill$\blacksquare$
```

---

## 3. Self-Containment Rules

- No references to other `core_reasoning_k` entries.
- No appeals to unnamed “standard results”.
- No named mechanisms, regimes, or theories without explicit symbolic definition.
- No omitted algebraic reductions or hand-waved limiting arguments.
- Any mathematical identity used must either:
  - appear explicitly as a premise,
  - or be derived within the proof.

---

## 4. Mathematical Rigor Rules

- All reasoning must be symbolic and explicit.
- Domain restrictions and limit assumptions must be stated clearly.
- Approximations must show discarded terms when relevant.
- Inequalities must justify each comparison step.
- Equality and boundary cases should be stated where relevant.
- Numerical conclusions must be derived symbolically before evaluation.
- Definitions introduced with `:=` must not be treated as derived equalities.

---

## 5. Faithfulness Rules

- Premises must originate from:
  - the paper,
  - assumptions explicitly used by the paper,
  - or clearly-labelled standard mathematical facts.

- The derivation must preserve the paper’s notation and conventions consistently.

- Boxed conclusions must be stated by, or directly implied by, the paper.

- No unsupported assumptions or externally introduced claims are allowed.

---

## 6. Selection Rules

Choose a set of non-trivial derivations that collectively represent the paper’s major mathematical reasoning.

Prefer coverage across distinct forms of reasoning such as:

- core derivations,
- asymptotic analyses,
- invariants,
- inequalities,
- boundary conditions,
- generalizations,
- validity conditions,
- parameter relationships,
- or mechanistic consequences.

Avoid:

- trivial one-step derivations,
- duplicated claims,
- superficial paraphrases of the same result.

Each selected claim should require meaningful multi-step reasoning.

---

## 7. Style Rules

- Use only ASCII-compatible LaTeX commands.
- Use explicit symbolic notation consistently.
- Keep derivations readable and structurally clear.
- Avoid prose explanations inside `align` environments except for formal justifications.

