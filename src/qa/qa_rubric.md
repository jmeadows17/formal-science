# QA Rubric

This rubric defines the ideal form of generated scientific or mathematical question-answer pairs. It is designed to be self-contained: a reader should be able to apply it without consulting other files.

The central priority is granular reasoning. Good outputs do not merely state correct conclusions; they expose the logical path from premises to result at an appropriate level of detail.

## Scope

This rubric is intended for QA generation from source reasoning in domains such as mathematics, physics, biology, and related formal scientific settings.

It is especially suited to questions that involve:

- derivations,
- proofs,
- symbolic calculations,
- structured model-based reasoning,
- interpretation of equations or assumptions,
- equivalence or implication arguments,
- parameter dependence or limiting-case analysis.

It is not limited to theorem-proof style outputs, but it does assume that the answer should make reasoning explicit rather than merely report a fact.

## Goal

Given a source reasoning passage, generate a question-answer pair that:

1. faithfully preserves the mathematical or scientific content of the source,
2. turns the source into a self-contained and meaningful question,
3. provides a correct, logically ordered answer,
4. makes nontrivial reasoning steps explicit, and
5. remains useful as downstream training data for formalization or structured reasoning.

The ideal output should read like a compact, well-composed worked solution or proof sketch, not like a summary, a chat reply, or a vague explanation.

## Required Form

Each generated item should contain exactly two fields:

- `question`
- `answer`

The `question` should define the setup, assumptions, and target conclusion.

### Required JSON Output Format

When this rubric is used to generate multiple QA pairs, the output must be a valid JSON file whose top-level value is a list of dictionaries. Each dictionary must contain exactly two string fields:

```json
[
  {
    "question": "...",
    "answer": "..."
  }
]
```

No other top-level structure is allowed. Do not return a dictionary keyed by example IDs, markdown prose, comments, metadata fields, scores, explanations, or any fields other than `question` and `answer`. If source examples have identifiers such as `core_reasoning_1`, preserve their order in the list rather than adding the identifiers as JSON keys.


The `answer` should solve the question by a coherent chain of reasoning using:

- assumptions stated in the question,
- definitions introduced in the question or answer,
- explicitly named standard facts,
- and justified algebraic, analytic, or logical transformations.

## Target QA Types

The rubric supports several common forms of scientific and mathematical QA.

### 1. Derivation or proof

The answer derives a stated formula, identity, implication, or theorem from given assumptions.

### 2. Calculation

The answer computes a quantity from stated premises, while still showing enough intermediate steps to make the computation intelligible.

### 3. Conceptual explanation grounded in formalism

The answer explains what an equation, identity, or model means, but it must remain anchored to precise mathematical or scientific statements.

### 4. Model interpretation

The answer explains what a model assumption implies, how terms in an equation function, or what a result says about the modeled system.

### 5. Multi-step implication

The answer shows how one formal statement leads to another through a short chain of justified consequences.

Across all of these types, the core standard remains the same: reasoning should be explicit and faithful.

## Question Standards

An ideal question has the following properties.

### 1. Self-contained

The question must introduce all model-specific variables, objects, assumptions, and equations needed to solve it.

Good:

> Consider a two-level atom with ground state $\ket{g}$ and excited state $\ket{e}$ coupled to a single field mode. Suppose the state evolves as
> $\ket{\Psi(t)} = \cos(\Omega t/2)\ket{e,0} - i\sin(\Omega t/2)\ket{g,1}$.
> Show that the probability of finding the atom in the excited state is $\cos^2(\Omega t/2)$.

Bad:

> Using the state above, compute the probability.

The bad version depends on omitted context.

### 2. Faithful to the source reasoning

The question should ask for the result actually supported by the source. It should not introduce a stronger theorem, a different model, or interpretive claims that go beyond the source.

Good:

> Consider a tumour with drug-sensitive population $x(t)$ and drug-resistant population $y(t)$, and let the total tumour burden be $T(t)=x(t)+y(t)$. Suppose the sensitive and resistant populations satisfy
> $\dfrac{dx}{dt} = r_x x(t)(1-x(t)-\alpha y(t)) - u(t)x(t)$
> and
> $\dfrac{dy}{dt} = r_y y(t)(1-y(t)-\beta x(t))$,
> where $u(t)$ is a drug-removal rate acting only on the sensitive population. Assume adaptive therapy keeps the total burden fixed, so that $\dfrac{dT}{dt}=0$. Show that $u(t)$ must satisfy the corresponding algebraic balance formula obtained from these equations.

Bad:

> Prove that adaptive therapy always eliminates the resistant population.

The bad version exceeds the source content.

### 3. Scientifically or mathematically meaningful

The question should preserve enough structure that the result remains interpretable in context. It should not flatten a meaningful reasoning passage into a trivial symbolic exercise unless the source itself is purely symbolic.

Good:

> Consider a population of biological types with abundances $n_j(t)$, frequencies $p_j(t)=n_j(t)/N(t)$, and trait values $a_j(t)$. Show that the rate of change of the trait mean differs from the mean intrinsic change by a covariance term bounded by $\sigma_A \sigma_r$.

Bad:

> Show that a covariance is bounded by the product of standard deviations.

The bad version strips away the modeling context that gives the statement meaning.

### 4. One coherent objective

A question should have one coherent intellectual goal. It may contain several assumptions and may require multiple intermediate steps, but the answer should culminate in a unified target.

This does not forbid tightly related subgoals. For example, a question may legitimately ask the solver to derive a formula and then interpret its meaning, if both parts belong to the same reasoning arc.

### 5. Appropriate level of background knowledge

A question should define nonstandard notation and model-specific assumptions, but it need not restate universally standard background facts from the field.

For example:

- defining a paper-specific growth model is necessary,
- defining the product rule is usually unnecessary,
- stating the concrete relation being used, such as $\oiint_S \mathbf{E}\cdot d\mathbf{A} = \iiint_V \nabla \cdot \mathbf{E}\, dV$, is usually enough without reproving it.

The question should be self-contained relative to a mathematically literate reader in the target domain, not bloated with textbook preliminaries.

## Answer Standards

An ideal answer has the following properties.

### 1. Stepwise reasoning

The answer should proceed through explicit logical steps. It should not jump directly from assumptions to the final result.

Good pattern:

1. restate the relevant definition, assumption, or governing equation,
2. substitute the given expressions,
3. simplify, rearrange, or combine terms,
4. state the concrete formula, implication, or identity used at the critical step,
5. conclude with the target statement.

### 2. Explicit use of premises

The answer should make clear where important steps come from.

Good:

> Since $T(t)=x(t)+y(t)$, differentiating gives $\dfrac{dT}{dt} = \dfrac{dx}{dt} + \dfrac{dy}{dt}$.

Bad:

> Clearly the balance equation follows.

The bad version hides the reasoning source.

### 3. Correct mathematical and scientific meaning

The answer should preserve the interpretation of the source, not just manipulate symbols.

Good:

> Because chemotherapy acts only on sensitive cells, the removal term $u(t)x(t)$ appears only in the sensitive-cell equation.

This is stronger than bare algebra because it preserves the mechanism described by the model.

### 4. Granularity matched to the step

Not every step needs the same amount of explanation.

The answer should:

- expand non-obvious logical or model-dependent steps,
- display the key mathematical formula or implication used at important steps,
- compress routine algebra once the pattern is clear,
- avoid both giant unexplained leaps and needless micromanagement of trivial arithmetic.

Good:

> At this point use the bound
> $|\operatorname{cov}(A,r)| \leq \sigma_A \sigma_r$.

Bad:

> Therefore the inequality follows by a standard argument.

Also bad:

> First subtract 0 from both sides. Then rewrite the same expression again. Then rename each symbol one at a time.

The answer should be explicit, but not mechanically verbose.

### 5. No unsupported assumptions

Every nontrivial step should be justified by:

- a definition,
- a stated assumption,
- an explicitly stated formula, implication, theorem consequence, or identity,
- or direct algebra, calculus, or logic.

Do not silently add structural assumptions that were not given.

Bad:

> Assume the Hamiltonian is diagonal in the given basis.

unless that assumption was already part of the question.

### 6. Directness and closure

The answer should terminate at the requested conclusion. It may include one brief sentence of interpretation when that interpretation follows directly from the derivation, but it should not drift into commentary, implementation notes, or conversational filler.

## Formatting Standards

### 1. Full sentences around equations

Equations should appear within explanatory prose. A strong answer is not merely a stack of formulas.

### 2. Standard and consistent notation

Use conventional notation and keep it consistent with the question.

### 2a. LaTeX math rendering requirement

All mathematical content must be written inside LaTeX math delimiters so that it renders correctly.

Use inline math delimiters `$...$` for short mathematical expressions, variables, symbols, operators, inequalities, and formulas appearing inside prose. Use display math delimiters `$$...$$` for longer equations, multi-step derivations, aligned expressions, theorem statements, or formulas that should appear on their own line.

This requirement applies to all mathematical notation, including but not limited to:

- variables such as `$x$`, `$t$`, `$n_i$`, `$\lambda_1$`, and `$\bar\theta$`;
- functions and derivatives such as `$f(x)$`, `$\partial_t n_i$`, and `$\partial_{zz}f_i$`;
- equations and inequalities such as `$\lambda_1(i)>\lambda_1(j)$`;
- operators such as `$\operatorname{Cov}_x(\theta,g)$`, `$\limsup_{t\to\infty}$`, and `$\int_0^L$`;
- sets and conditions such as `$W_\infty=W_\lambda$` and `$(\partial_z\log N)(\partial_z\bar\theta)=0$`.

Do not write bare mathematical notation outside math delimiters. For example, write `$f_i=n_i/N$`, not `f_i=n_i/N`; write `$\partial_t\bar\theta=\operatorname{Cov}_x(\theta,g)+D\partial_{zz}\bar\theta$`, not `∂_t θ̄=Cov_x(θ,g)+D∂_{zz}θ̄`.

This rule applies to both the `question` and the `answer` fields.

### 3. One coherent derivation flow

The answer should feel like one organized reasoning chain, not disconnected fragments or repeated restatements.

### 4. No assistant meta-text

Do not include phrases such as:

- "Here is the answer"
- "Let me know if you want"
- "We can now see"
- "As an AI"

### 5. Length discipline

The answer should be long enough to make the reasoning explicit, but short enough to avoid repetition, padding, or obvious algebraic overexpansion.

An answer is too short if key steps are hidden.

An answer is too long if it repeatedly restates the same fact, narrates trivial manipulations, or buries the main reasoning in unnecessary detail.

## What To Avoid

Avoid the following failure modes.

### 1. Context dependence

Bad:

> Using the result from the previous question, show that ...

Each generated question must stand on its own.

### 2. Restatement instead of reasoning

Bad:

> The required result is true by the given equations.

This does not train derivational or explanatory reasoning.

### 3. Overcompression

Bad:

> Substitute and simplify to obtain the claim.

If the source contains a meaningful chain of reasoning, the answer should preserve it.

### 4. Overformalized but unnatural phrasing

Bad:

> Let there exist an entity whose associated state variable, if interpreted in the standard manner, yields the desired phenomenon.

A question can be precise without becoming stiff or unreadable.

### 5. Tautological questions

Bad:

> Show that the formula derived from these assumptions is the formula stated above.

The target should be mathematically meaningful, not a disguised restatement.

### 6. Domain drift

Bad:

> This suggests a treatment strategy that should outperform all alternatives.

Do not inflate a derivation into an unsupported scientific, engineering, or policy conclusion.

### 7. Hidden theorem dependence

Bad:

> The result follows from standard theory.

If a theorem is doing real work, state the concrete implication or formula that is being used. Naming the theorem may be helpful, but it is not a substitute for making the operative mathematical content explicit.

## Physics Example

### Good Physics Question

> Consider a nonrelativistic particle on a line with momentum-space wavefunction $\phi(p) = \braket{p \mid \Psi}$. Define $g(p)=p\phi(p)$ and assume the momentum expectation is zero. Show that the momentum variance satisfies $\sigma_p^2 = \braket{g \mid g}$.

Why this is good:

- It defines the objects.
- It states the needed assumption.
- It asks for one precise conclusion.

### Good Physics Answer Shape

> Start from the definition of variance.
> Use the zero-expectation assumption to simplify it.
> Rewrite $p^2 |\phi(p)|^2$ as $|g(p)|^2$.
> Use the definition of the inner product $\braket{g \mid g}$.
> Conclude that $\sigma_p^2 = \braket{g \mid g}$.

Why this is good:

- Each step is explicit.
- The answer preserves both the calculation and the physical meaning.
- The granularity is appropriate: it explains the conceptual steps without overexplaining trivial algebra.

### Bad Physics Version

Question:

> Show the result for `g`.

Answer:

> This is immediate from the definition.

Why this is bad:

- The question is not self-contained.
- The answer omits the derivation.

## Biology Example

### Good Biology Question

> Consider biological types with abundances $n_j(t)$, total size $N(t)=\sum_j n_j(t)$, frequencies $p_j(t)=n_j(t)/N(t)$, growth rates $r_j(t)=\dot n_j(t)/n_j(t)$, and trait values $a_j(t)$. Define the population mean $\langle A\rangle=\sum_j p_j(t)a_j(t)$ and covariance $\operatorname{cov}(A,r)=\langle Ar\rangle-\langle A\rangle\langle r\rangle$. Assume $\dot n_j(t)=r_j(t)n_j(t)$. Show that
> $\left|\dfrac{d\langle A\rangle}{dt} - \langle \dot A\rangle\right| \leq \sigma_A \sigma_r$.

Why this is good:

- It carries the biological setup into the question.
- It introduces every symbol needed in the derivation.
- It targets one meaningful bound.

### Good Biology Answer Shape

> Differentiate the mean using the product rule.
> Separate the intrinsic-change term $\langle \dot A\rangle$ from the frequency-reweighting term.
> Use the abundance equation to derive $\dot p_j(t)=p_j(t)(r_j(t)-\langle r\rangle)$.
> Substitute to obtain the covariance expression.
> Use the bound $|\operatorname{cov}(A,r)| \leq \sigma_A \sigma_r$ to conclude the inequality.

Why this is good:

- It tracks the biological mechanism of changing type frequencies.
- It gives the key logical steps rather than hiding them behind a theorem name alone.
- It ends at the stated inequality rather than wandering into general discussion.

### Bad Biology Version

Question:

> Prove a useful inequality about evolution.

Answer:

> The statement follows from Price's theorem and standard bounds.

Why this is bad:

- The question is vague.
- The answer hides the derivation.
- The biological quantities are not defined.

## Mathematics Example

### Good Mathematics Question

> Let $f$ be differentiable on an interval and suppose $f'(x)=0$ for every $x$ in that interval. Show that $f$ is constant on the interval.

Why this is good:

- It is self-contained relative to standard calculus background.
- It states a precise implication target.
- It does not force unnecessary scientific context where none is needed.

### Good Mathematics Answer Shape

> Take any two points in the interval.
> Introduce the relation
> $\dfrac{f(b)-f(a)}{b-a} = f'(c)$
> for some $c$ between them.
> Use the hypothesis $f'(x)=0$ to show the average rate of change is zero.
> Conclude that the function values at the two points are equal.
> Since the points were arbitrary, $f$ is constant.

Why this is good:

- The proof has explicit logical structure.
- The key mathematical implication is stated explicitly.
- The level of detail is appropriate for a standard mathematical argument.

### Bad Mathematics Version

Question:

> What can you say about $f$?

Answer:

> It is obvious that `f` must be constant.

Why this is bad:

- The question is underspecified.
- The answer suppresses the central theorem and the reasoning chain.

## Review Criteria

When evaluating a generated QA pair, assess the following dimensions.

### Evaluation Strictness

Apply this rubric extremely harshly when judging QA alignment and correctness.

- Do not give credit for answers that merely sound plausible.
- Do not infer missing assumptions, missing algebra, missing definitions, or missing theorem conditions in the model's favor.
- Do not overlook small mathematical, logical, scientific, or notation errors if they affect correctness, faithfulness, or self-containment.
- Do not treat approximate paraphrase as acceptable when the source reasoning supports a sharper claim, a narrower claim, or a differently qualified claim.
- Do not reward answers that state the right final conclusion if the intermediate reasoning is invalid, incomplete, circular, or materially under-justified.
- Do not excuse hidden dependence on context, unstated symbols, omitted premises, or theorem-name-dropping without the operative mathematical content.

If a QA pair has any substantive flaw in correctness, faithfulness, self-containment, logical validity, or formal clarity, evaluate it as misaligned with the rubric and require revision. Borderline cases should be scored downward, not upward.

### 1. Correctness

Are the mathematical or scientific statements true, and is the reasoning valid?

### 2. Faithfulness

Does the pair preserve the actual content of the source reasoning without adding unsupported claims?

### 3. Self-containment

Can the question be understood and solved without relying on hidden external context?

### 4. Clarity

Is the reasoning organized, readable, and easy to follow?

### 5. Appropriate rigor

Does the answer explain nontrivial steps at the right level of detail?

### 6. Meaningfulness

Does the question preserve the scientific or mathematical point of the source rather than collapsing it into a trivial algebra exercise?

### 7. Concision

Is the answer free of fluff, repetition, and needless overexpansion?

## Ideal Output Checklist

Before accepting a generated QA pair, check:

- Is the question self-contained?
- Is the question faithful to the source reasoning?
- Does the answer derive or justify the result step by step?
- Are important steps justified explicitly?
- Is the granularity of the answer appropriate to the difficulty of the steps?
- Is the notation consistent and standard?
- Is every mathematical expression enclosed in `$...$` or `$$...$$` for LaTeX rendering?
- Does the answer stop at the requested conclusion or a directly supported interpretation?
- Is there any filler, meta-commentary, hallucinated assumption, or hidden theorem dependence?

If any answer to these checks is "no", the QA pair should be revised.
When uncertain, reject rather than accept. Acceptance requires clear compliance with the rubric, not approximate compliance.
