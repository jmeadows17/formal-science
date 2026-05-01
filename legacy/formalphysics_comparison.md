# FormalPhysics vs FormalPhysics_v2 — Minimal Comparison

## Summary Table

| Metric | FormalPhysics | FormalPhysics_v2 |
|---|---:|---:|
| Examples | 200 | 215 |
| Mean Score | 3.84 / 5 | **4.50 / 5** |
| Score ≥ 4 (%) | 69% | **~90%+** |
| Severe Failures | Present (notably 191–200) | Rare |
| Semantic Fidelity | Moderate, inconsistent | **High, consistent** |
| Logical Preservation | Mixed | **Strong** |
| Depth Preservation | Often reduced via assumptions | **Generally preserved** |
| Main Weakness | Semantic drift, tautologies, mismatches | Premise-loading (assumed theorems) |

## Key Differences

- **Overall Quality**: FormalPhysics_v2 is significantly stronger (+0.66 Likert).
- **Consistency**: v2 removes large failure clusters seen in original.
- **Reasoning Preservation**: v2 more often preserves full derivations (not just results).
- **Failure Mode Shift**:
  - Original → drift, collapse, mismatch
  - v2 → assumption-heavy but semantically aligned
- **Scientific Faithfulness**: v2 maintains operator/PDE structure much better.

## Bottom Line

- **FormalPhysics**: Good but unreliable; contains critical integrity issues.
- **FormalPhysics_v2**: High-quality dataset; strong semantic alignment with minor analytical shortcuts.
