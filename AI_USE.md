# AI_USE.md — AI-Assisted Learning Log
### Entry 1 — Bias–Variance Decomposition

| Field | Content |
|-------|---------|
| **Week/date** | W01–W04 / 2026-09-26 |
| **Learning question** | Algebraic expansion steps for the Bias–Variance decomposition of expected MSE? |
| **Pre-AI evidence** | `exercises/release-baseline-w01-w04.pdf` (Q2(a)) |
| **AI tool** | Gemini |
| **Prompt purpose** | Socratic hint on cross-term simplification |
| **Hint received** | Independence: $E[\epsilon \hat{f}(x)] = E[\epsilon]E[\hat{f}(x)] = 0$ |
| **Verification source** | CO3117 lecture notes; Murphy (2022) §4.1 |
| **What changed** | Misconception: treated cross-term as nonzero. Correction: $\epsilon \perp \hat{f}(x)$. |
| **Closed-book reproduction** | Yes — 2026-09-29, `exercises/release-baseline-w01-w04.pdf` |

### Entry 2 — CART Continuous Attributes

| Field | Content |
|-------|---------|
| **Week/date** | W01–W04 / 2026-09-26 |
| **Learning question** | How does CART convert continuous features into binary splits without infinite thresholds? |
| **Pre-AI evidence** | `exercises/release-baseline-w01-w04.pdf` (Q4(a)) |
| **AI tool** | Gemini |
| **Prompt purpose** | Conceptual step-by-step (no code) |
| **Hint received** | sort → midpoints $(v_i + v_{i+1})/2$ → maximize $\Delta G$ |
| **Verification source** | CO3117 *Decision Trees* Slide 56 |
| **What changed** | Understood $n-1$ candidate thresholds (midpoints only). |
| **Closed-book reproduction** | Yes — 2026-09-29, `exercises/release-baseline-w01-w04.pdf` |

