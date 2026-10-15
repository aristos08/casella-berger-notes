# Casella &amp; Berger — Statistical Inference: Course Notes

A measure-theoretic, proof-driven walk through *Statistical Inference* (Casella &amp; Berger, 2nd ed.), with every major result tied to a seeded simulation and real-world intuition. Written for a reader comfortable through real analysis and measure-theoretic probability.

## How to view

The notes are a self-contained static website in [`site/`](site/). No build step, no server needed:

1. Download / clone this repo.
2. Open **`site/index.html`** in any browser.

Math renders via MathJax loaded from a CDN (your browser fetches it), so an internet connection is needed the first time a page loads. From any page you can `Ctrl/Cmd+P → Save as PDF`.

Each chapter also has a **LaTeX twin** (`site/chNN/chNN.tex`) — paste into [Overleaf](https://overleaf.com) and recompile for a typeset PDF.

## Status

| Chapter | Title | Status |
|---------|-------|--------|
| 5 | Properties of a Random Sample | ✅ complete |
| 6 | Principles of Data Reduction | ⬜ skeleton |
| 7 | Point Estimation | ⬜ skeleton |
| 8 | Hypothesis Testing | ⬜ skeleton |
| 9 | Interval Estimation | ⬜ skeleton |
| 10 | Asymptotic Evaluations | ⬜ skeleton |
| 11 | ANOVA &amp; Regression | ⬜ skeleton |
| 12 | Regression Models | ⬜ skeleton |

Chapters 1–4 are assumed. Remaining chapters are scaffolded with their full section structure and filled in one at a time.

## Simulations

Every *Simulation Check* box in the notes reports real output from the seeded, pure-standard-library Python scripts in [`simulations/`](simulations/). Reproduce with:

```sh
python3 simulations/sim_ch5.py
python3 simulations/sim_ch5_part2.py
```

(No third-party dependencies — only `random`, `math`, `statistics`.)

## Layout

```
site/
├── index.html          landing page
├── assets/style.css    shared styling
└── chNN/
    ├── index.html      chapter page
    └── chNN.tex        LaTeX twin
simulations/            seeded Python simulation scripts
```
