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
| 6 | Principles of Data Reduction | ✅ complete |
| 7 | Point Estimation | ✅ complete |
| 8 | Hypothesis Testing | ✅ complete |
| 9 | Interval Estimation | ✅ complete |
| 10 | Asymptotic Evaluations | ⬜ skeleton |
| 11 | ANOVA &amp; Regression | ⬜ skeleton |
| 12 | Regression Models | ⬜ skeleton |

Chapters 1–4 are assumed. Remaining chapters are scaffolded with their full section structure and filled in one at a time.

## Simulations

Every *Simulation Check* box in the notes reports real output from the seeded, pure-standard-library Python scripts in [`simulations/`](simulations/). Reproduce with:

```sh
python3 simulations/sim_ch5.py
python3 simulations/sim_ch5_part2.py
python3 simulations/sim_ch5_extra.py
python3 simulations/sim_ch6.py
python3 simulations/sim_ch7.py
python3 simulations/sim_ch8.py
python3 simulations/sim_ch9.py
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

## Caveats

These notes were drafted by an AI assistant working interactively. Known limitations, also
surfaced on the site's landing page and as inline **⚠ Flag** boxes on individual chapter pages:

- **Cross-check against the book.** Every proof was written out rather than copied, so every
  proof is also a place an error could have entered. C&B (2nd ed.) is the authority.
- **The `.tex` files have never been compiled** — no TeX engine was available where they were
  written. Expect small fixes on first compile.
- **The rendered pages were never viewed by their author** (no browser, no outside network).
  Layout, math rendering and overflow were checked by script, not by eye.
- **Simulation numbers come from single seeded runs.** Nothing is invented, but Monte Carlo
  error affects the last digit or two; theoretical targets are printed alongside. Re-running
  the scripts reproduces the quoted figures exactly.
- **Pure standard library only** — no NumPy/SciPy. Distributions, quantiles and the incomplete
  beta function are implemented in `simulations/statlib.py`, which self-checks against published
  table values but is not a tested numerical library.
- **Simulations illustrate, they do not prove.** A matching simulation is evidence a derivation
  was transcribed correctly, not that a theorem is true.
