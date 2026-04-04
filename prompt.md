Previously you got interrupted, now you have to finish the work you started before. Continue from where you left off, writing the two '.md' files as requested ("Write EXPERIMENTS_AND_RESULTS.md" and "Write CODEBASE_DESCRIPTION.md").

For full context, i'll attach another time the prompt describing the task you have to do:

I need to prepare the final documentation files that will serve as the sole source of truth for writing our academic report. Please create/update two files:

---

**File 1 — Update `CODEBASE_DESCRIPTION.md`**

Rewrite this file to reflect the **current and final state** of the entire codebase. It must be self-contained and cover:

- **Project overview**: what the project does (ELM + Lasso, two optimizers), the course context (Optimization for Data Science, Università di Pisa), and the team.
- **Directory structure**: annotated tree of every file and folder, with a one-line description of each file's purpose.
- **Module-by-module breakdown**: for every `.py` file, document every class and function — its signature, what it does, its inputs/outputs, and any key implementation decisions (e.g. why soft-thresholding is applied after the momentum step, why `H` is precomputed once, how `L_g = 2·λ_max(HᵀH)` is computed without forming `HᵀH` explicitly).
- **Algorithm implementations**: concisely describe the exact update rules coded for (a) Proximal Heavy Ball — both CM and NAG variants, with the momentum schedule from Eq. 5 of Sutskever et al. — and (b) Nesterov Smoothing with Huber approximation and FISTA-style acceleration. Note any deviations from the theoretical ideal.
- **Data pipeline**: how synthetic data is generated (dimensions, distributions), how any real dataset is loaded and preprocessed, and how `H = σ(X·W₁ᵀ)` is computed and stored.
- **Logging and stopping criteria**: what is recorded per iteration (objective value, gradient norm, wall-clock time, distance to reference solution if applicable) and what stopping conditions are used.
- **Validation approach**: how the reference solution is obtained (e.g. `sklearn.Lasso` or `cvxpy`) and how numerical correctness is verified (gradient finite-difference checks, objective value comparison).
- **Dependencies and reproducibility**: Python version, key libraries, how to install and run the experiments end to end.

---

**File 2 — Create `EXPERIMENTS_AND_RESULTS.md`**

Write a thorough, standalone document explaining **every experiment we ran, every plot we generated, and every CSV we produced**. For each experiment, structure the content as follows:

1. **Motivation** — what scientific/algorithmic question this experiment answers, and why it matters for the report.
2. **Setup** — exact problem dimensions (n, d, h, k, p = h·k), activation function, λ value, algorithm parameters used, number of iterations, hardware.
3. **What was measured** — which metrics were logged and why (e.g. `f(w_k) − f*` on log scale to reveal convergence rate, wall-clock time to reveal practical cost, sparsity fraction to assess L1 effect).
4. **Results** — describe the outcome in detail: numbers, trends, comparisons. For each plot, explain what is on each axis, what each curve/line represents, and what the visual pattern means algorithmically (e.g. "the log-linear decay confirms the theoretically predicted linear convergence of proximal heavy ball on this strongly convex instance"). For each CSV, describe its columns and what the rows represent.
5. **Interpretation and takeaways** — what conclusion can be drawn, whether it matches theory, and any surprising or noteworthy observations.

The experiments to document include (but are not limited to):
- **Convergence behavior**: `f(w_k) − f*` vs iteration and vs wall-clock time (log-scale y-axis) for both algorithms on a fixed medium-scale problem.
- **CM vs NAG comparison**: convergence curves comparing Classical Momentum and Nesterov Accelerated Gradient variants of Heavy Ball at various `μ_max` values (matching Table 1 of Sutskever et al.).
- **Momentum schedule sensitivity**: effect of varying `μ_max ∈ {0.9, 0.99, 0.995, 0.999}` and of using a fixed vs scheduled momentum.
- **Step size sensitivity for Heavy Ball**: convergence behavior as `ε` is varied around `1/L_g`.
- **Smoothing parameter sensitivity**: convergence curves for `μ ∈ {10⁻¹, 10⁻², 10⁻³, 10⁻⁴, 10⁻⁵}`, showing the approximation-quality vs convergence-speed trade-off.
- **Regularization strength**: effect of varying `λ` across orders of magnitude on convergence speed and solution sparsity.
- **Scalability**: time to convergence vs problem size (n, h, k), ideally on a log-log scale.
- **Sparsity analysis**: fraction of near-zero entries in `W₂` as a function of `λ`, comparing the exact sparsity of the proximal method vs the approximate sparsity of the smoothing method.
- **Algorithm head-to-head**: side-by-side comparison of both algorithms on the same problem — iterations, time, final objective, sparsity, distance to reference solution.

Analyze every plot and CSV you have access to carefully before writing. The document must be detailed enough that someone writing the report from scratch — without looking at any code or plot directly — could accurately describe every result, quote relevant numbers, and draw the correct conclusions.

Both files will be the **only inputs** used when writing the final academic report, so they must be complete, precise, and self-contained.