# PA09 Code: BFGS

Release: November 20, 2026. Due: **November 30, 2026, 11:59 p.m. Arizona time**.
Code: 50 points; Written: 50 points. Budget 4–5 hours for the whole package.
Submit `bfgs.py` and your attribution record to **PA09 Code**.

Work individually. Acknowledge reused code and AI assistance in `AUTHORS.md`,
and record independent numerical checks. NumPy array operations and linear
solves are allowed. Implement the assigned updates and solver loops yourself;
library root finders or optimizers may be used only as reference comparisons.
Do not change the supplied support module or function signatures.

Create an environment with `python3 -m venv .venv`, activate it, and install
`python -m pip install -r requirements.txt`. Run `make check` for public smoke
checks, `make experiments` for the supplied figures/JSON, and `make submission`
to create the upload. The untouched starter intentionally raises
`NotImplementedError`; that failure does not indicate an installation problem.

Upload **submission.zip alone** to the Code assignment, or upload the required
Python source directly. Do not upload both: duplicate source files are rejected.
The grader supplies its own `support.py`. No PDF or `solution.tex` is required
by the programming autograder. Submit the separate fixed-layout PDF to Written.

## Implement three functions

### `inverse_update(H, s, y, curvature_tol=1e-12)` — 20 points

Set `ys=y@s` and `scale=norm(y,2)*norm(s,2)`. If `scale==0` or
`ys <= curvature_tol*scale`, return an unchanged **copy** of H. Otherwise use
`rho=1/ys`, `V=I-rho*outer(s,y)`, and return

    V @ H @ V.T + rho * outer(s,s).

Do not mutate inputs. For SPD H and accepted positive curvature the result is
SPD and satisfies H_new@y=s. Tests allocate 15 points to the matrix identities
and 5 to the skip policy. Skip tests do not claim that Wolfe steps in exact
arithmetic have negative curvature; they verify the stated numerical safeguard.

### `solve(f,g,x0,H0=None,tol=1e-8,max_iter=200,`
### `      line_search=strong_wolfe,curvature_tol=1e-12)` — 20 points

Use I when H0 is None; otherwise assume H0 is SPD. Copy inputs. Compute f and g
at the initial point; test `norm(g,inf) <= tol` before taking a step, even with
a zero iteration budget. For a nonstationary point compute `p=-H@g` and call
the **passed callable** `line_search(f_counted,g_counted,x,p,fx,gx)`. This supplied
wrapper returns a positive strong-Wolfe step length or `None`. You do not need
to implement bracketing/zoom. Count every f and g evaluation, including calls
made inside this wrapper; pass counting closures rather than the raw functions.

If the line search returns None, retain the current point and return
`line_search_failed`. After an accepted step compute the new value and gradient,
set `s=x_new-x`, `y=g_new-g`, apply `inverse_update`, and check convergence at the
new point. A skipped matrix update still counts as an accepted iterate.

Return `x`, `objective`, `gradient`, `status`, `converged`, `iterations`, `nfev`,
`ngev`, and `history`. Statuses are `converged`, `max_iter`, `line_search_failed`,
`not_descent` (if g@p>=0), and `nonfinite`. `iterations` counts accepted steps.
Histories include iteration 0. Each row has `iteration`, `objective`,
`gradient_norm` (infinity norm), `step` (infinity norm), `curvature` (s@y), and
`skipped` (Boolean). Initially step=0, curvature=None, skipped=False.
The converged Boolean is true exactly for status `converged`.

Assume compatible real arrays, nonnegative budget, and positive tolerance.
Tests use SPD quadratics, Rosenbrock, initial stationary points, a zero budget,
and a supplied line-search failure stub. They also compare your counts with
the actual observed f/g calls: excluding line-search work loses credit.

### `experiment()` — 10 points

Return a dictionary keyed `quadratic` and `rosenbrock`; each contains results
keyed `bfgs` and `steepest`. Use the supplied functions and gradients, starts
`(2,-1)` and `(-1.2,1)`, and tol=1e-8, max_iter=200 for both methods.
`steepest_descent` is supplied and uses the same line search.
Retain the full results, even if steepest descent exhausts its budget.
Run the supplied plotting driver. Compare objective/gradient histories and
evaluation counts. A result that fails to reach tolerance must not be relabeled
as successful. Only your three functions are submitted for automatic grading.
