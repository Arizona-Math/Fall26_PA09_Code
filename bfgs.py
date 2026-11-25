"""PA09: implement these three functions. See problem.md for the exact contract."""
import numpy as np
from support import (strong_wolfe, steepest_descent, quadratic, quadratic_gradient,
                     rosenbrock, rosenbrock_gradient)

def inverse_update(H,s,y,curvature_tol=1e-12):
    """Return inverse BFGS, or a copy of H if the relative curvature test fails.

    Do not mutate inputs. Use the formula and skip criterion in problem.md.
    """
    raise NotImplementedError('Implement the inverse-BFGS update')

def solve(f,g,x0,H0=None,tol=1e-8,max_iter=200,line_search=strong_wolfe,curvature_tol=1e-12):
    """Integrate inverse BFGS with the supplied line search.

    Return x, objective, gradient, status, converged, iterations, nfev, ngev,
    and history. Count all f/g calls, including those inside line_search.
    H0 defaults to I. Check convergence initially and after accepted steps.
    Use the callable passed in line_search; tests also supply a failure stub.
    """
    raise NotImplementedError('Implement BFGS with the supplied line search')

def experiment():
    """Return quadratic and rosenbrock dictionaries, each with bfgs and steepest.

    Starts: (2,-1) for quadratic, (-1.2,1) for Rosenbrock.
    Use tol=1e-8 and max_iter=200 for both methods. Steepest descent is supplied.
    Keep failure results: exhausting 200 steps is valid evidence, not a zero.
    """
    raise NotImplementedError('Compare BFGS with steepest descent')
