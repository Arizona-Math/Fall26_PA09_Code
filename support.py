"""Supplied examples, Newton reference, and strong-Wolfe line search."""
import warnings
import numpy as np
from scipy.optimize import line_search

def circle(x):
    u,v=x
    return np.array([u*u+v*v-1, u*u-v*v+0.5])

def circle_jacobian(x):
    u,v=x
    return np.array([[2*u,2*v],[2*u,-2*v]])

def newton(F,J,x0,tol=1e-10,max_iter=60):
    x=np.array(x0,dtype=float,copy=True); fx=F(x); nfev=1; njev=0
    history=[dict(iteration=0,residual=float(np.linalg.norm(fx,np.inf)),step=0.)]
    status='max_iter'
    for k in range(max_iter+1):
        if np.linalg.norm(fx,np.inf)<=tol: status='converged'; break
        if k==max_iter: break
        njev+=1
        try: s=np.linalg.solve(J(x),-fx)
        except np.linalg.LinAlgError: status='singular'; break
        x=x+s; fx=F(x); nfev+=1
        history.append(dict(iteration=k+1,residual=float(np.linalg.norm(fx,np.inf)),
                            step=float(np.linalg.norm(s,np.inf))))
        if not np.isfinite(fx).all(): status='nonfinite'; break
    return dict(x=x,residual=fx,converged=status=='converged',status=status,
                iterations=len(history)-1,nfev=nfev,njev=njev,history=history)

def quadratic(x):
    return float(.5*(x[0]**2+10*x[1]**2))

def quadratic_gradient(x):
    return np.array([x[0],10*x[1]])

def rosenbrock(x):
    return float(100*(x[1]-x[0]**2)**2+(1-x[0])**2)

def rosenbrock_gradient(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])

def strong_wolfe(f,g,x,p,fx,gx):
    """Return a checked strong-Wolfe step or None; never substitute Armijo only."""
    slope=float(gx@p)
    if not np.isfinite(slope) or slope>=0: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        alpha,*_=line_search(f,g,x,p,gfk=gx,old_fval=fx,c1=1e-4,c2=.9,maxiter=30)
    if alpha is None or not np.isfinite(alpha) or alpha<=0: return None
    trial=x+alpha*p; value=f(trial); grad=g(trial)
    if not np.isfinite(value) or not np.isfinite(grad).all(): return None
    if value>fx+1e-4*alpha*slope or abs(grad@p)>.9*abs(slope): return None
    return float(alpha)

def steepest_descent(f,g,x0,tol=1e-8,max_iter=200):
    """Supplied comparison; each accepted step uses the same Wolfe wrapper."""
    x=np.array(x0,dtype=float,copy=True); nfev=ngev=0
    def fun(z):
        nonlocal nfev
        nfev+=1; return f(z)
    def grad(z):
        nonlocal ngev
        ngev+=1; return g(z)
    fx=fun(x); gx=grad(x)
    history=[dict(iteration=0,objective=float(fx),gradient_norm=float(np.linalg.norm(gx,np.inf)),step=0.)]
    status='max_iter'
    for k in range(max_iter+1):
        if np.linalg.norm(gx,np.inf)<=tol: status='converged'; break
        if k==max_iter: break
        p=-gx; alpha=strong_wolfe(fun,grad,x,p,fx,gx)
        if alpha is None: status='line_search_failed'; break
        s=alpha*p; x=x+s; fx=fun(x); gx=grad(x)
        history.append(dict(iteration=k+1,objective=float(fx),gradient_norm=float(np.linalg.norm(gx,np.inf)),step=float(np.linalg.norm(s,np.inf))))
    return dict(x=x,objective=float(fx),gradient=gx,converged=status=='converged',status=status,
                iterations=len(history)-1,nfev=nfev,ngev=ngev,history=history)
