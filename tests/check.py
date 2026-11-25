"""Public smoke checks. Failing TODOs are expected in an untouched starter."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import bfgs as student

H=np.eye(2); s=np.array([1.,0.]); y=np.array([2.,1.])
assert np.allclose(student.inverse_update(H,s,y),[[.75,-.5],[-.5,1.]])
assert np.allclose(student.inverse_update(H,s,-s),H)
r=student.experiment()
assert r['quadratic']['bfgs']['converged']
assert r['rosenbrock']['bfgs']['converged']

print('Public numerical checks passed.')
