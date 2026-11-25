"""Supplied plots and JSON, after implementation of experiment()."""
import argparse
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [plain(v) for v in x]
    return x

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    args=parser.parse_args(); dest=Path(args.output); dest.mkdir(parents=True,exist_ok=True)
    data=importlib.import_module('bfgs').experiment()
    (dest/'experiments.json').write_text(json.dumps(plain(data),indent=2)+'\n')
    groups={'circle':data} if 'broyden' in data else data
    for case,results in groups.items():
        fig,ax=plt.subplots()
        for method,result in results.items():
            rows=result['history']; key='residual' if 'residual' in rows[0] else 'gradient_norm'
            ax.semilogy([r['iteration'] for r in rows],[max(r[key],1e-16) for r in rows],label=method)
            print(case,method,result['status'],'iterations',result['iterations'],'nfev',result['nfev'])
        ax.set(xlabel='Accepted iterations',ylabel=key.replace('_',' '),title=case)
        ax.legend(); ax.grid(); fig.tight_layout(); fig.savefig(dest/(case+'.png')); plt.close(fig)

if __name__=='__main__': main()
