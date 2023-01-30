#!/usr/bin/env python

import numpy as np
import sys
from cmath import phase

def bootstrap(xs, ws=None, N=100, Bs=50):
    if Bs > len(xs):
        Bs = len(xs)
    B = len(xs)//Bs
    if ws is None:
        ws = xs*0 + 1
    # Block
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    # Regular bootstrap
    y = x * w
    m = (sum(y) / sum(w))
    ms = []
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real) + 1j*np.std(ms.imag)

lines = [l for l in sys.stdin.readlines() if l[0] != '#']
dat = np.array([[complex(x) for x in l.split()] for l in lines])

boltz = dat[:,0]
dat = dat[:,1:]

# Reweighting (equivalent to sign problem)
rew, rew_err = bootstrap(boltz)
print(f'Reweighting: {rew} {rew_err} {abs(rew)} {phase(rew)}')

# Observables
for i in range(dat.shape[1]):
    print(*bootstrap(dat[:,i],boltz))
