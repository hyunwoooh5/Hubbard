#!/usr/bin/env python

import argparse
import itertools
import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')

from mc import metropolis,replica
from contour import PeriodicManifold,RealManifold,Manifold
from nf import Flow,action_effective

parser = argparse.ArgumentParser(description="Train contour")
parser.add_argument('model', type=str, help="model filename")
parser.add_argument('contour', type=str, help="contour filename")
parser.add_argument('-r', '--replica', action='store_true', help="use replica exchange")
parser.add_argument('--nreplicas', type=int, default=30, help="number of replicas (with -r)")
parser.add_argument('--max-hbar', type=float, default=10., help="maximum hbar (with -r)")
parser.add_argument('-N', '--samples', default=-1, type=int,
        help='number of samples before termination')
parser.add_argument('-S', '--skip', default=30, type=int,
        help='number of steps to skip')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--seed-time', action='store_true', help="seed PRNG with current time")
args = parser.parse_args()

seed = args.seed
if args.seed_time:
    seed = time.time_ns()
key = jax.random.PRNGKey(seed)

with open(args.model, 'rb') as f:
    model = pickle.load(f)

with open(args.contour, 'rb') as f:
    manifold, manifold_params = pickle.load(f)

manifold_ikey, chain_key = jax.random.split(key, 2)

V = model.lattice.dof
skip = args.skip

if args.skip == 30:
    skip = V

@jax.jit
def Seff(x, p):
    j = jax.jacfwd(lambda y: manifold.apply(p, y))(x)
    s, logdet = jnp.linalg.slogdet(j)
    xt = manifold.apply(p, x)
    Seff = model.action(xt) - jnp.log(s) - logdet
    return Seff

@jax.jit
def observe(x, p):
    phase = jnp.exp(-1j*Seff(x, p).imag)
    phi = manifold.apply(p, x)
    return phase, model.observe(phi)

if args.replica:
    chain = replica.ReplicaExchange(lambda x: Seff(x, manifold_params), jnp.zeros(V), chain_key, max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
else:
    chain = metropolis.Chain(lambda x: Seff(x, manifold_params), jnp.zeros(V), chain_key)
chain.calibrate()

try:
    slc = lambda it: it
    if args.samples >= 0:
        slc = lambda it: itertools.islice(it, args.samples)
    for x in slc(chain.iter(skip)):
        phase, obs = observe(x, manifold_params)
        obsstr = " ".join([str(x) for x in obs])
        print(f'{phase} {obsstr}', flush=True)
except KeyboardInterrupt:
    pass

