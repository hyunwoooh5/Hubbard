#!/usr/bin/env python

from models import hubbard
from mc import metropolis, replica, hmc
from contour import *
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


parser = argparse.ArgumentParser(description="Generate configurations")
parser.add_argument('model', type=str, help="model filename")
parser.add_argument('contour', type=str, help="contour filename")
parser.add_argument('config', type=str, help="config filename")
parser.add_argument('-r', '--replica', action='store_true',
                    help="use replica exchange")
parser.add_argument('-H', '--hmc', action='store_true',
                    help="use HMC")
parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
                    help="number of replicas (with -r)")
parser.add_argument('-maxh', '--max-hbar', type=float,
                    default=10., help="maximum hbar (with -r)")
parser.add_argument('-N', '--samples', default=-1, type=int,
                    help='number of samples before termination')
parser.add_argument('-S', '--skip', default=30, type=int,
                    help='number of steps to skip')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--seed-time', action='store_true',
                    help="seed PRNG with current time")
parser.add_argument('--dp', action='store_true',
                    help="turn on double precision")
parser.add_argument('-T', '--thermalize', default=0,
                    type=int, help="number of MC steps (* d.o.f) to thermalize")
args = parser.parse_args()

seed = args.seed
if args.seed_time:
    seed = time.time_ns()
key = jax.random.PRNGKey(seed)

if args.dp:
    jax.config.update('jax_enable_x64', True)

with open(args.model, 'rb') as f:
    model = eval(f.read())

with open(args.contour, 'rb') as f:
    contour, contour_params = pickle.load(f)

contour_ikey, chain_key = jax.random.split(key, 2)

V = model.dof
skip = args.skip

if args.skip == 30:
    skip = V
if args.hmc:
    skip = 1

if type(contour) == RealContour or type(contour) == ConstantShift:
    @jax.jit
    def Seff(x, p):
        xt = contour.apply(p, x)
        Seff = model.action(xt)
        return Seff

else:
    @jax.jit
    def Seff(x, p):
        j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
        s, logdet = jnp.linalg.slogdet(j)
        xt = contour.apply(p, x)
        Seff = model.action(xt) - jnp.log(s) - logdet
        return Seff


@jax.jit
def observe(x, p):
    phase = jnp.exp(-1j*Seff(x, p).imag)
    phi = contour.apply(p, x)
    return phase, model.observe(phi)


if args.replica:
    chain = replica.ReplicaExchange(lambda x: Seff(x, contour_params), jnp.zeros(
        V), chain_key, delta=1./jnp.sqrt(V), max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
elif args.hmc:
    chain = hmc.Chain(lambda x: Seff(x, contour_params), jnp.zeros(V), chain_key, L=20, dt=0.2)
else:
    chain = metropolis.Chain(lambda x: Seff(x, contour_params), jnp.zeros(V), chain_key, delta=1./jnp.sqrt(V))

chain.calibrate()
chain.step(N=args.thermalize*V)
chain.calibrate()

configs = []
def save():
    with open(args.config, 'wb') as f:
        pickle.dump(jnp.array(configs), f)

try:
    def slc(it): return it
    if args.samples >= 0:
        def slc(it): return itertools.islice(it, args.samples)

    for x in slc(chain.iter(skip)):
        phase, obs = observe(x)
        if jnp.size(obs) == 1:
            obsstr = str(obs)
        else:
            obsstr = " ".join([str(x) for x in obs])
        print(f'{phase} {obsstr} {chain.acceptance_rate()}', flush=True)
        configs.append(x)
        if len(configs) % 1000 == 0:
            save()

except KeyboardInterrupt:
    pass
