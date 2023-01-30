#!/usr/bin/env python

import argparse
import ast
import pickle
import sys

from models import hbar, hubbard

parser = argparse.ArgumentParser(
        description="""Define a model.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
        )
parser.add_argument('--hbar', type=complex, default=1., help='hbar')
parser.add_argument('model', type=str, help='model filename')
subparsers = parser.add_subparsers(title="Models", help="Type of theory to define", required=True)

def mkhubbard(args):
    lattice = hubbard.Lattice(args.L, args.nt)
    model = hubbard.ImprovedModel(lattice, args.beta, args.Kappa, args.U, args.Mu, args.dt)
    return model


parser_hubbard = subparsers.add_parser('hubbard', help='Hubbard model')
parser_hubbard.add_argument('L', type=int, help='lattice size')
parser_hubbard.add_argument('nt', type=int, help='number of time slices')
parser_hubbard.add_argument('beta', type=float, help='H-S parameter')
parser_hubbard.add_argument('Kappa', type=float, help='hopping')
parser_hubbard.add_argument('U', type=float, help='potential')
parser_hubbard.add_argument('Mu', type=float, help='chemical potential')
parser_hubbard.add_argument('dt', type=float, help='1/T=nt*dt')
parser_hubbard.add_argument('--conventional', action='store_true',
                            help='Use conventional action (default is improved)')
parser_hubbard.set_defaults(func=mkhubbard)


args = parser.parse_args()
model = args.func(args)

if args.hbar != 1.:
    model = hbar.WrapHbar(model, args.hbar)

with open(args.model, 'wb') as f:
    pickle.dump(model, f)
