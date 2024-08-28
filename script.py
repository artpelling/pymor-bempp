#!/usr/bin/env python3

from .model import fom

# collect snapshots
wk = np.logspace(20, 100, 10)
mus = [bempp_model.parameters.parse(w) for w in wk]
snapshots = fom.solve(mus)

# create reduced pod basis
from pymor.algorithms.pod import pod
V, sv = pod(snapshots)

# reduce model
from pymor.reductors.basic import StationaryRBReductor
srbred = StationaryRBReductor(fom, RB=V)
