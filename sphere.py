#!/usr/bin/env python3

import numpy as np
import bempp.api

from pymor.algorithms.pod import pod
from pymor.models.basic import StationaryModel
from pymor.reductors.basic import StationaryRBReductor

from bindings import BemppBoundaryOperator, BemppRhsOperator


# PARAMETERS
#--------------------
k_range = 2*np.pi*np.array((20, 20000))/343
num_snaps = 30
pod_rtol = 1e-7
num_validate = 50
#--------------------


# SET UP MODEL
#--------------------
grid = bempp.api.shapes.regular_sphere(3)
piecewise_const_space = bempp.api.function_space(grid, "DP", 0)
A = BemppBoundaryOperator(piecewise_const_space)
f = BemppRhsOperator(piecewise_const_space)
fom = StationaryModel(A, f, name='BEM_full')


# MODEL ORDER REDUCTION
#--------------------
# collect snapshots
k_snaps = np.geomspace(*k_range, num_snaps)
snaps = fom.solution_space.empty()
for mu in [fom.parameters.parse(k) for k in k_snaps]:
    snaps.append(fom.solve(mu))

# create reduced pod basis
V, sv = pod(snaps, modes=num_snaps, rtol=pod_rtol)

# reduce model
reductor = StationaryRBReductor(fom, RB=V)
rom = reductor.reduce()
rom.enable_logging()
