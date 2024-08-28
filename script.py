#!/usr/bin/env python3

import numpy as np

from model import fom


# PARAMETERS
#--------------------
freq_range = (20, 120)
num_snaps = 30
pod_rtol = 1e-7
num_validate = 50
#--------------------


# MODEL ORDER REDUCTION
#--------------------

# collect snapshots
w_snaps = np.geomspace(*freq_range, num_snaps)
snaps = fom.solution_space.empty()
for mu in [fom.parameters.parse(w) for w in w_snaps]:
    snaps.append(fom.solve(mu))

# create reduced pod basis
from pymor.algorithms.pod import pod
V, sv = pod(snaps, rtol=pod_rtol)

# reduce model
from pymor.reductors.basic import StationaryRBReductor
reductor = StationaryRBReductor(fom, RB=V)
rom = reductor.reduce()
rom.enable_logging()


# VALIDATION
#--------------------

# compute fom and rom solutions
w_validate = np.geomspace(*freq_range, num_validate)
U = fom.solution_space.empty()
u_r = rom.solution_space.empty()
for mu in [fom.parameters.parse(w) for w in w_validate]:
    U.append(fom.solve(mu))
    u_r.append(rom.solve(mu))

# compute relative error
U_r = reductor.reconstruct(u_r)
rel_error = (U - U_r).norm2() / U.norm2()

# plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(dpi=200)
ax.semilogx(w_validate, 20*np.log10(rel_error))
ax.set_xlim(freq_range)
plt.savefig('relative_error.png')
