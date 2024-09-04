#!/usr/bin/env python3

import numpy as np

from bempp.api.shapes import regular_sphere
from bempp.api import function_space

from pymor.algorithms.pod import pod
from pymor.models.basic import StationaryModel
from pymor.reductors.basic import StationaryRBReductor

from bindings import BemppBoundaryOperator, BemppRhsOperator


# PARAMETERS
#--------------------
k_range = 2*np.pi*np.array((10, 1000))/343
num_snaps = 30
pod_rtol = 1e-7
num_validate = 50
#--------------------


# SET UP MODEL
#--------------------
grid = regular_sphere(3)
space = function_space(grid, "DP", 0)
A = BemppBoundaryOperator(space)
f = BemppRhsOperator(space)
fom = StationaryModel(A, f, name='BEM_full')


# MODEL ORDER REDUCTION
#--------------------
# collect snapshots
k_snaps = np.linspace(*k_range, num_snaps)
snaps = fom.solution_space.empty()
for mu in [fom.parameters.parse(k) for k in k_snaps]:
    snaps.append(fom.solve(mu))

# create reduced pod basis
V, sv = pod(snaps, modes=num_snaps, rtol=pod_rtol)

# reduce model
reductor = StationaryRBReductor(fom, RB=V)
rom = reductor.reduce()
rom.enable_logging()


# PLOTTING
#--------------------
if __name__ == '__main__':
    # compute fom and rom solutions
    k_validate = np.geomspace(*k_range, num_validate)
    U = fom.solution_space.empty()
    u_r = rom.solution_space.empty()
    for mu in [fom.parameters.parse(k) for k in k_validate]:
        U.append(fom.solve(mu))
        u_r.append(rom.solve(mu))

    # compute relative error
    U_r = reductor.reconstruct(u_r)
    rel_error = (U - U_r).norm2() / U.norm2()
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(dpi=300, ncols=2)
    _, sv = pod(snaps, modes=num_snaps, rtol=0, method='qr_svd')
    sv /= np.max(sv)
    ax = axes[0]
    ax.semilogy(np.arange(num_snaps)+1, sv)
    ax.vlines(rom.order, np.min(sv), np.max(sv), color='r', linestyle=':')
    ax.set_title('POD Spectrum')
    ax.set_xlim((1, num_snaps))

    ax = axes[1]
    ax.semilogx(k_validate, 20*np.log10(rel_error))
    ax.set_title('Relative Error')
    ax.set_xlim(k_range)
    plt.savefig('plots/rom_quality.png')
    plt.show()
