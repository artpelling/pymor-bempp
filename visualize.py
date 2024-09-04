import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bempp.api import GridFunction
from bempp.api.operators.potential import helmholtz as helmholtz_potential

from pymor.algorithms.pod import pod

from sphere import A, fom, rom, snaps, num_snaps, reductor, piecewise_const_space

    
# PARAMETERS
#--------------------
k = 15


# COMPUTE SOLUTIONS
#--------------------
mu = A.parameters.parse(k)
b = fom.solve(mu)
fom_sol = GridFunction(piecewise_const_space, coefficients=b.to_numpy().squeeze())
b = reductor.reconstruct(rom.solve(mu))
rom_sol = GridFunction(piecewise_const_space, coefficients=b.to_numpy().squeeze())


# PLOTTING
#--------------------
# create grid
Nx = 200
Ny = 200
xmin, xmax, ymin, ymax = [-3, 3, -3, 3]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))
x, y, z = points
idx = np.sqrt(x**2 + y**2) > 1.0

# create potential for evaluation
slp_pot = helmholtz_potential.single_layer(
    piecewise_const_space, points[:, idx], k)

# compute solutions on potential
error = np.zeros(points.shape[1])
error[:] = np.nan

fom_solution = error.copy()
fom_solution[idx] = np.real(np.exp(1j*k*points[0, idx]) - slp_pot.evaluate(fom_sol)).flat
rom_solution = error.copy()
rom_solution[idx] = np.real(np.exp(1j*k*points[0, idx]) - slp_pot.evaluate(rom_sol)).flat

error[idx] = fom_solution[idx] - rom_solution[idx]

# make plot
fig, axes = plt.subplots(figsize=(8, 8), dpi=300, ncols=2, nrows=2)
vmin = np.nan_to_num([fom_solution, rom_solution], nan=0).min()
vmax = np.nan_to_num([fom_solution, rom_solution], nan=0).max()
for vec, ax, label in zip((fom_solution, rom_solution, error), axes.ravel()[:-1], ('FOM', 'ROM', 'Error')):
    im = ax.imshow(vec.reshape((Nx, Ny)).T, extent=[-3, 3, -3, 3])
    ax.set_title(label)
    ax.set_xticks(())
    ax.set_yticks(())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


# compute full spectrum
_, sv = pod(snaps, modes=num_snaps, rtol=0, method='qr_svd')
sv /= np.max(sv)
ax = axes.ravel()[3]
ax.set_aspect('auto')
ax.semilogy(np.arange(num_snaps+1, sv)
ax.vlines(rom.order, np.min(sv), np.max(sv), color='r', linestyle=':')
ax.set_title('POD Spectrum')
ax.set_xlim((1, num_snaps))
ax.yaxis.tick_right()
fig.suptitle("Scattering from the unit sphere, solution in plane z=0")
plt.show()
