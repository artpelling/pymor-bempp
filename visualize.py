import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla

from mpl_toolkits.axes_grid1 import make_axes_locatable

from bempp.api import GridFunction
from bempp.api.operators.potential.helmholtz import single_layer

from pymor.algorithms.pod import pod

from sphere import A, fom, rom, snaps, num_snaps, reductor, space

    
# PARAMETERS
#--------------------
k = 15


# COMPUTE SOLUTIONS
#--------------------
mu = A.parameters.parse(k)
b = fom.solve(mu)
fom_sol = GridFunction(space, coefficients=b.to_numpy().squeeze())
b = reductor.reconstruct(rom.solve(mu))
rom_sol = GridFunction(space, coefficients=b.to_numpy().squeeze())


# PLOTTING
#--------------------
if __name__ == '__main__':
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
    slp_pot = single_layer(space, points[:, idx], k)

    # compute solutions on potential
    error = np.zeros(points.shape[1])
    error[:] = np.nan

    fom_solution = error.copy()
    fom_solution[idx] = np.real(np.exp(1j*k*points[0, idx]) - slp_pot.evaluate(fom_sol)).flat
    rom_solution = error.copy()
    rom_solution[idx] = np.real(np.exp(1j*k*points[0, idx]) - slp_pot.evaluate(rom_sol)).flat
    abs_error = error.copy()
    abs_error[idx] = np.abs(fom_solution[idx] - rom_solution[idx])
    rel_error = error.copy()
    rel_error[idx] = abs_error[idx] / np.abs(fom_solution[idx])

    abs_norm = spla.norm(abs_error[idx])
    rel_norm = abs_norm / spla.norm(fom_solution[idx])
    # make plot
    fig, axes = plt.subplots(figsize=(8, 8), dpi=300, ncols=2, nrows=2)
    for vec, ax, label in zip(
            (fom_solution, rom_solution, 20*np.log10(abs_error), 20*np.log10(rel_error)),
            axes.ravel(),
            ('FOM', 'ROM', f'Absolute error: {20*np.log10(abs_norm):2f} dB', f'Relative error: {20*np.log10(rel_norm):2f} dB')
    ):
        im = ax.imshow(vec.reshape((Nx, Ny)).T, extent=[-3, 3, -3, 3])
        ax.set_title(label)
        ax.set_xticks(())
        ax.set_yticks(())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    fig.suptitle(f"Unit sphere scattering, k={k}")
    plt.savefig(f'sphere_{k}.png')
    plt.show()
