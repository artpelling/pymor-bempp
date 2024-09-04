
import bempp.api
from bempp.api.operators.boundary import helmholtz, sparse
from bempp.api.operators.potential import helmholtz as helmholtz_potential
from bempp.api.linalg import gmres
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import identity


# Room geometry is imported from mesh
bempp.core.opencl_kernels.set_default_cpu_device(1, 0)


def ang_freq (f):
    return 2 * np.pi * f


def wavenumber (f, c=343):
    return ang_freq(f) / c


def inc_pr_field (src, rec, f, Q, rho_0=1.21):
    omega_inc = ang_freq(f)
    k_inc = wavenumber(f)
    r_vec = rec - src
    r_inc = np.linalg.norm(r_vec)
    p_inc = 1j * omega_inc * rho_0 * Q * np.exp(1j * k_inc * r_inc) / (4 * np.pi * r_inc)
    return p_inc


# Air Parameters
rho_0 = 1.21
c = 343

# Room Parameters
abs_coeff_room = 0.02
beta = abs_coeff_room / 8

# Source Parameters
s_x = 2.92
s_y = 3.18
s_z = 1.20
Q = 1e-3
src = [s_x, s_y, s_z]

# Receiver Position 
r_x = 3.61
r_y = 1.98
r_z = 1.20
rec = np.array([r_x, r_y, r_z])

# Import Mesh
grid = bempp.api.import_grid('cuboid.msh')
function_space =  bempp.api.function_space(grid, "P", 1)


# def BEMPP_f(f):
#     k = wavenumber(f)
#     omega = ang_freq(f)

#     @bempp.api.complex_callable
#     def u_inc_callable(x, n, domain_index, result):
#         r = np.sqrt((s_x - x[0])**2 + (s_y - x[1])**2 + (s_z - x[2])**2)
#         result[0] = 1j * rho_0 * omega * Q * np.exp(1j * k * r) / (4 * np.pi * r)

#     return bempp.api.GridFunction(space, fun=u_inc_callable).evaluate_on_vertices()



# wrap BEMPP operators
from pymor.operators.numpy import NumpyMatrixBasedOperator, NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyBEMPPBoundaryOperator(NumpyMatrixBasedOperator):
    def __init__(self, space, source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        dim = space.global_dof_count
        self.source = NumpyVectorSpace(dim, source_id)
        self.range = NumpyVectorSpace(dim, range_id)
        self.parameters_own = {'w': 1}

    def _assemble(self, mu=None):
        f = mu['w']
        k = wavenumber(f)
        identity = sparse.identity(self.space, self.space, self.space)
        single_layer = helmholtz.single_layer(self.space, self.space, self.space, k, device_interface='opencl')
        double_layer = helmholtz.double_layer(self.space, self.space, self.space, k, device_interface='opencl')
        return double_layer.weak_form().A - 0.5 * identity.weak_form().A + 1j * k * beta * single_layer.weak_form().A


class NumpyBEMPPrhsOperator(NumpyMatrixBasedOperator):
    def __init__(self, space, source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        dim = space.global_dof_count
        self.source = NumpyVectorSpace(1, source_id)
        self.range = NumpyVectorSpace(dim, range_id)
        self.parameters_own = {'w': 1}

    def _assemble(self, mu=None):
        f = np.squeeze(mu['w'])
        k = wavenumber(f)
        omega = ang_freq(f)
        @bempp.api.complex_callable
        def u_inc_callable(x, n, domain_index, result):
            r = np.sqrt((s_x - x[0])**2 + (s_y - x[1])**2 + (s_z - x[2])**2)
            result[0] = 1j * rho_0 * omega * Q * np.exp(1j * k * r) / (4 * np.pi * r)
        u_inc = -bempp.api.GridFunction(self.space, fun=u_inc_callable).projections().reshape(-1, 1)
        return u_inc


A = NumpyBEMPPBoundaryOperator(function_space)
f = NumpyBEMPPrhsOperator(function_space)

from pymor.models.basic import StationaryModel
fom = StationaryModel(A, f, name="BEM")
