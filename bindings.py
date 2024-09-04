#!/usr/bin/env python3

import numpy as np

from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from bempp.api import complex_callable, GridFunction
from bempp.api.space.space import FunctionSpace as BemppFunctionSpace
from bempp.api.operators.boundary.helmholtz import adjoint_double_layer, single_layer
from bempp.api.operators.boundary.sparse import identity


class BemppBoundaryOperator(NumpyMatrixBasedOperator):
    def __init__(self, space, source_id=None, range_id=None, solver_options=None, name=None):
        assert isinstance(space, BemppFunctionSpace)
        self.__auto_init(locals())
        dim = space.global_dof_count
        self.source = NumpyVectorSpace(dim, source_id)
        self.range = NumpyVectorSpace(dim, range_id)
        self.parameters_own = {'k': 1}

    def _assemble(self, mu=None):
        k = mu['k'][0]
        id = identity(self.space, self.space, self.space)
        sl = single_layer(self.space, self.space, self.space, k)
        dl = adjoint_double_layer(self.space, self.space, self.space, k)
        op = 0.5*id - 1j*k*sl + dl
        return op.weak_form().A


class BemppRhsOperator(NumpyMatrixBasedOperator):
    def __init__(self, space, source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        dim = space.global_dof_count
        self.source = NumpyVectorSpace(1, source_id)
        self.range = NumpyVectorSpace(dim, range_id)
        self.parameters_own = {'k': 1}

    def _assemble(self, mu=None):
        k = mu['k'][0]
        @complex_callable
        def callable(x, n, domain_index, result):
            result[0] = 1j*k*np.exp(1j*k*x[0])*(n[0] - 1)
        return GridFunction(self.space, fun=callable).projections().reshape(-1, 1)


from pymor.operators.interface import Operator
from pymor.core.pickle import unpicklable
from pymor.vectorarrays.list import CopyOnWriteVector, ComplexifiedListVectorSpace

@unpicklable
class BemppVector(CopyOnWriteVector):
    def __init__(self, impl):
        self.impl = impl

    def to_numpy(self, ensure_copy=False):
        return self.impl.projections()

    def __scal__(self, alpha):
        return BemppVector(alpha * self.impl)

    def norm(self):
        return self.impl.l2_norm()

    def norm2(self):
        return self.impl.l2_norm()**2

    def __add__(self, other):
        return BemppVector(self.impl + other.impl)

    def __sub__(self, other):
        return BemppVector(self.impl - other.impl)

    def __neg__(self):
        return BemppVector(self.impl.__neg__())

class BemppVectorSpace(ComplexifiedListVectorSpace):
    pass

class BemppOperator(Operator):
    """Wraps a BEMpp operator as an |Operator|"""

    def __init__(self, impl):
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(self * U.impl)
