#!/usr/bin/env python3


# wrap BEMPP operators


# set up the pyMOR model
from pymor.models.basic import StationaryModel
model = StationaryModel(A, f)
