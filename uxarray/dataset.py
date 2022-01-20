# Grid class and helper functions
# This software is provided under a slightly modified version
# of the Apache Software License. See the accompanying LICENSE file
# for more information.
#
# Description:
#
#
from logging import raiseExceptions
import xarray as xr
import numpy as np
from pathlib import PurePath
import os
from .grid import *


def open_dataset(filename, type):
    print("opening dataset: ", filename)
    if type == "exo" or "ugrid" or "scrip" or "shp":
        ux_grid = Grid(str(filename))
    else:
        print("Meshfile type is not supported")
    return ux_grid.grid_ds
