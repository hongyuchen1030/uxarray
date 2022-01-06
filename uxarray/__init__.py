from selectors import EpollSelector
from .grid import *


def open_dataset(filename, type):
    print("opening dataset: ", filename)
    if type == "exo2" or "exo" or "ugrid" or "scrip" or "ux" or "shp":
        ux_grid = Grid(str(filename))
    else:
        print("Meshfile type is not supported")
    return ux_grid.grid_ds
