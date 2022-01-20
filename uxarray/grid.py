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


# grid class
class Grid:
    # Import methods
    from ._exodus import read_exodus, write_exodus
    from ._ugrid import read_ugrid, write_ugrid
    from ._shpfile import read_shpfile
    from ._scrip import read_scrip

    # Load the grid file specified by file.
    # The routine will automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.
    def __init__(self, *args, **kwargs):

        # initialize possible variables
        self.filepath = None
        self.gridspec = None
        self.vertices = None
        self.islatlon = None
        self.concave = None
        self.meshFileType = None

        self.grid_ds = None

        # determine initialization type
        in_type = type(args[0])
        # print(in_type, "intype", os.getcwd(), args[0], os.path.isfile(args[0]))

        # initialize for vertices
        if in_type is np.ndarray:
            self.vertices = args[0]
            self._init_vert()

        # initialize for file
        elif in_type is str and os.path.isfile(args[0]):
            self.filepath = args[0]
            self._init_file()

        # initialize for gridspec (??)
        elif in_type is str:
            self.gridspec = args[0]
            self._init_gridspec()

        else:
            pass

    # vertices init
    def _init_vert(self):
        print("initializing with vertices")

    # gridspec init
    def _init_gridspec(self):
        print("initializing with gridspec")

    # file init
    def _init_file(self):
        # find the file type
        try:
            # extract the file name and extension
            path = PurePath(self.filepath)
            file_extension = path.suffix

            # open dataset with xarray
            self.grid_ds = xr.open_dataset(self.filepath, mask_and_scale=False)
        except (TypeError, AttributeError) as e:
            msg = str(e) + ': {}'.format(self.filepath)
            print(msg)
            raise RuntimeError(msg)
            exit
        except (RuntimeError, OSError) as e:
            # check if this is a ugrid file
            # we won't use xarray to load that file
            if file_extension == ".ugrid":
                self.meshFileType = "ugrid"
            elif file_extension == ".shp":
                self.meshFileType = "shp"
            elif file_extension == ".ug":
                self.meshFileType = ".ug"
            else:
                msg = str(e) + ': {}'.format(self.filepath)
                print(msg)
                raise RuntimeError(msg)
                exit

        # Detect mesh file type:
        # if ds has coordx or coord - call it exo format
        # if ds has grid_size - call it SCRIP format
        # if ds has ? read as shape file populate_scrip_dataformat
        # TODO: add detection of shpfile
        try:
            self.grid_ds.coordx
            self.meshFileType = "exo"
        except AttributeError as e:
            pass
        try:
            self.grid_ds.grid_center_lon
            self.meshFileType = "scrip"
        except AttributeError as e:
            pass
        try:
            self.grid_ds.coord
            self.meshFileType = "exo"
        except AttributeError as e:
            pass
        try:
            self.grid_ds.Mesh2
            self.meshFileType = "ugrid"
        except AttributeError as e:
            pass

        if self.meshFileType is None:
            print("mesh file not supported")

        print(self.filepath, " is of type: ", self.meshFileType)

        # Now set Mesh2 ds
        # exodus file is cartesian grid, must convert to lat/lon?
        # use - pyproj https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python?

        if self.meshFileType == "exo":
            self._init_mesh2()
            self.read_exodus(self.grid_ds)
        elif self.meshFileType == "scrip":
            self.read_scrip(self.grid_ds)
        elif self.meshFileType == "ugrid":
            self.read_ugrid(self.filepath)
        elif self.meshFileType == "shp":
            self.read_shpfile(self.filepath)

    # initialize mesh2 DataVariable for uxarray
    def _init_mesh2(self):
        # set default values and initialize Datavariable "Mesh2" for uxarray
        self.grid_ds["Mesh2"] = xr.DataArray(
            data=0,
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of 2D unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })

    # renames the grid file
    def saveas_file(self, filename):
        path = PurePath(self.filepath)
        old_filename = path.name
        new_filepath = path.parent / filename
        self.filepath = str(new_filepath)
        self.write_ugrid(self.filepath)
        print(self.filepath)

    # Calculate the area of all faces.
    def calculate_total_face_area(self):
        pass

    # Build the node-face connectivity array.
    def build_node_face_connectivity(self):
        pass

    # Build the edge-face connectivity array.
    def build_edge_face_connectivity(self):
        pass

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        pass

    # Validate that the grid conforms to the UXGrid standards.
    def validate(self):
        pass

    def write(self, outfile, format=""):
        if format == "ugrid":
            self.write_ugrid(outfile)
        else:
            print("format not supported for writing")
