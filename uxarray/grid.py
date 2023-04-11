"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
import copy
from intervaltree import Interval, IntervalTree
# reader and writer imports
from ._exodus import _read_exodus, _encode_exodus
from ._ugrid import _read_ugrid, _encode_ugrid
from ._shapefile import _read_shpfile
from ._scrip import _read_scrip, _encode_scrip
from ._mpas import _read_mpas
from .helpers import get_all_face_area_from_coords, parse_grid_type, _convert_node_xyz_to_lonlat_rad, _convert_node_lonlat_rad_to_xyz, _normalize_in_place, _within, _get_radius_of_latitude_rad, get_intersection_pt, _close_face_nodes
from .constants import INT_DTYPE, INT_FILL_VALUE
from ._latlonbound_utilities import insert_pt_in_latlonbox, get_intersection_point_gcr_gcr

class Grid:
    """
    Examples
    ----------

    Open an exodus file with Uxarray Grid object

    >>> xarray_obj = xr.open_dataset("filename.g")
    >>> mesh = ux.Grid(xarray_obj)

    Encode as a `xarray.Dataset` in the UGRID format

    >>> mesh.encode_as("ugrid")
    """

    def __init__(self, dataset, **kwargs):
        """Initialize grid variables, decide if loading happens via file, verts
        or gridspec.

        Parameters
        ----------
        dataset : xarray.Dataset, ndarray, list, tuple, required
            Input xarray.Dataset or vertex coordinates that form one face.

        Other Parameters
        ----------------
        islatlon : bool, optional
            Specify if the grid is lat/lon based
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_type: str, optional
            Specify the mesh file type, eg. exo, ugrid, shp, mpas, etc
        use_dual: bool, optional
            Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas
        Raises
        ------
            RuntimeError
                If specified file not found
        """
        # initialize internal variable names
        self.__init_ds_var_names__()

        # initialize face_area variable
        self._face_areas = None

        # TODO: fix when adding/exercising gridspec

        # unpack kwargs
        # sets default values for all kwargs to None
        kwargs_list = [
            'gridspec', 'vertices', 'islatlon', 'concave', 'source_grid',
            'use_dual'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        # check if initializing from verts:
        if isinstance(dataset, (list, tuple, np.ndarray)):
            dataset = np.asarray(dataset)
            # grid with multiple faces
            if dataset.ndim == 3:
                self.__from_vert__(dataset)
                self.source_grid = "From vertices"
            # grid with a single face
            elif dataset.ndim == 2:
                dataset = np.array([dataset])
                self.__from_vert__(dataset)
                self.source_grid = "From vertices"
            else:
                raise RuntimeError(
                    f"Invalid Input Dimension: {dataset.ndim}. Expected dimension should be "
                    f"3: [nMesh2_face, nMesh2_node, Two/Three] or 2 when only "
                    f"one face is passed in.")
        # check if initializing from string
        # TODO: re-add gridspec initialization when implemented
        elif isinstance(dataset, xr.Dataset):
            self.mesh_type = parse_grid_type(dataset)
            self.__from_ds__(dataset=dataset)
        else:
            raise RuntimeError("Dataset is not a valid input type.")

        # initialize convenience attributes
        self.__init_grid_var_attrs__()

    def __init_ds_var_names__(self):
        """Populates a dictionary for storing uxarray's internal representation
        of xarray object.

        Note ugrid conventions are flexible with names of variables, see:
        http://ugrid-conventions.github.io/ugrid-conventions/
        """
        self.ds_var_names = {
            "Mesh2": "Mesh2",
            "Mesh2_node_x": "Mesh2_node_x",
            "Mesh2_node_y": "Mesh2_node_y",
            "Mesh2_node_z": "Mesh2_node_z",
            "Mesh2_face_nodes": "Mesh2_face_nodes",
            # initialize dims
            "nMesh2_node": "nMesh2_node",
            "nMesh2_face": "nMesh2_face",
            "nMaxMesh2_face_nodes": "nMaxMesh2_face_nodes"
        }

    def __init_grid_var_attrs__(self) -> None:
        """Initialize attributes for directly accessing UGRID dimensions and
        variables.

        Examples
        ----------
        Assuming the mesh node coordinates for longitude are stored with an input
        name of 'mesh_node_x', we store this variable name in the `ds_var_names`
        dictionary with the key 'Mesh2_node_x'. In order to access it, we do:

        >>> x = grid.ds[grid.ds_var_names["Mesh2_node_x"]]

        With the help of this function, we can directly access it through the
        use of a standardized name based on the UGRID conventions

        >>> x = grid.Mesh2_node_x
        """

        # Iterate over dict to set access attributes
        for key, value in self.ds_var_names.items():
            # Set Attributes for Data Variables
            if self.ds.data_vars is not None:
                if value in self.ds.data_vars:
                    setattr(self, key, self.ds[value])

            # Set Attributes for Coordinates
            if self.ds.coords is not None:
                if value in self.ds.coords:
                    setattr(self, key, self.ds[value])

            # Set Attributes for Dimensions
            if self.ds.dims is not None:
                if value in self.ds.dims:
                    setattr(self, key, len(self.ds[value]))

    def __from_vert__(self, dataset):
        """Create a grid with faces constructed from vertices specified by the
        given argument.

        Parameters
        ----------
        dataset : ndarray, list, tuple, required
            Input vertex coordinates that form our face(s)
        """
        self.ds = xr.Dataset()
        self.ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })
        self.ds.Mesh2.attrs['topology_dimension'] = dataset.ndim

        # set default coordinate units to spherical coordinates
        # users can change to cartesian if using cartesian for initialization
        x_units = "degrees_east"
        y_units = "degrees_north"
        if dataset[0][0].size > 2:
            z_units = "elevation"
        x_coord = dataset[:, :, 0].flatten()
        y_coord = dataset[:, :, 1].flatten()
        if dataset[0][0].size > 2:
            z_coord = dataset[:, :, 2].flatten()

        # Identify unique vertices and their indices
        unique_verts, indices = np.unique(dataset.reshape(
            -1, dataset.shape[-1]),
                                          axis=0,
                                          return_inverse=True)

        # Nodes index that contain a fill value
        fill_value_mask = np.logical_or(unique_verts[:, 0] == INT_FILL_VALUE,
                                        unique_verts[:, 1] == INT_FILL_VALUE)
        if dataset[0][0].size > 2:
            fill_value_mask = np.logical_or(
                unique_verts[:, 0] == INT_FILL_VALUE,
                unique_verts[:, 1] == INT_FILL_VALUE,
                unique_verts[:, 2] == INT_FILL_VALUE)

        # Get the indices of all the False values in fill_value_mask
        false_indices = np.where(fill_value_mask == True)[0]

        # Check if any False values were found
        indices = indices.astype(INT_DTYPE)
        if false_indices.size > 0:

            # Remove the rows corresponding to False values in unique_verts
            unique_verts = np.delete(unique_verts, false_indices, axis=0)

            # Update indices accordingly
            for i, idx in enumerate(false_indices):
                fill_value = INT_FILL_VALUE
                indices[indices == idx] = INT_FILL_VALUE
                indices[(indices > idx) & (indices != fill_value)] -= 1

        # Create coordinate DataArrays
        self.ds["Mesh2_node_x"] = xr.DataArray(data=unique_verts[:, 0],
                                               dims=["nMesh2_node"],
                                               attrs={"units": x_units})
        self.ds["Mesh2_node_y"] = xr.DataArray(data=unique_verts[:, 1],
                                               dims=["nMesh2_node"],
                                               attrs={"units": y_units})
        if dataset.shape[-1] > 2:
            self.ds["Mesh2_node_z"] = xr.DataArray(data=unique_verts[:, 2],
                                                   dims=["nMesh2_node"],
                                                   attrs={"units": z_units})

        # Create connectivity array using indices of unique vertices
        connectivity = indices.reshape(dataset.shape[:-1])
        self.ds["Mesh2_face_nodes"] = xr.DataArray(
            data=xr.DataArray(connectivity).astype(INT_DTYPE),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role": "face_node_connectivity",
                "_FillValue": INT_FILL_VALUE,
                "start_index": 0
            })

    # load mesh from a file
    def __from_ds__(self, dataset):
        """Loads a mesh dataset."""
        # call reader as per mesh_type
        if self.mesh_type == "exo":
            self.ds = _read_exodus(dataset, self.ds_var_names)
        elif self.mesh_type == "scrip":
            self.ds = _read_scrip(dataset)
        elif self.mesh_type == "ugrid":
            self.ds, self.ds_var_names = _read_ugrid(dataset, self.ds_var_names)
        elif self.mesh_type == "shp":
            self.ds = _read_shpfile(dataset)
        elif self.mesh_type == "mpas":
            # select whether to use the dual mesh
            if self.use_dual is not None:
                self.ds = _read_mpas(dataset, self.use_dual)
            else:
                self.ds = _read_mpas(dataset)
        else:
            raise RuntimeError("unknown mesh type")

        dataset.close()

    def encode_as(self, grid_type):
        """Encodes the grid as a new `xarray.Dataset` per grid format supplied
        in the `grid_type` argument.

        Parameters
        ----------
        grid_type : str, required
            Grid type of output dataset.
            Currently supported options are "ugrid", "exodus", and "scrip"

        Returns
        -------
        out_ds : xarray.Dataset
            The output `xarray.Dataset` that is encoded from the this grid.

        Raises
        ------
        RuntimeError
            If provided grid type or file type is unsupported.
        """

        if grid_type == "ugrid":
            out_ds = _encode_ugrid(self.ds)

        elif grid_type == "exodus":
            out_ds = _encode_exodus(self.ds, self.ds_var_names)

        elif grid_type == "scrip":
            out_ds = _encode_scrip(self.Mesh2_face_nodes, self.Mesh2_node_x,
                                   self.Mesh2_node_y, self.face_areas)
        else:
            raise RuntimeError("The grid type not supported: ", grid_type)

        return out_ds

    def calculate_total_face_area(self, quadrature_rule="triangular", order=4):
        """Function to calculate the total surface area of all the faces in a
        mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Sum of area of all the faces in the mesh : float
        """

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        return np.sum(face_areas)

    def compute_face_areas(self, quadrature_rule="triangular", order=4):
        """Face areas calculation function for grid class, calculates area of
        all faces in the grid.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Area of all the faces in the mesh : np.ndarray

        Examples
        --------
        Open a uxarray grid file

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/ugrid/outCSne30/outCSne30.ug")

        Get area of all faces in the same order as listed in grid.ds.Mesh2_face_nodes

        >>> grid.get_face_areas
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        if self._face_areas is None:
            # area of a face call needs the units for coordinate conversion if spherical grid is used
            coords_type = "spherical"
            if not "degree" in self.Mesh2_node_x.units:
                coords_type = "cartesian"

            face_nodes = self.Mesh2_face_nodes.data
            dim = self.Mesh2.attrs['topology_dimension']

            # initialize z
            z = np.zeros((self.nMesh2_node))

            # call func to cal face area of all nodes
            x = self.Mesh2_node_x.data
            y = self.Mesh2_node_y.data
            # check if z dimension
            if self.Mesh2.topology_dimension > 2:
                z = self.Mesh2_node_z.data

            # call function to get area of all the faces as a np array
            self._face_areas = get_all_face_area_from_coords(
                x, y, z, face_nodes, dim, quadrature_rule, order, coords_type)

        return self._face_areas

    # use the property keyword for declaration on face_areas property
    @property
    def face_areas(self):
        """Declare face_areas as a property."""

        if self._face_areas is None:
            self.compute_face_areas()
        return self._face_areas

    def integrate(self, var_ds, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the given mesh.

        Parameters
        ----------
        var_ds : Xarray dataset, required
            Xarray dataset containing values to integrate on this grid
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open grid file only

        >>> xr_grid = xr.open_dataset("grid.ug")
        >>> grid = ux.Grid.(xr_grid)
        >>> var_ds = xr.open_dataset("centroid_pressure_data_ug")

        # Compute the integral
        >>> integral_psi = grid.integrate(var_ds)
        """
        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        var_key = list(var_ds.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The xarray dataset file has more than one variable, using the first variable for integration"
            )
        var_key = var_key[0]
        face_vals = var_ds[var_key].to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral

    def _populate_cartesian_xyz_coord(self):
        """A helper function that populates the xyz attribute in UXarray.ds.
        This function is called when we need to use the cartesian coordinates
        for each node to do the calculation but the input data only has the
        "Mesh2_node_x" and "Mesh2_node_y" in degree.

        Note
        ----
        In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
        corresponding units as stated below:

        Mesh2_node_x
         unit:  "degree_east" for longitude
        Mesh2_node_y
         unit:  "degrees_north" for latitude
        Mesh2_node_z
         unit:  "m"
        Mesh2_node_cart_x
         unit:  "m"
        Mesh2_node_cart_y
         unit:  "m"
        Mesh2_node_cart_z
         unit:  "m"
        """

        # Check if the cartesian coordinates are already populated
        if "Mesh2_node_cart_x" in self.ds.keys():
            return

        # check for units and create Mesh2_node_cart_x/y/z set to self.ds
        nodes_lon_rad = np.deg2rad(self.Mesh2_node_x.values)
        nodes_lat_rad = np.deg2rad(self.Mesh2_node_y.values)
        nodes_rad = np.stack((nodes_lon_rad, nodes_lat_rad), axis=1)
        nodes_cart = np.asarray(
            list(map(_convert_node_lonlat_rad_to_xyz, list(nodes_rad))))

        self.ds["Mesh2_node_cart_x"] = xr.DataArray(
            data=nodes_cart[:, 0],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian x",
                "units": "m",
            })
        self.ds["Mesh2_node_cart_y"] = xr.DataArray(
            data=nodes_cart[:, 1],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian y",
                "units": "m",
            })
        self.ds["Mesh2_node_cart_z"] = xr.DataArray(
            data=nodes_cart[:, 2],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian z",
                "units": "m",
            })

    def _populate_lonlat_coord(self):
        """Helper function that populates the longitude and latitude and store
        it into the Mesh2_node_x and Mesh2_node_y. This is called when the
        input data has "Mesh2_node_x", "Mesh2_node_y", "Mesh2_node_z" in
        meters. Since we want "Mesh2_node_x" and "Mesh2_node_y" always have the
        "degree" units. For more details, please read the following.

        Raises
        ------
            RuntimeError
                Mesh2_node_x/y/z are not represented in the cartesian format with the unit 'm'/'meters' when calling this function"

        Note
        ----
        In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
        corresponding units as stated below:

        Mesh2_node_x
         unit:  "degree_east" for longitude
        Mesh2_node_y
         unit:  "degrees_north" for latitude
        Mesh2_node_z
         unit:  "m"
        Mesh2_node_cart_x
         unit:  "m"
        Mesh2_node_cart_y
         unit:  "m"
        Mesh2_node_cart_z
         unit:  "m"
        """

        # Check if the "Mesh2_node_x" is already in longitude
        if "degree" in self.ds.Mesh2_node_x.units:
            return

        # Check if the input Mesh2_node_xyz" are represented in the cartesian format with the unit "m"
        if ("m" not in self.ds.Mesh2_node_x.units) or ("m" not in self.ds.Mesh2_node_y.units) \
                or ("m" not in self.ds.Mesh2_node_z.units):
            raise RuntimeError(
                "Expected: Mesh2_node_x/y/z should be represented in the cartesian format with the "
                "unit 'm' when calling this function")

        # Put the cartesian coordinates inside the proper data structure
        self.ds["Mesh2_node_cart_x"] = xr.DataArray(
            data=self.ds["Mesh2_node_x"].values)
        self.ds["Mesh2_node_cart_y"] = xr.DataArray(
            data=self.ds["Mesh2_node_y"].values)
        self.ds["Mesh2_node_cart_z"] = xr.DataArray(
            data=self.ds["Mesh2_node_z"].values)

        # convert the input cartesian values into the longitude latitude degree
        nodes_cart = np.stack(
            (self.ds["Mesh2_node_x"].values, self.ds["Mesh2_node_y"].values,
             self.ds["Mesh2_node_z"].values),
            axis=1).tolist()
        nodes_rad = list(map(_convert_node_xyz_to_lonlat_rad, nodes_cart))
        nodes_degree = np.rad2deg(nodes_rad)
        self.ds["Mesh2_node_x"] = xr.DataArray(
            data=nodes_degree[:, 0],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "longitude",
                "long_name": "longitude of mesh nodes",
                "units": "degrees_east",
            })
        self.ds["Mesh2_node_y"] = xr.DataArray(
            data=nodes_degree[:, 1],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "lattitude",
                "long_name": "latitude of mesh nodes",
                "units": "degrees_north",
            })


    def _build_edge_node_connectivity(self):
        """Constructs the UGRID connectivity variable (``Mesh2_edge_nodes``)
        and stores it within the internal (``Grid.ds``) and through the
        attribute (``Grid.Mesh2_edge_nodes``).
        Additionally, the attributes (``inverse_indices``) and
        (``fill_value_mask``) are stored for constructing other
        connectivity variables.
        """

        # padded face nodes: [nMesh2_face x nMaxMesh2_face_nodes + 1]
        padded_face_nodes = _close_face_nodes(self.Mesh2_face_nodes.values,
                                              self.nMesh2_face,
                                              self.nMaxMesh2_face_nodes)

        # construct an array of empty edge nodes where each entry is a pair of indices
        edge_nodes = np.empty((self.nMesh2_face * self.nMaxMesh2_face_nodes, 2),
                              dtype=INT_DTYPE)

        # first index includes starting node up to non-padded value
        edge_nodes[:, 0] = padded_face_nodes[:, :-1].ravel()

        # second index includes second node up to padded value
        edge_nodes[:, 1] = padded_face_nodes[:, 1:].ravel()

        # all edge nodes that contain a fill value
        fill_value_mask = np.logical_or(edge_nodes[:, 0] == INT_FILL_VALUE,
                                        edge_nodes[:, 1] == INT_FILL_VALUE)

        # all edge nodes that do not contain a fill value
        non_fill_value_mask = np.logical_not(fill_value_mask)

        # filter out all invalid edges
        edge_nodes = edge_nodes[non_fill_value_mask]

        # sorted edge nodes
        edge_nodes.sort(axis=1)

        # unique edge nodes
        edge_nodes_unique, inverse_indices = np.unique(edge_nodes,
                                                       return_inverse=True,
                                                       axis=0)
        # add mesh2_edge_nodes to internal dataset
        self.ds['Mesh2_edge_nodes'] = xr.DataArray(
            edge_nodes_unique,
            dims=["nMesh2_edge", "Two"],
            attrs={
                "cf_role":
                    "edge_node_connectivity",
                "long_name":
                    "Maps every edge to the two nodes that it connects",
                "start_index":
                    INT_DTYPE(0),
                "inverse_indices":
                    inverse_indices,
                "fill_value_mask":
                    fill_value_mask
            })

        setattr(self, "Mesh2_edge_nodes", self.ds['Mesh2_edge_nodes'])
        setattr(self, "nMesh2_edge", edge_nodes_unique.shape[0])

    def build_face_edges_connectivity(self):
        """A DataArray of indices indicating edges that are neighboring each
        face.
        Notes
        -----
        This function will add `Grid.ds.Mesh2_face_edges` to the `Grid` class, which is an integer
        DataArray of size (nMesh2_face, MaxNumNodesPerFace)
        """
        padded_face_nodes = _close_face_nodes(self.Mesh2_face_nodes.values,
                                              self.nMesh2_face,
                                              self.nMaxMesh2_face_nodes)

        # construct an array of empty edge nodes where each entry is a pair of indices
        edge_nodes = np.empty((self.nMesh2_face * self.nMaxMesh2_face_nodes, 2),
                              dtype=INT_DTYPE)

        # first index includes starting node up to non-padded value
        edge_nodes[:, 0] = padded_face_nodes[:, :-1].ravel()

        # second index includes second node up to padded value
        edge_nodes[:, 1] = padded_face_nodes[:, 1:].ravel()

        # all edge nodes that contain a fill value
        fill_value_mask = np.logical_or(edge_nodes[:, 0] == INT_FILL_VALUE,
                                        edge_nodes[:, 1] == INT_FILL_VALUE)

        # all edge nodes that do not contain a fill value
        non_fill_value_mask = np.logical_not(fill_value_mask)

        # filter out all invalid edges
        edge_nodes = edge_nodes[non_fill_value_mask]

        # sorted edge nodes
        edge_nodes.sort(axis=1)

        # unique edge nodes
        edge_nodes_unique, inverse_indices = np.unique(edge_nodes,
                                                       return_inverse=True,
                                                       axis=0)

        # In mesh2_edge_nodes, we want to remove all dummy edges (edge that has "INT_FILL_VALUE" node index)
        # But we want to preserve that in our mesh2_face_edges so make the datarray has the same dimensions
        has_fill_value = np.logical_or(edge_nodes_unique[:, 0] == INT_FILL_VALUE,
              edge_nodes_unique[:, 1] == INT_FILL_VALUE)
        mesh2_edge_nodes = edge_nodes_unique[~has_fill_value]
        inverse_indices = inverse_indices.reshape(self.nMesh2_face, self.nMaxMesh2_face_nodes)
        mesh2_face_edges = inverse_indices # We only need to store the edge index

        self.ds["Mesh2_face_edges"] = xr.DataArray(
            data=mesh2_face_edges,
            dims=["nMesh2_face", "nMaxMesh2_face_edges"],
            attrs={
                "cf_role": "face_edges_connectivity",
                "start_index": 0
            })
        self.ds["Mesh2_edge_nodes"] = xr.DataArray(data=mesh2_edge_nodes,
                                                   dims=["nMesh2_edge", "Two"])

    def buildlatlon_bounds(self):

        # First make sure the Grid object has the Mesh2_face_edges

        if "Mesh2_face_edges" not in self.ds.keys():
            self.build_face_edges_connectivity()

        if "Mesh2_node_cart_x" not in self.ds.keys():
            self._populate_cartesian_xyz_coord()

        # All value are inialized as 404.0 to indicate that they're null
        temp_latlon_array = [[[404.0, 404.0], [404.0, 404.0]]
                             ] * self.ds["Mesh2_face_edges"].sizes["nMesh2_face"]

        reference_tolerance = 1.0e-12

        # Build an Interval tree based on the Latitude interval to store latitude-longitude boundaries
        self._latlonbound_tree = IntervalTree()

        for i in range(0, len(self.ds["Mesh2_face_nodes"])):
            face_edges = [[0, 0]] * len(self.ds["Mesh2_face_nodes"][i])
            for j in range(1, len(self.ds["Mesh2_face_nodes"][i])):
                face_edges[j - 1] = [self.ds["Mesh2_face_nodes"].values[i][j - 1],
                                     self.ds["Mesh2_face_nodes"].values[i][j]]
            face_edges[len(self.ds["Mesh2_face_nodes"][i]) - 1] = [
                self.ds["Mesh2_face_nodes"].values[i][len(self.ds["Mesh2_face_nodes"][i]) - 1],
                self.ds["Mesh2_face_nodes"].values[i][0]]
            # Check if face_edges contains pole points
            _lambda = 0
            v1 = [0, 0, 1]
            v2 = _normalize_in_place([np.cos(_lambda), np.sin(_lambda), -1.0])

            num_intersects = self.__count_face_edge_intersection(face_edges, [v1, v2], i)
            if num_intersects == -1:
                # if one edge of the grid cell is parallel to the arc (v1 , v2 ) then vary the choice of v2 .
                sorted_edges_avg_lon = self.__avg_edges_longitude(face_edges)
                # only need to iterate the first two keys to average the longitude:
                sum_lon = sorted_edges_avg_lon[0] + sorted_edges_avg_lon[1]

                v2 = [np.cos(sum_lon / 2), np.sin(sum_lon / 2), 0]
                num_intersects = self.__count_face_edge_intersection(
                    face_edges, [v1, v2])

            if num_intersects % 2 != 0:
                if i==0 and j == 3:
                    pass
                # if the face_edges contains the pole point
                for j in range(0, len(face_edges)):
                    edge = face_edges[j]
                    # Skip the dummy edges
                    if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                        continue
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info

                    # Get the edge end points in 3D [x, y, z] coordinates
                    n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                          self.ds["Mesh2_node_cart_y"].values[edge[0]],
                          self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                    n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                          self.ds["Mesh2_node_cart_y"].values[edge[1]],
                          self.ds["Mesh2_node_cart_z"].values[edge[1]]]

                    # Set the latitude extent
                    d_lat_extent_rad = 0.0
                    if j == 0:
                        if n1[2] < 0.0:
                            d_lat_extent_rad = -0.5 * np.pi
                        else:
                            d_lat_extent_rad = 0.5 * np.pi

                    # insert edge endpoint into box
                    if np.absolute(self.ds["Mesh2_node_y"].values[
                                       edge[0]]) < d_lat_extent_rad:
                        d_lat_extent_rad = self.ds["Mesh2_node_y"].values[
                            edge[0]]

                    # Determine if latitude is maximized between endpoints
                    dot_n1_n2 = np.dot(n1, n2)
                    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)
                    if np.absolute(d_de_nom) < reference_tolerance:
                        continue

                    d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / d_de_nom
                    if (d_a_max > 0.0) and (d_a_max < 1.0):
                        node3 = [0.0, 0.0, 0.0]
                        node3[0] = n1[0] * (1 - d_a_max) + n2[0] * d_a_max
                        node3[1] = n1[1] * (1 - d_a_max) + n2[1] * d_a_max
                        node3[2] = n1[2] * (1 - d_a_max) + n2[2] * d_a_max
                        node3 = _normalize_in_place(node3)

                        d_lat_rad = node3[2]

                        if d_lat_rad > 1.0:
                            d_lat_rad = 0.5 * np.pi
                        elif d_lat_rad < -1.0:
                            d_lat_rad = -0.5 * np.pi
                        else:
                            d_lat_rad = np.arcsin(d_lat_rad)

                        if np.absolute(d_lat_rad) < np.absolute(
                                d_lat_extent_rad):
                            d_lat_extent_rad = d_lat_rad

                    if d_lat_extent_rad < 0.0:
                        lon_list = [0.0, 2.0 * np.pi]
                        lat_list = [-0.5 * np.pi, d_lat_extent_rad]
                    else:
                        lon_list = [0.0, 2.0 * np.pi]
                        lat_list = [d_lat_extent_rad, 0.5 * np.pi]

                    temp_latlon_array[i] = [lat_list, lon_list]
            else:
                # normal face_edges
                for j in range(0, len(face_edges)):
                    if i == 0 and j == 3:
                        pass
                    edge = face_edges[j]
                    # Skip the dummy edges
                    if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                        continue

                    # For each edge, we only need to consider the first end point in each loop
                    # Check if the end point is the pole point
                    n1 = [self.ds["Mesh2_node_x"].values[edge[0]],
                          self.ds["Mesh2_node_y"].values[edge[0]]]

                    # North Pole:
                    if (np.absolute(n1[0] - 0) < reference_tolerance and np.absolute(
                            n1[1] - 90) < reference_tolerance) or (
                            np.absolute(n1[0] - 180) < reference_tolerance and np.absolute(
                        n1[1] - 90) < reference_tolerance):
                        # insert edge endpoint into box
                        d_lat_rad = np.deg2rad(
                            self.ds["Mesh2_node_y"].values[edge[0]])
                        d_lon_rad = 404.0
                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])
                        continue

                    # South Pole:
                    if (np.absolute(n1[0] - 0) < reference_tolerance and np.absolute(
                            n1[1] - (-90)) < reference_tolerance) or (
                            np.absolute(n1[0] - 180) < reference_tolerance and np.absolute(
                        n1[1] - (-90)) < reference_tolerance):
                        d_lat_rad = np.deg2rad(
                            self.ds["Mesh2_node_y"].values[edge[0]])
                        d_lon_rad = 404.0
                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])
                        continue

                    # Only consider the great circles arcs
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info

                    # Get the edge end points in 3D [x, y, z] coordinates
                    n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                          self.ds["Mesh2_node_cart_y"].values[edge[0]],
                          self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                    n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                          self.ds["Mesh2_node_cart_y"].values[edge[1]],
                          self.ds["Mesh2_node_cart_z"].values[edge[1]]]

                    # Determine if latitude is maximized between endpoints
                    # TODO: Replace this with the get_gcr_max_lat_rad function
                    dot_n1_n2 = np.dot(n1, n2)
                    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

                    # insert edge endpoint into box
                    d_lat_rad = np.deg2rad(
                        self.ds["Mesh2_node_y"].values[edge[0]])
                    d_lon_rad = np.deg2rad(
                        self.ds["Mesh2_node_x"].values[edge[0]])
                    temp_latlon_array[i] = insert_pt_in_latlonbox(
                        copy.deepcopy(temp_latlon_array[i]),
                        [d_lat_rad, d_lon_rad])

                    if np.absolute(d_de_nom) < reference_tolerance:
                        continue

                    # Maximum latitude occurs between endpoints of edge
                    d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / d_de_nom
                    if 0.0 < d_a_max < 1.0:
                        node3 = [0.0, 0.0, 0.0]
                        node3[0] = n1[0] * (1 - d_a_max) + n2[0] * d_a_max
                        node3[1] = n1[1] * (1 - d_a_max) + n2[1] * d_a_max
                        node3[2] = n1[2] * (1 - d_a_max) + n2[2] * d_a_max
                        node3 = _normalize_in_place(node3)

                        d_lat_rad = node3[2]

                        if d_lat_rad > 1.0:
                            d_lat_rad = 0.5 * np.pi
                        elif d_lat_rad < -1.0:
                            d_lat_rad = -0.5 * np.pi
                        else:
                            d_lat_rad = np.arcsin(d_lat_rad)

                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])
            if temp_latlon_array[i][0][0] == temp_latlon_array[i][0][1]:
                pass
            if temp_latlon_array[i][1][0] == temp_latlon_array[i][1][1]:
                pass
            assert temp_latlon_array[i][0][0] != temp_latlon_array[i][0][1]
            assert temp_latlon_array[i][1][0] != temp_latlon_array[i][1][1]
            lat_list = temp_latlon_array[i][0]
            lon_list = temp_latlon_array[i][1]
            self._latlonbound_tree[lat_list[0]:lat_list[1]] = i

        self.ds["Mesh2_latlon_bounds"] = xr.DataArray(
            data=temp_latlon_array, dims=["nMesh2_face", "Latlon", "Two"])

        # Helper function to get the average longitude of each edge in sorted order (ascending0

    def __avg_edges_longitude(self, face):
        """Helper function to get the average longitude of each edge in sorted order (ascending0
        Parameters
        ----------
        edge_list: 2D float array:
        [[lon, lat],
         [lon, lat]
         ...
         [lon, lat]
        ]
        Returns: 1D float array, record the average longitude of each edge
        """
        edge_list = []
        for edge in face:
            # Skip the dump edges
            if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                continue
            n1 = [
                self.ds["Mesh2_node_x"].values[edge[0]],
                self.ds["Mesh2_node_y"].values[edge[0]]
            ]
            n2 = [
                self.ds["Mesh2_node_x"].values[edge[1]],
                self.ds["Mesh2_node_y"].values[edge[1]]
            ]

            # Since we only want to sort the Edge based on their longitude,
            # We can utilize the Edge class < operator here by creating the Edge only using the longitude
            edge_list.append((n1[0] + n2[0]) / 2)

        edge_list.sort()

        return edge_list

        # Count the number of total intersections of an edge and face (Algo. 2.4 Determining if a grid cell contains a
        # given point)

    def __count_face_edge_intersection(self, face, ref_edge, i=-1):
        """Helper function to count the total number of intersections points
        between the reference edge and a face.
        Parameters
        ----------
        face: xarray.DataArray, list, required
        ref_edge: 2D list, the reference edge that intersect with the face (stored in 3D xyz coordinates) [[x1, y1, z1], [x2, y2, z2]]
        Returns:
        num_intersection: number of intersections
        -1: the ref_edge is parallel to one of the edge of the face and need to vary the ref_edge
        """
        v1 = ref_edge[0]
        v2 = ref_edge[1]
        intersection_set = set()
        num_intersection = 0
        for edge in face:
            # Skip the dump edges
            if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                continue
            # All the following calculation is based on the 3D XYZ coord

            # Convert the 2D [lon, lat] to 3D [x, y, z]
            w1 = _convert_node_lonlat_rad_to_xyz([
                np.deg2rad(self.ds["Mesh2_node_x"].values[edge[0]]),
                np.deg2rad(self.ds["Mesh2_node_y"].values[edge[0]])
            ])
            w2 = _convert_node_lonlat_rad_to_xyz([
                np.deg2rad(self.ds["Mesh2_node_x"].values[edge[1]]),
                np.deg2rad(self.ds["Mesh2_node_y"].values[edge[1]])
            ])

            res = get_intersection_point_gcr_gcr(w1, w2, v1, v2)

            # two vectors are intersected within range and not parralel
            if (res != [0, 0, 0]) and (res != [-1, -1, -1]):
                intersection_set.add(frozenset(np.round(res, decimals=12).tolist()))
                num_intersection += 1
            elif res[0] == 0 and res[1] == 0 and res[2] == 0:
                # if two vectors are parallel
                return -1

        # If the intersection point number is 1, make sure the gcr is not going through a vertex of the face
        # In this situation, the intersection number will be 0 because the gcr doesn't go across the face technically
        if len(intersection_set) == 1:
            intersection_pt = intersection_set.pop()
            for edge in face:
                if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                    continue
                w1 = _convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(self.ds["Mesh2_node_x"].values[edge[0]]),
                    np.deg2rad(self.ds["Mesh2_node_y"].values[edge[0]])
                ])
                w2 = _convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(self.ds["Mesh2_node_x"].values[edge[1]]),
                    np.deg2rad(self.ds["Mesh2_node_y"].values[edge[1]])
                ])

                if list(intersection_pt) == w1 or list(intersection_pt) == w2:
                    return 0

        return len(intersection_set)

    # Get the non-conservative zonal average of the input variable
    def get_nc_zonal_avg(self, var_ds, latitude_rad):
        '''
         Algorithm:
            For each face:
                Find the constantLat Arc intersection points with the face:
                How to find:
                    Use the root calculation to get the approximate point location
                    Then based on the approximate results, use the newton-raphson method to

        '''

        if "Mesh2_latlon_bounds" not in self.ds.keys() or "Mesh2_latlon_bounds" is None:
            self.buildlatlon_bounds()

        #  First Get the list of faces that falls into this latitude range
        candidate_faces_index_list = []

        # Search through the interval tree for all the candidates face
        candidate_face_set = self._latlonbound_tree.at(latitude_rad)
        for interval in candidate_face_set:
            candidate_faces_index_list.append(interval.data)
        candidate_faces_weight_list = self._get_zonal_face_weights_at_constlat(candidate_faces_index_list, latitude_rad)
        # Get the candidate face values:
        var_key = list(var_ds.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The xarray dataset file has more than one variable, using the first variable for integration"
            )
        var_key = var_key[0]
        face_vals = var_ds[var_key].to_numpy()
        candidate_faces_vals_list = [0.0] * len(candidate_faces_index_list)
        for i in range(0, len(candidate_faces_index_list)):
            face_index = candidate_faces_index_list[i]
            candidate_faces_vals_list[i] = face_vals[face_index]
        zonal_average = np.dot(candidate_faces_weight_list, candidate_faces_vals_list)
        return zonal_average

    def _get_zonal_face_weights_at_constlat(self, candidate_faces_index_list, latitude_rad):
        # Then calculate the weight of each face
        # First calculate the perimeter this constant latitude circle
        candidate_faces_weight_list = [0.0] * len(candidate_faces_index_list)

        for i in range(0, len(candidate_faces_index_list)):
            face_index = candidate_faces_index_list[i]
            [face_lon_bound_min, face_lon_bound_max] = self.ds["Mesh2_latlon_bounds"].values[face_index][1]
            face_edges = [[0, 0]] * len(self.ds["Mesh2_face_nodes"][i])
            for j in range(1, len(self.ds["Mesh2_face_nodes"][i])):
                face_edges[j - 1] = [self.ds["Mesh2_face_nodes"].values[i][j - 1],
                                     self.ds["Mesh2_face_nodes"].values[i][j]]
            face_edges[len(self.ds["Mesh2_face_nodes"][i]) - 1] = [
                self.ds["Mesh2_face_nodes"].values[i][len(self.ds["Mesh2_face_nodes"][i]) - 1],
                self.ds["Mesh2_face_nodes"].values[i][0]]

            pt_lon_min = 3 * np.pi
            pt_lon_max = -3 * np.pi

            intersections_pts_list_lonlat = []
            for j in range(0, len(face_edges)):
                edge = face_edges[j]
                # Get the edge end points in 3D [x, y, z] coordinates
                n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                      self.ds["Mesh2_node_cart_y"].values[edge[0]],
                      self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                      self.ds["Mesh2_node_cart_y"].values[edge[1]],
                      self.ds["Mesh2_node_cart_z"].values[edge[1]]]
                n1_lonlat = _convert_node_xyz_to_lonlat_rad(n1)
                n2_lonlat = _convert_node_xyz_to_lonlat_rad(n2)
                intersections = get_intersection_pt([n1, n2], latitude_rad)
                if intersections[0] == [-1, -1, -1] and intersections[1] == [-1, -1, -1]:
                    # The constant latitude didn't cross this edge
                    continue
                elif intersections[0] != [-1, -1, -1] and intersections[1] != [-1, -1, -1]:
                    # The constant latitude goes across this edge ( 1 in and 1 out):
                    pts1_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[0])
                    pts2_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[1])
                    intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                    intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))
                else:
                    if intersections[0] != [-1, -1, -1]:
                        pts1_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[0])
                        intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                    else:
                        pts2_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[1])
                        intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))
            if len(intersections_pts_list_lonlat) == 2:
                [pt_lon_min, pt_lon_max] = np.sort(
                    [intersections_pts_list_lonlat[0][0], intersections_pts_list_lonlat[1][0]])
            if face_lon_bound_min < face_lon_bound_max:
                # Normal case
                cur_face_mag_rad = pt_lon_max - pt_lon_min
            else:
                # Longitude wrap-around
                # TODO: Need to think more marginal cases

                if pt_lon_max >= np.pi and pt_lon_min >= np.pi:
                    # They're both on the "left side" of the 0-lon
                    cur_face_mag_rad = pt_lon_max - pt_lon_min
                if pt_lon_max <= np.pi and pt_lon_min <= np.pi:
                    # They're both on the "right side" of the 0-lon
                    cur_face_mag_rad = pt_lon_max - pt_lon_min
                else:
                    # They're at the different side of the 0-lon
                    cur_face_mag_rad = 2 * np.pi - pt_lon_max + pt_lon_min
            if cur_face_mag_rad > np.pi:
                print("At face: " + str(face_index) + "Problematic lat is " + str(
                    latitude_rad) + " And the cur_face_mag_rad is " + str(cur_face_mag_rad))
                # assert(cur_face_mag_rad <= np.pi)

                # Calculate the weight from each face by |intersection line length| / total perimeter
            candidate_faces_weight_list[i] = cur_face_mag_rad

            # Sum up all the weights to get the total
            candidate_faces_weight_list = np.array(candidate_faces_weight_list) / np.sum(candidate_faces_weight_list)
            return candidate_faces_weight_list

        # Sum up all the weights to get the total
        candidate_faces_weight_list = np.array(candidate_faces_weight_list) / np.sum(candidate_faces_weight_list)
        return candidate_faces_weight_list

    def get_conservative_zonal_avg(self, var_key, latitude_rad_range):
        '''
        var_key: varaible key to be averaged
        latitude_rad_range: The query latitude_rad_range [lat_min, lat_max]
                            ranges are inclusive of the lower limit, but non-inclusive of the upper limit
        '''
        #  First Get the list of faces that falls into this latitude range
        candidate_faces_index_list = []
        min_lat, max_lat = latitude_rad_range

        # Search through the interval tree for all the candidates face
        regrid_candidate_face_set = self._latlonbound_tree[
                                    min_lat: max_lat]  # All faces that instersect with this zonal tile
        enveloped_face = self._latlonbound_tree.envelop(min_lat,
                                                        max_lat)  # Faces that are fully enveloped by this zonal tile
        not_enveloped_face = regrid_candidate_face_set - enveloped_face  # Faces that are cut by the zonal tile and need to be regrid.
        for interval in not_enveloped_face:
            candidate_faces_index_list.append(interval.data)

        # Calcuate the weight of each face by its weight
        candidate_faces_weight_list = [0.0] * len(candidate_faces_index_list)
        for face_idx in regrid_candidate_face_set:
            # Reconstruct the face if it's not fully envelope by the latitude range
            face = self.ds["Mesh2_face_edges"].values[face_idx]
            x = self.ds["Mesh2_node_cart_x"].values[face[:, 0]]
            y = self.ds["Mesh2_node_cart_y"].values[face[:, 0]]
            z = self.ds["Mesh2_node_cart_z"].values[face[:, 0]]
            intersections_pts_list_lonlat = []  # Maximum size:4
            for j in range(0, len(face)):
                edge = face[j]
                # Get the edge end points in 3D [x, y, z] coordinates
                n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                      self.ds["Mesh2_node_cart_y"].values[edge[0]],
                      self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                      self.ds["Mesh2_node_cart_y"].values[edge[1]],
                      self.ds["Mesh2_node_cart_z"].values[edge[1]]]

                for lat_bound in latitude_rad_range:
                    intersections = get_intersection_pt([n1, n2], lat_bound)
                    if intersections[0] == [-1, -1, -1] and intersections[1] == [-1, -1, -1]:
                        # The constant latitude didn't cross this edge
                        continue
                    elif intersections[0] != [-1, -1, -1] and intersections[1] != [-1, -1, -1]:
                        # The constant latitude goes across this edge ( 1 in and 1 out):
                        pts1_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[0])
                        pts2_lonlat = _convert_node_xyz_to_lonlat_rad(intersections[1])
                        intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                        intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))
                    else:
                        if intersections[0] != [-1, -1, -1]:
                            intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                        else:
                            intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))

            if len(intersections_pts_list_lonlat) == 2:
                # Only one constant latitude goes through this face
                pass
            elif len(intersections_pts_list_lonlat) == 4:
                # Both constant latitude goes through this face
                pass
            else:
                print("Exception")




