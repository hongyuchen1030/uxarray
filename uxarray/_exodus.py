import xarray as xr


# Exodus Number is one-based.
def read_exodus(self, ds):
    print("read_exodus called..")
    for key, value in ds.variables.items():
        if key == "qa_records":
            # print(value)
            pass
        elif key == "cord":
            ds.Mesh2.attrs['topology_dimension'] = ds.dims['num_dim']
            ds["Mesh2_node_x"] = xr.DataArray(
                data=ds.coord[0],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
            ds["Mesh2_node_y"] = xr.DataArray(
                data=ds.coord[1],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
            ds["Mesh2_node_z"] = xr.DataArray(
                data=ds.coord[2],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
            # TODO: remove all exodus DVs
            # ds = ds.drop("coord")
        elif key == "coordx":
            ds["Mesh2_node_x"] = xr.DataArray(
                data=ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "coordy":
            ds["Mesh2_node_y"] = xr.DataArray(
                data=ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "coordz":
            ds["Mesh2_node_z"] = xr.DataArray(
                data=ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "connect1":
            for k, v in value.attrs.items():
                if k == "elem_type":
                    etype = v
            ds["Mesh2_face_nodes"] = xr.DataArray(
                data=(ds.connect1[:] - 1),
                dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
                attrs={
                    "cf_role": "face_node_connectivity",
                    "_FillValue": -1,
                    "start_index":
                        0  #NOTE: This might cause an error if numbering has holes
                })


def write_exodus(self, ds, filename):
    # Note this is 1-based unlike native Mesh2 construct
    print("writing exodus file: ", filename)
    # check if data variable has Mesh2 construct
    # if not, throw a message that mesh isn't in native uxarray format before write called
    # if Mesh2 construct

    # write header

    # from mesh2
    # dimen = ds["Mesh2"].topology_dimension
    # ds = ds.expand_dims({"num_dim":dimen})
    # print(ds)

    # # ds.dims["num_dim"].assign()

    # # ds.dims["num_dim"] = ds["Mesh2"].topology_dimension
    # self.grid_ds.dims["num_nodes"] = ds.dims["nMesh2_node"]
    # self.grid_ds.dims["num_elem"] = ds.dims["nMesh2_face"]
    # self.grid_ds.dims["num_el_blk"] = 1 # set to one now #ds.dims["nMesh2_nodes"]
    # write qa header/string
    # write nodes
    # write element blocks
    #
