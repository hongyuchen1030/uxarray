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
    print("writing exodus file: ", filename)
