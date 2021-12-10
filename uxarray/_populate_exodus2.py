import xarray as xr


def populate_exo2_data(self, ds):
    print("populating exo2 data..")
    ds.Mesh2.attrs['topology_dimension'] = ds.dims['num_dim']
    ds["Mesh2_node_x"] = xr.DataArray(
        data=ds.coord[0],
        dims=["nMesh2_node"],
        attrs={
            # nothing specified set to cartesian
            "standard_name": "cartesian",
            "long_name": ds.title,
            # nothing specified set to m
            "units": "m",
        })
    ds["Mesh2_node_y"] = xr.DataArray(
        data=ds.coord[1],
        dims=["nMesh2_node"],
        attrs={
            # nothing specified set to cartesian
            "standard_name": "cartesian",
            "long_name": ds.title,
            # nothing specified set to m
            "units": "m",
        })
    ds["Mesh2_node_z"] = xr.DataArray(
        data=ds.coord[2],
        dims=["nMesh2_node"],
        attrs={
            # nothing specified set to cartesian
            "standard_name": "cartesian",
            "long_name": ds.title,
            # nothing specified set to m
            "units": "m",
        })
