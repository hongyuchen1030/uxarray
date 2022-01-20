def read_ugrid(self, filename):
    print("read_ugrid called: ", filename)
    # return simple data from xarray load
    return self.grid_ds


# Write a uxgrid to a file with specified format.
def write_ugrid(self, outfile):
    self.grid_ds.to_netcdf(outfile)
