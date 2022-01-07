import xarray as xr


@xr.register_dataarray_accessor('ux_integrate')
class IntegrateAccessor:

    def __init__(self, dataarray):
        self.dataarray = dataarray

    def __call__(self):
        print("accessor called")
