import xarray as xr


@xr.register_dataarray_accessor('ux_integrate')
class IntegrateAccessor:

    def __init__(self, dataarray):
        self.dataarray = dataarray

    def __call__(self):
        print("accessor called")


@xr.register_dataset_accessor('integrate2')
class IntegrateAccessor:

    def __init__(self, dataarray):
        self.dataarray = dataarray

    def __call__(self):
        print("accessor2 called")
