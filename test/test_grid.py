import sys
from unittest import TestCase

from pathlib import Path
import os

import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    import uxarray as ux
else:
    import uxarray as ux


class test_grid(TestCase):

    def test_load_exofile(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        exo_filename = current_path / "meshfiles" / "hex_2x2x2_ss.exo"
        tgrid = ux.Grid(str(exo_filename))

    def test_rename(self):

        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        exo_filename = current_path / "meshfiles" / "hex_2x2x2_ss.exo"
        tgrid = ux.Grid(str(exo_filename))

        # check rename filename function
        new_filename = "1hex.exo"
        tgrid.rename_file(new_filename)
        new_filepath = current_path / "meshfiles" / new_filename
        assert (tgrid.filepath == str(new_filepath))

    def test_load_exo2file(self):

        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.Grid(str(exo2_filename))
        outfile = current_path / "write_test_outCSne8.ug"
        tgrid.write(str(outfile))

    def test_load_scrip(self):

        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        scrip_filename = current_path / "meshfiles" / "outCSne8.nc"
        tgrid = ux.Grid(str(scrip_filename))

    def test_load_ugrid(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        ugrid_file = current_path / "meshfiles" / "sphere_mixed.1.lb8.ugrid"
        tgrid = ux.Grid(ugrid_file)

    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
    def test_load_shpfile(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        shp_filename = current_path / "meshfiles" / "grid_file.shp"
        tgrid = ux.Grid(shp_filename)

    def test_load_uxarray(self):

        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        tgrid1 = ux.Grid(str(ug_filename1))
        tgrid2 = ux.Grid(str(ug_filename2))
        tgrid3 = ux.Grid(str(ug_filename3))
        # TODO: add checks after loading this native file format