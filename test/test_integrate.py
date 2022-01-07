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


class test_ux_integrate(TestCase):

    def test_open_dataset(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        uds1_name = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"
        uds1 = ux.open_dataset(uds1_name, "ux")
        uds1.Mesh2_face_nodes.ux_integrate()
