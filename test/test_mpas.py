from uxarray._mpas import _replace_padding, _replace_zeros, _to_zero_index
from uxarray._mpas import _read_mpas, _primal_to_ugrid, _dual_to_ugrid
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np
import os
from pathlib import Path

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestMPAS(TestCase):
    """Test suite for Read MPAS functionality."""

    # sample mpas dataset
    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_xr_ds = xr.open_dataset(mpas_grid_path)

    # fill value (remove once there is a unified approach in uxarray)
    fv = INT_FILL_VALUE

    def test_read_mpas(self):
        """Tests execution of _read_mpas()"""
        mpas_primal_ugrid = _read_mpas(self.mpas_xr_ds, use_dual=False)
        mpas_dual_ugrid = _read_mpas(self.mpas_xr_ds, use_dual=True)

    def test_mpas_to_grid(self):
        """Tests creation of Grid object from converted MPAS dataset."""
        mpas_uxgrid_primal = ux.Grid(self.mpas_xr_ds, use_dual=False)
        mpas_uxgrid_primal = ux.Grid(self.mpas_xr_ds, use_dual=True)

    def test_primal_to_ugrid_conversion(self):
        """Verifies that the Primal-Mesh was converted properly."""

        # primal-mesh encoded in the UGRID conventions
        ds = _read_mpas(self.mpas_xr_ds, use_dual=False)

        # check for correct dimensions
        expected_ugrid_dims = [
            'nMesh2_node', "nMesh2_face", "nMaxMesh2_face_nodes"
        ]
        for dim in expected_ugrid_dims:
            assert dim in ds.sizes

        # check for correct length of coordinates
        assert len(ds['Mesh2_node_x']) == len(ds['Mesh2_node_y'])
        assert len(ds['Mesh2_face_x']) == len(ds['Mesh2_face_y'])

        # check for correct shape of face nodes
        nMesh2_face = ds.sizes['nMesh2_face']
        nMaxMesh2_face_nodes = ds.sizes['nMaxMesh2_face_nodes']
        assert ds['Mesh2_face_nodes'].shape == (nMesh2_face,
                                                nMaxMesh2_face_nodes)

    def test_dual_to_ugrid_conversion(self):
        """Verifies that the Dual-Mesh was converted properly."""

        # dual-mesh encoded in the UGRID conventions
        ds = _read_mpas(self.mpas_xr_ds, use_dual=True)

        # check for correct dimensions
        expected_ugrid_dims = ['nMesh2_node', "nMesh2_face", "Three"]
        for dim in expected_ugrid_dims:
            assert dim in ds.sizes

        # check for correct length of coordinates
        assert len(ds['Mesh2_node_x']) == len(ds['Mesh2_node_y'])
        assert len(ds['Mesh2_face_x']) == len(ds['Mesh2_face_y'])

        # check for correct shape of face nodes
        nMesh2_face = ds.sizes['nMesh2_face']
        assert ds['Mesh2_face_nodes'].shape == (nMesh2_face, 3)

    def test_add_fill_values(self):
        """Test _add_fill_values() implementation, output should be both be
        zero-indexed and padded values should be replaced with fill values."""

        # two cells with 2, 3 and 2 padded faces respectively
        verticesOnCell = np.array([[1, 2, 1, 1], [3, 4, 5, 3], [6, 7, 0, 0]],
                                  dtype=INT_DTYPE)

        # cell has 2, 3 and 2 nodes respectively
        nEdgesOnCell = np.array([2, 3, 2])

        # expected output of _add_fill_values()
        gold_output = np.array([[0, 1, self.fv, self.fv], [2, 3, 4, self.fv],
                                [5, 6, self.fv, self.fv]],
                               dtype=INT_DTYPE)

        # test data output
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)
        verticesOnCell = _replace_zeros(verticesOnCell)
        verticesOnCell = _to_zero_index(verticesOnCell)

        assert np.array_equal(verticesOnCell, gold_output)