"""uxarray grid module."""

import numpy as np
from intervaltree import Interval, IntervalTree
# reader and writer imports

from .helpers import get_all_face_area_from_coords, parse_grid_type, _convert_node_xyz_to_lonlat_rad, _convert_node_lonlat_rad_to_xyz, _normalize_in_place, _within, _get_radius_of_latitude_rad, get_intersection_point_gcr_constlat, _close_face_nodes
from .constants import INT_DTYPE, INT_FILL_VALUE
from ._latlonbound_utilities import insert_pt_in_latlonbox, get_intersection_point_gcr_gcr


def _get_zonal_face_weights_at_constlat(self, candidate_faces_index_list, latitude_rad):
    # Then calculate the weight of each face
    # First calculate the perimeter this constant latitude circle
    candidate_faces_weight_list = [0.0] * len(candidate_faces_index_list)

    # An interval tree that stores the edges that are overlaped by the constant latitude
    overlap_interval_tree = IntervalTree()

    for i in range(0, len(candidate_faces_index_list)):
        face_index = candidate_faces_index_list[i]
        [face_lon_bound_min, face_lon_bound_max] = self.ds["Mesh2_latlon_bounds"].values[face_index][1]
        face_edges = np.zeros((len(self.ds["Mesh2_face_edges"].values[face_index]), 2), dtype=INT_DTYPE)
        face_edges = face_edges.astype(INT_DTYPE)
        for iter in range(0, len(self.ds["Mesh2_face_edges"].values[face_index])):
            edge_idx = self.ds["Mesh2_face_edges"].values[face_index][iter]
            if edge_idx == INT_FILL_VALUE:
                edge_nodes = [INT_FILL_VALUE, INT_FILL_VALUE]
            else:
                edge_nodes = self.ds['Mesh2_edge_nodes'].values[edge_idx]
            face_edges[iter] = edge_nodes
        # sort edge nodes in counter-clockwise order
        starting_two_nodes_index = [self.ds["Mesh2_face_nodes"][face_index][0],
                                    self.ds["Mesh2_face_nodes"][face_index][1]]
        face_edges[0] = starting_two_nodes_index
        for idx in range(1, len(face_edges)):
            if face_edges[idx][0] == face_edges[idx - 1][1]:
                continue
            else:
                # Swap the node index in this edge
                temp = face_edges[idx][0]
                face_edges[idx][0] = face_edges[idx][1]
                face_edges[idx][1] = temp

        pt_lon_min = 3 * np.pi
        pt_lon_max = -3 * np.pi

        intersections_pts_list_lonlat = []
        for j in range(0, len(face_edges)):
            edge = face_edges[j]

            # Skip the dummy edge
            if edge[0] == INT_FILL_VALUE or edge[1] == INT_FILL_VALUE:
                continue
            # Get the edge end points in 3D [x, y, z] coordinates
            n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                  self.ds["Mesh2_node_cart_y"].values[edge[0]],
                  self.ds["Mesh2_node_cart_z"].values[edge[0]]]
            n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                  self.ds["Mesh2_node_cart_y"].values[edge[1]],
                  self.ds["Mesh2_node_cart_z"].values[edge[1]]]
            n1_lonlat = _convert_node_xyz_to_lonlat_rad(n1)
            n2_lonlat = _convert_node_xyz_to_lonlat_rad(n2)
            intersections = get_intersection_point_gcr_constlat([n1, n2], latitude_rad)
            if intersections[0] == [-1, -1, -1] and intersections[1] == [-1, -1, -1]:
                # The constant latitude didn't cross this edge
                continue
            elif intersections[0] != [-1, -1, -1] and intersections[1] != [-1, -1, -1]:
                # The constant latitude goes across this edge ( 1 in and 1 out):
                intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))
            else:
                if intersections[0] != [-1, -1, -1]:
                    intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[0]))
                else:
                    intersections_pts_list_lonlat.append(_convert_node_xyz_to_lonlat_rad(intersections[1]))

        # If an edge of a face is overlapped by the constant lat, then it will have 4 non-unique intersection pts
        unique_intersection = np.unique(intersections_pts_list_lonlat, axis=0)
        if len(unique_intersection) == 2:
            # The normal convex case:
            [pt_lon_min, pt_lon_max] = np.sort(
                [unique_intersection[0][0], unique_intersection[1][0]])
        elif len(unique_intersection) != 0:
            # The concave cases
            raise ValueError(
                "UXarray doesn't support concave face [" + str(face_index) + "] with intersections points as [" + str(
                    len(unique_intersection)) + "] currently, please modify your grids accordingly")
        elif len(unique_intersection) == 0:
            # No intersections are found in this face
            raise ValueError("No intersections are found for face [" + str(
                face_index) + "], please make sure the buil_latlon_box generates the correct results")
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
        if np.abs(cur_face_mag_rad) > 2 * np.pi:
            print("At face: " + str(face_index) + "Problematic lat is " + str(
                latitude_rad) + " And the cur_face_mag_rad is " + str(cur_face_mag_rad))
            # assert(cur_face_mag_rad <= np.pi)

        # TODO：Marginal Case when two faces share an edge that overlaps with the constant latitude
        if n1_lonlat[1] == n2_lonlat[1] and (
                np.abs(np.abs(n1_lonlat[1] - n2_lonlat[1]) - np.abs(cur_face_mag_rad)) < 1e-12):
            # An edge overlaps with the constant latitude
            overlap_interval_tree.addi(n1_lonlat[0], n2_lonlat[0], face_index)

        candidate_faces_weight_list[i] = cur_face_mag_rad

    # Break the overlapped interval into the smallest fragments
    overlap_interval_tree.split_overlaps()
    overlaps_intervals = sorted(overlap_interval_tree.all_intervals)
    # Calculate the weight from each face by |intersection line length| / total perimeter
    candidate_faces_weight_list = np.array(candidate_faces_weight_list) / np.sum(candidate_faces_weight_list)
    return candidate_faces_weight_list