"""The Intersection Over Union (IoU) for 3D oriented bounding boxes."""

import numpy as np
import scipy.spatial as sp
from scipy.stats import special_ortho_group

from . import box as Box

import torch
import torch.nn.functional as F
# from pytorch3d import _C
# from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles

_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1


class IoU(object):
    """General Intersection Over Union cost for Oriented 3D bounding boxes."""

    def __init__(self, box1, box2):
        self._box1 = box1
        self._box2 = box2
        self._intersection_points = []

    def iou(self):
        """Computes the exact IoU using Sutherland-Hodgman algorithm."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)
        if self._intersection_points:
            intersection_volume = sp.ConvexHull(
                self._intersection_points).volume
            box1_volume = self._box1.volume
            box2_volume = self._box2.volume
            union_volume = box1_volume + box2_volume - intersection_volume
            return intersection_volume / union_volume
        else:
            return 0.

    def iou_sampling(self, num_samples=10000):
        """Computes intersection over union by sampling points.
        Generate n samples inside each box and check if those samples are inside
        the other box. Each box has a different volume, therefore the number o
        samples in box1 is estimating a different volume than box2. To address
        this issue, we normalize the iou estimation based on the ratio of the
        volume of the two boxes.
        Args:
          num_samples: Number of generated samples in each box
        Returns:
          IoU Estimate (float)
        """
        p1 = [self._box1.sample() for _ in range(num_samples)]
        p2 = [self._box2.sample() for _ in range(num_samples)]
        box1_volume = self._box1.volume
        box2_volume = self._box2.volume
        box1_intersection_estimate = 0
        box2_intersection_estimate = 0
        for point in p1:
            if self._box2.inside(point):
                box1_intersection_estimate += 1
        for point in p2:
            if self._box1.inside(point):
                box2_intersection_estimate += 1
        # We are counting the volume of intersection twice.
        intersection_volume_estimate = (
            box1_volume * box1_intersection_estimate +
            box2_volume * box2_intersection_estimate) / 2.0
        union_volume_estimate = (box1_volume * num_samples + box2_volume *
                                 num_samples) - intersection_volume_estimate
        iou_estimate = intersection_volume_estimate / union_volume_estimate
        return iou_estimate

    def _compute_intersection_points(self, box_src, box_template):
        """Computes the intersection of two boxes."""
        # Transform the source box to be axis-aligned
        inv_transform = np.linalg.inv(box_src.transformation)
        box_src_axis_aligned = box_src.apply_transformation(inv_transform)
        template_in_src_coord = box_template.apply_transformation(
            inv_transform)
        for face in range(len(Box.FACES)):
            indices = Box.FACES[face, :]
            poly = [template_in_src_coord.vertices[indices[i], :]
                    for i in range(4)]
            clip = self.intersect_box_poly(box_src_axis_aligned, poly)
            for point in clip:
                # Transform the intersection point back to the world coordinate
                point_w = np.matmul(
                    box_src.rotation, point) + box_src.translation
                self._intersection_points.append(point_w)

        for point_id in range(Box.NUM_KEYPOINTS):
            v = template_in_src_coord.vertices[point_id, :]
            if box_src_axis_aligned.inside(v):
                point_w = np.matmul(box_src.rotation, v) + box_src.translation
                self._intersection_points.append(point_w)

    def intersect_box_poly(self, box, poly):
        """Clips the polygon against the faces of the axis-aligned box."""
        for axis in range(3):
            poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
            poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
        return poly

    def _clip_poly(self, poly, plane, normal, axis):
        """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.
        See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
        the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
        from "Real-Time Collision Detection", by Christer Ericson, page 370.
        Args:
          poly: List of 3D vertices defining the polygon.
          plane: The 3D vertices of the (2D) axis-aligned plane.
          normal: normal
          axis: A tuple defining a 2D axis.
        Returns:
          List of 3D vertices of the clipped polygon.
        """
        # The vertices of the clipped polygon are stored in the result list.
        result = []
        if len(poly) <= 1:
            return result

        # polygon is fully located on clipping plane
        poly_in_plane = True

        # Test all the edges in the polygon against the clipping plane.
        for i, current_poly_point in enumerate(poly):
            prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
            d1 = self._classify_point_to_plane(
                prev_poly_point, plane, normal, axis)
            d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                               axis)
            if d2 == _POINT_BEHIND_PLANE:
                poly_in_plane = False
                if d1 == _POINT_IN_FRONT_OF_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
            elif d2 == _POINT_IN_FRONT_OF_PLANE:
                poly_in_plane = False
                if d1 == _POINT_BEHIND_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)

                result.append(current_poly_point)
            else:
                if d1 != _POINT_ON_PLANE:
                    result.append(current_poly_point)

        if poly_in_plane:
            return poly
        else:
            return result

    def _intersect(self, plane, prev_point, current_point, axis):
        """Computes the intersection of a line with an axis-aligned plane.
        Args:
          plane: Formulated as two 3D points on the plane.
          prev_point: The point on the edge of the line.
          current_point: The other end of the line.
          axis: A tuple defining a 2D axis.
        Returns:
          A 3D point intersection of the poly edge with the plane.
        """
        alpha = (current_point[axis] - plane[axis]) / (
            current_point[axis] - prev_point[axis])
        # Compute the intersecting points using linear interpolation (lerp)
        intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
        return intersection_point

    def _inside(self, plane, point, axis):
        """Check whether a given point is on a 2D plane."""
        # Cross products to determine the side of the plane the point lie.
        x, y = axis
        u = plane[0] - point
        v = plane[1] - point

        a = u[x] * v[y]
        b = u[y] * v[x]
        return a >= b

    def _classify_point_to_plane(self, point, plane, normal, axis):
        """Classify position of a point w.r.t the given plane.
        See Real-Time Collision Detection, by Christer Ericson, page 364.
        Args:
          point: 3x1 vector indicating the point
          plane: 3x1 vector indicating a point on the plane
          normal: scalar (+1, or -1) indicating the normal to the vector
          axis: scalar (0, 1, or 2) indicating the xyz axis
        Returns:
          Side: which side of the plane the point is located.
        """
        signed_distance = normal * (point[axis] - plane[axis])
        if signed_distance > _PLANE_THICKNESS_EPSILON:
            return _POINT_IN_FRONT_OF_PLANE
        elif signed_distance < -_PLANE_THICKNESS_EPSILON:
            return _POINT_BEHIND_PLANE
        else:
            return _POINT_ON_PLANE

    @property
    def intersection_points(self):
        return self._intersection_points


if __name__ == '__main__':
    box1 = Box.Box.from_transformation(
        special_ortho_group.rvs(3), np.zeros((3,)), np.array([1, 1, 1]))
    box2 = Box.Box.from_transformation(
        np.eye(3), np.zeros((3,)), np.array([1, 1, 1]))
    
    print(IoU(box1, box2).iou())

def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that plane vertices are coplanar.
    Returns a bool tensor of size B, where True indicates a box is coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
    """
    Checks that the sides of the box have a non zero area.
    Returns a bool tensor of size B, where True indicates a box is nonzero.
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return (face_areas > eps).all(1).view(B)

def omni3d_box3d_overlap(
    boxes_dt: torch.Tensor, boxes_gt: torch.Tensor, 
    eps_coplanar: float = 1e-4, eps_nonzero: float = 1e-8
) -> torch.Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero 
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero  = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print('Warning: skipping {:d} non-coplanar boxes at eval.'.format(int(invalid_coplanar.float().sum())))
    
    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print('Warning: skipping {:d} zero volume boxes at eval.'.format(int(invalid_nonzero.float().sum())))

    return ious