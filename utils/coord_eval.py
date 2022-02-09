import numpy as np
import torch

import dsacstar
import cv2
from typing import Tuple


def pick_valid_points(coord_input, nodata_value, boolean=False):
    """
    Pick valid 3d points from provided ground-truth labels.
    @param   coord_input   [B, C, N] or [C, N] tensor for 3D labels such as scene coordinates or depth.
    @param   nodata_value  Scalar to indicate NODATA element of ground truth 3D labels.
    @param   boolean       Return boolean variable or explicit index.
    @return  val_points    [B, N] or [N, ] Boolean tensor or valid points index.
    """
    batch_mode = True
    if len(coord_input.shape) == 2:
        # coord_input shape is [C, N], let's make it compatible
        batch_mode = False
        coord_input = coord_input.unsqueeze(0)  # [B, C, N], with B = 1

    val_points = torch.sum(coord_input == nodata_value, dim=1) == 0  # [B, N]
    val_points = val_points.to(coord_input.device)
    if not batch_mode:
        val_points = val_points.squeeze(0)  # [N, ]
    if boolean:
        pass
    else:
        val_points = torch.nonzero(val_points, as_tuple=True)  # a tuple for rows and columns indices
    return val_points
	
def get_pose_err(gt_pose: np.ndarray, est_pose: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation and rotation error between two 4x4 transformation matrices.
    """
    transl_err = np.linalg.norm(gt_pose[0:3, 3] - est_pose[0:3, 3])

    rot_err = est_pose[0:3, 0:3].T.dot(gt_pose[0:3, 0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1, 3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
    rot_err = rot_err[0]
    return transl_err, rot_err


def scene_coords_eval(scene_coords, gt_coords, gt_pose, nodata_value, focal_length, image_h, image_w,
                      hypotheses=64, threshold=10, inlier_alpha=100, max_pixel_error=100, output_subsample=8) \
        -> Tuple[float, float, list, list]:
    """
    Evaluate predicted scene coordinates.
    DSAC* PnP solver is adopted. Code reference: https://github.com/vislearn/dsacstar.
    @param scene_coords             [1, 3, H, W], predicted scene coordinates.
    @param gt_coords                [1, 3, H, W], ground-truth scene coordinates.
    @param gt_pose                  [1, 4, 4] cam-to-world
    @param nodata_value             Nodata value.
    @param focal_length             Camera focal length
    @param image_h                  Image height
    @param image_w                  Image width
    @param hypotheses               DSAC* PnP solver parameter.
    @param threshold                DSAC* PnP solver parameter.
    @param inlier_alpha             DSAC* PnP solver parameter.
    @param max_pixel_error          DSAC* PnP solver parameter.
    @param output_subsample         DSAC* PnP solver parameter.
    @return: t_err, r_err, est_xyz, coords_error for has-data pixels
    """
    gt_pose = gt_pose[0].cpu()

    """metrics on camera pose"""
    # compute 6D camera pose
    out_pose = torch.zeros((4, 4))
    scene_coords = scene_coords.cpu()
    dsacstar.forward_rgb(
        scene_coords,
        out_pose,
        hypotheses,
        threshold,
        focal_length,
        float(image_w / 2),  # principal point assumed in image center
        float(image_h / 2),
        inlier_alpha,
        max_pixel_error,
        output_subsample)

    # calculate pose error
    t_err, r_err = get_pose_err(gt_pose.numpy(), out_pose.numpy())

    # estimated XYZ position
    est_xyz = out_pose[0:3, 3].tolist()

    """metrics on regression error"""
    scene_coords = scene_coords.view(scene_coords.size(0), 3, -1)  # [1, 3, H*W]
    gt_coords = gt_coords.view(gt_coords.size(0), 3, -1)  # [1, 3, H*W]
    mask_gt_coords_valdata = pick_valid_points(gt_coords, nodata_value, boolean=True)  # [1, H*W]

    coords_error = torch.norm(gt_coords - scene_coords, dim=1, p=2)  # [1, H*W]
    coords_error_valdata = coords_error[mask_gt_coords_valdata].tolist()  # [X]

    print("\nRotation Error: %.2f deg, Translation Error: %.1f m, Mean coord prediction error: %.1f m" % (
        r_err, t_err, np.mean(coords_error_valdata)))
    return t_err, r_err, est_xyz, coords_error_valdata