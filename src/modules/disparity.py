import cv2
import numpy as np


def compute_disparity_map(rect_img0, rect_img1, num_disparities=48, block_size=5):
    """
    Produces a disparity map between two stereo rectified images.
    """

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,  # Range of disparity
        blockSize=block_size,
        P1=8 * 3 * block_size**2,  # Penalty on disparity changes
        P2=32 * 3 * block_size**2,  # Stronger penalty for larger changes
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = stereo.compute(rect_img0, rect_img1)

    disparity_map = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity_map = np.uint8(disparity_map)

    return disparity_map


def compute_depth_map(disparity_map, Q):
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    depth_map = points_3D[:, :, 2]

    depth_map[depth_map <= 0] = np.nan
    depth_map_normalized = cv2.normalize(
        depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    depth_map_normalized = np.uint8(depth_map_normalized)
    return depth_map
