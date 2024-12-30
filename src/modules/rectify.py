import cv2
import numpy as np


def rectify_imgs(
    img0,
    img1,
    cam0_matrix,
    cam1_matrix,
    cam0_distortion_coeffs,
    cam1_distortion_coeffs,
    cam0_to_body,
    cam1_to_body,
    baseline,
):
    R = np.linalg.inv(cam1_to_body[:3, :3]) @ cam0_to_body[:3, :3]
    T = cam1_to_body[:3, 3] - cam0_to_body[:3, 3]

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cam0_matrix,
        cam0_distortion_coeffs,
        cam1_matrix,
        cam1_distortion_coeffs,
        (img0.shape[1], img0.shape[0]),
        R,
        T,
        alpha=0,
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cam0_matrix,
        cam0_distortion_coeffs,
        R1,
        P1,
        (img0.shape[1], img0.shape[0]),
        cv2.CV_32FC1,
    )

    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cam1_matrix,
        cam1_distortion_coeffs,
        R2,
        P2,
        (img1.shape[1], img1.shape[0]),
        cv2.CV_32FC1,
    )

    rect_img0 = cv2.remap(img0, map1_x, map1_y, cv2.INTER_LINEAR)
    rect_img1 = cv2.remap(img1, map2_x, map2_y, cv2.INTER_LINEAR)

    return rect_img0, rect_img1, Q
