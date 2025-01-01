import os
import cv2
import numpy as np
from config import *
from modules.feature import filter_keypoints_and_descriptors, match_points
from modules.rectify import rectify_imgs
from modules.disparity import compute_disparity_map, compute_depth_map


def main_loop(
    c0_images,
    c1_images,
    C0_MATRIX,
    C1_MATRIX,
    C0_DIST_COEFFS,
    C1_DIST_COEFFS,
    C0_TO_BODY,
    C1_TO_BODY,
    baseline,
):
    orb = cv2.ORB_create()
    for img0_name, img1_name in zip(c0_images, c1_images):
        img0 = cv2.imread(f"./mav0/cam0/data/{img0_name}")
        img1 = cv2.imread(f"./mav0/cam1/data/{img1_name}")

        img0, img1, Q = rectify_imgs(
            img0,
            img1,
            C0_MATRIX,
            C1_MATRIX,
            C0_DIST_COEFFS,
            C1_DIST_COEFFS,
            C0_TO_BODY,
            C1_TO_BODY,
            baseline,
        )

        # Convert rectified images to grayscale
        gray_img0, gray_img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
            img1, cv2.COLOR_BGR2GRAY
        )

        h, w = img0.shape[:2]
        kp0, des0 = orb.detectAndCompute(img0, None)
        kp1, des1 = orb.detectAndCompute(img1, None)

        kp0, des0 = filter_keypoints_and_descriptors(kp0, des0, h, w)
        kp1, des1 = filter_keypoints_and_descriptors(kp1, des1, h, w)

        img0_kp = cv2.drawKeypoints(img0, kp0, None, color=(0, 255, 0), flags=0)
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)

        matches, match_frame = match_points(
            des0, des1, kp0, kp1, img0, img1, dist_threshold=50
        )
        combined_frame = cv2.hconcat([img0_kp, img1_kp])

        disparity_map = compute_disparity_map(
            gray_img0, gray_img1, num_disparities=16, block_size=5
        )
        depth_map = compute_depth_map(disparity_map, Q)
        disparity_colormap = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)

        cv2.imshow("Disparity Map", disparity_colormap)
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Feature Extraction - Rectified Stereo Feed", combined_frame)
        cv2.imshow("Feature Matching - Rectified Stereo Feed", match_frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_cam0 = sorted(os.listdir("./mav0/cam0/data/"))
    images_cam1 = sorted(os.listdir("./mav0/cam1/data/"))

    baseline = np.linalg.norm(C1_TO_BODY[:3, 3] - C0_TO_BODY[:3, 3])

    main_loop(
        images_cam0,
        images_cam1,
        C0_MATRIX,
        C1_MATRIX,
        C0_DIST_COEFFS,
        C1_DIST_COEFFS,
        C0_TO_BODY,
        C1_TO_BODY,
        baseline,
    )
