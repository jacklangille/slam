import os
import cv2
import numpy as np

MX_KP_PER_CELL = 250
GRID_SZ = (20,20)

def filter_keypoints_and_descriptors(
    keypoints, descriptors, h, w, grid_size=GRID_SZ, max_per_cell=MX_KP_PER_CELL
):
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    selected_kps = []
    selected_des = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            cell_x_start = col * cell_w
            cell_y_start = row * cell_h
            cell_x_end = cell_x_start + cell_w
            cell_y_end = cell_y_start + cell_h
            cell_kps = [
                (kp, descriptors[i])
                for i, kp in enumerate(keypoints)
                if cell_x_start <= kp.pt[0] < cell_x_end
                and cell_y_start <= kp.pt[1] < cell_y_end
            ]
            cell_kps = sorted(cell_kps, key=lambda x: x[0].response, reverse=True)
            selected_kps.extend([kp for kp, _ in cell_kps[:max_per_cell]])
            selected_des.extend([des for _, des in cell_kps[:max_per_cell]])

    selected_des = np.array(selected_des) if selected_des else None

    return selected_kps, selected_des


def match_points(des0, des1, kp0, kp1, img0, img1, dist_threshold=50):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m in matches if m.distance < dist_threshold]
    match_frame = cv2.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return matches, match_frame


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
    """

    """


    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    depth_map = points_3D[:, :, 2]

    depth_map[depth_map <= 0] = np.nan
    depth_map_normalized = cv2.normalize(
        depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    depth_map_normalized = np.uint8(depth_map_normalized)
    return depth_map


def main_loop(
    cam0_images,
    cam1_images,
    cam0_matrix,
    cam1_matrix,
    cam0_distortion_coeffs,
    cam1_distortion_coeffs,
    cam0_to_body,
    cam1_to_body,
    baseline,
):
    orb = cv2.ORB_create()
    for img0_name, img1_name in zip(cam0_images, cam1_images):
        img0 = cv2.imread(f"./mav0/cam0/data/{img0_name}")
        img1 = cv2.imread(f"./mav0/cam1/data/{img1_name}")

        img0, img1, Q = rectify_imgs(
            img0,
            img1,
            cam0_matrix,
            cam1_matrix,
            cam0_distortion_coeffs,
            cam1_distortion_coeffs,
            cam0_to_body,
            cam1_to_body,
            baseline,
        )

        # Convert rectified images to grayscale
        gray_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

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

        disparity_colormap = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)

        cv2.imshow("Disparity Map", disparity_colormap)

        depth_map = compute_depth_map(disparity_map, Q)

        cv2.imshow("Depth Map", depth_map)

        cv2.imshow("Feature Extraction - Rectified Stereo Feed", combined_frame)

        # cv2.imshow("Feature Matching - Rectified Stereo Feed", match_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_cam0 = sorted(os.listdir("./mav0/cam0/data/"))
    images_cam1 = sorted(os.listdir("./mav0/cam1/data/"))

    # CAM0 PARAMS
    cam0_matrix = np.array(
        [[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]]
    )  # Intrinsic matrix

    cam0_distortion_coeffs = np.array(
        [-0.28340811, 0.07395907, 0.00019359, 0.0000176187114]
    )

    cam0_to_body = np.array(
        [
            [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
            [0, 0, 0, 1],
        ]
    )  # Rotation matrix

    # CAM1 PARAMS
    cam1_matrix = np.array([[457.587, 0, 379.999], [0, 456.134, 255.238], [0, 0, 1]])
    cam1_distortion_coeffs = np.array(
        [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
    )

    cam1_to_body = np.array(
        [
            [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
            [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
            [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    baseline = np.linalg.norm(cam1_to_body[:3, 3] - cam0_to_body[:3, 3])
    main_loop(
        images_cam0,
        images_cam1,
        cam0_matrix,
        cam1_matrix,
        cam0_distortion_coeffs,
        cam1_distortion_coeffs,
        cam0_to_body,
        cam1_to_body,
        baseline,
    )
