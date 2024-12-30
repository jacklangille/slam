import cv2
import numpy as np

MX_KP_PER_CELL = 250
GRID_SZ = (20, 20)


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
