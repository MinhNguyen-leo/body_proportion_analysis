from utils import euclidean_distance
import cv2
import numpy as np

def estimate_real_waist_width(image, keypoints):

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]
    left_hip = person[11]
    right_hip = person[12]

    # waist height
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    mid_hip_y = (left_hip[1] + right_hip[1]) / 2
    waist_y = int(mid_shoulder_y + 0.55*(mid_hip_y - mid_shoulder_y))

    # 🔥 GIỚI HẠN VÙNG NGANG TORSO
    x_min = int(min(left_shoulder[0], right_shoulder[0]))
    x_max = int(max(left_shoulder[0], right_shoulder[0]))

    if waist_y >= image.shape[0]:
        return euclidean_distance(left_hip, right_hip)

    slice = image[waist_y:waist_y+3, x_min:x_max]

    gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)

    x_coords = np.where(edges > 0)[1]

    if len(x_coords) < 10:
        return euclidean_distance(left_hip, right_hip)

    left = np.min(x_coords)
    right = np.max(x_coords)

    waist_width = right - left

    return waist_width