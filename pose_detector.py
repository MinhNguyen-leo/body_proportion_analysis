from ultralytics import YOLO
import cv2
import numpy as np

# Load pose model (chỉ load 1 lần)
model = YOLO("yolov8n-pose.pt")

def detect_pose(image_path):

    results = model(image_path)

    keypoints = results[0].keypoints.xy.cpu().numpy()

    return keypoints, results

# Ham lay toa do vai hong (shoulder va hip) tu keypoints
def get_body_points(keypoints):

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]

    left_hip = person[11]
    right_hip = person[12]

    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip
    }
    
    from utils import euclidean_distance

from utils import euclidean_distance
def calculate_visual_waist(image, keypoints):

    torso = crop_torso(image, keypoints)

    if torso is None or torso.size == 0:
        return None

    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)

    # Blur nhẹ để ổn định edge
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Lấy contour lớn nhất (thường là body)
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    return w

def calculate_v_taper(image, keypoints):

    if keypoints is None or len(keypoints) == 0:
        print("No person detected")
        return None, None, None

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]
    left_hip = person[11]
    right_hip = person[12]

    if (left_hip == [0,0]).all() or (right_hip == [0,0]).all():
        print("Hip not detected")
        return None, None, None

    # 1️⃣ Shoulder width
    shoulder_width = euclidean_distance(left_shoulder, right_shoulder)

    # 2️⃣ Skeleton waist
    skeleton_waist = euclidean_distance(left_hip, right_hip)

    # 3️⃣ Visual waist
    visual_waist = calculate_visual_waist(image, keypoints)

    if visual_waist is None:
        effective_waist = skeleton_waist
    else:
        # 🔥 Lấy cái lớn hơn để tránh cheat
        effective_waist = max(skeleton_waist, visual_waist)

    if effective_waist < 10:
        print("Waist width quá nhỏ")
        return None, None, None

    v_ratio = shoulder_width / effective_waist

    return v_ratio, shoulder_width, effective_waist

def crop_torso(image, keypoints):

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]
    left_hip = person[11]
    right_hip = person[12]

    x_min = int(min(left_shoulder[0], right_shoulder[0]))
    x_max = int(max(left_shoulder[0], right_shoulder[0]))

    y_min = int(min(left_shoulder[1], right_shoulder[1]))
    y_max = int(max(left_hip[1], right_hip[1]))

    torso = image[y_min:y_max, x_min:x_max]

    return torso

# crop tung bo phan tu anh de tinh mat do co
def crop_chest(image, keypoints):

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]
    left_hip = person[11]
    right_hip = person[12]

    x_min = int(min(left_shoulder[0], right_shoulder[0]))
    x_max = int(max(left_shoulder[0], right_shoulder[0]))

    y_min = int(min(left_shoulder[1], right_shoulder[1]))
    y_max = int((left_hip[1] + right_hip[1]) / 2)

    return image[y_min:y_max, x_min:x_max]

def crop_left_arm(image, keypoints):

    person = keypoints[0]

    shoulder = person[5]
    elbow = person[7]

    x_min = int(min(shoulder[0], elbow[0]))
    x_max = int(max(shoulder[0], elbow[0]))

    y_min = int(min(shoulder[1], elbow[1]))
    y_max = int(max(shoulder[1], elbow[1]))

    return image[y_min:y_max, x_min:x_max]

def crop_right_arm(image, keypoints):

    person = keypoints[0]

    shoulder = person[6]
    elbow = person[8]

    x_min = int(min(shoulder[0], elbow[0]))
    x_max = int(max(shoulder[0], elbow[0]))

    y_min = int(min(shoulder[1], elbow[1]))
    y_max = int(max(shoulder[1], elbow[1]))

    return image[y_min:y_max, x_min:x_max]

def crop_shoulders(image, keypoints):

    person = keypoints[0]

    left_shoulder = person[5]
    right_shoulder = person[6]

    x_min = int(min(left_shoulder[0], right_shoulder[0]))
    x_max = int(max(left_shoulder[0], right_shoulder[0]))

    y_min = int(min(left_shoulder[1], right_shoulder[1])) - 40
    y_max = int(max(left_shoulder[1], right_shoulder[1])) + 40

    return image[y_min:y_max, x_min:x_max]