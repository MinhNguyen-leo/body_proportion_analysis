import cv2
import numpy as np

from pose_detector import detect_pose, crop_chest, crop_left_arm, crop_right_arm, crop_shoulders
from muscle_density import muscle_density_score
from symmetry import calculate_symmetry
from bodyfat import estimate_body_fat, body_fat_percentage
from waist_estimator import estimate_real_waist_width
from utils import euclidean_distance


# ==============================
# LOAD IMAGE
# ==============================
image_path = "test4.webp"
image = cv2.imread(image_path)

# ve khung xuong de debug


# ==============================
# 1️⃣ DETECT POSE (ONLY ONCE)
# ==============================
keypoints, results = detect_pose(image_path)

if keypoints is None:
    print("❌ No person detected")
    exit()

person = keypoints[0]

left_shoulder = person[5]
right_shoulder = person[6]
left_hip = person[11]
right_hip = person[12]


# ==============================
# 2️⃣ ALIGN BODY (FIX LEAN POSE)
# ==============================
def get_body_axis(keypoints):
    person = keypoints[0]
    mid_shoulder = (person[5] + person[6]) / 2
    mid_hip = (person[11] + person[12]) / 2
    return mid_shoulder, mid_hip

def get_body_angle(mid_shoulder, mid_hip):
    dx = mid_hip[0] - mid_shoulder[0]
    dy = mid_hip[1] - mid_shoulder[1]
    return np.degrees(np.arctan2(dx, dy))

def align_body(image, keypoints):
    mid_shoulder, mid_hip = get_body_axis(keypoints)
    angle = get_body_angle(mid_shoulder, mid_hip)

    h, w = image.shape[:2]

    M = cv2.getRotationMatrix2D(
        (int(mid_shoulder[0]), int(mid_shoulder[1])),
        angle,
        1
    )

    return cv2.warpAffine(image, M, (w, h))

aligned_image = align_body(image, keypoints)


# ==============================
# 3️⃣ REAL WAIST WIDTH (FIX BELLY)
# ==============================
shoulder_width = euclidean_distance(left_shoulder, right_shoulder)

real_waist_width = estimate_real_waist_width(
    aligned_image,
    keypoints
)

v_ratio = shoulder_width / real_waist_width

print("Shoulder width:", shoulder_width)
print("Real Waist width:", real_waist_width)
print("V-Taper Ratio:", round(v_ratio,2))


# ==============================
# 4️⃣ SHAPE BODYFAT
# ==============================
def shape_bodyfat(shoulder_width, waist_width):

    ratio = waist_width / shoulder_width

    if ratio < 0.75:
        return 12
    elif ratio < 0.85:
        return 16
    elif ratio < 0.95:
        return 20
    elif ratio < 1.05:
        return 25
    else:
        return 30

shape_bf = shape_bodyfat(shoulder_width, real_waist_width)


# ==============================
# 5️⃣ TEXTURE BODYFAT
# ==============================
torso = aligned_image.copy()

edge_density = estimate_body_fat(torso)
texture_bf = body_fat_percentage(edge_density)


# ==============================
# 6️⃣ HYBRID BODYFAT
# ==============================
def hybrid_bodyfat(texture_bf, shape_bf, shoulder_width, waist_width):

    ratio = waist_width / shoulder_width

    # belly case
    if ratio > 0.9:
        return max(texture_bf, shape_bf)

    # lean case
    elif ratio < 0.75:
        return texture_bf

    else:
        return texture_bf * 0.5 + shape_bf * 0.5

fat_percent = hybrid_bodyfat(texture_bf, shape_bf, shoulder_width, real_waist_width)

print("Texture BF:", texture_bf)
print("Shape BF:", shape_bf)
print("Hybrid BF:", fat_percent)


# ==============================
# 7️⃣ MUSCLE DENSITY
# ==============================
chest = crop_chest(aligned_image, keypoints)
left_arm = crop_left_arm(aligned_image, keypoints)
right_arm = crop_right_arm(aligned_image, keypoints)

shoulder_region = crop_shoulders(aligned_image, keypoints)

h, w, _ = shoulder_region.shape
left_shoulder_img = shoulder_region[:, :w//2]
right_shoulder_img = shoulder_region[:, w//2:]

chest_density = muscle_density_score(chest)
left_arm_density = muscle_density_score(left_arm)
right_arm_density = muscle_density_score(right_arm)
left_shoulder_density = muscle_density_score(left_shoulder_img)
right_shoulder_density = muscle_density_score(right_shoulder_img)


# ==============================
# 8️⃣ SYMMETRY
# ==============================
arm_symmetry = calculate_symmetry(left_arm_density, right_arm_density)
shoulder_symmetry = calculate_symmetry(left_shoulder_density, right_shoulder_density)

upper_symmetry_score = (
    arm_symmetry * 0.5 +
    shoulder_symmetry * 0.5
)

print("Upper Symmetry:", upper_symmetry_score)


# ==============================
# 9️⃣ NORMALIZE ALL TO 100
# ==============================
def normalize_v_taper(v_ratio):
    return max(0, min((v_ratio - 1.0)/(1.8-1.0)*100,100))

def normalize_bodyfat(bf):
    return max(0, min((30 - bf)/(30-8)*100,100))

def normalize_density(density):
    return max(0, min((density - 5)/(60-5)*100,100))

arm_density = (left_arm_density + right_arm_density)/2
shoulder_density = (left_shoulder_density + right_shoulder_density)/2

muscle_score = (
    normalize_density(arm_density)*0.4 +
    normalize_density(shoulder_density)*0.3 +
    normalize_density(chest_density)*0.3
)


# ==============================
# 🔟 FINAL FITNESS SCORE
# ==============================
final_score = (
    normalize_v_taper(v_ratio)*0.4 +
    normalize_bodyfat(fat_percent)*0.2 +
    muscle_score*0.25 +
    upper_symmetry_score*0.15
)

print("🔥 BODY AESTHETIC SCORE:", round(final_score,2), "/100")