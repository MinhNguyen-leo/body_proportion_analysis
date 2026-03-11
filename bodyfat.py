import cv2
import numpy as np

def estimate_body_fat(torso_img):

    if torso_img is None or torso_img.size == 0:
        return None

    gray = cv2.cvtColor(torso_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels

    return edge_density

def body_fat_percentage(edge_density):

    # scale thực nghiệm (có thể chỉnh sau)
    fat = 30 - (edge_density * 200)

    # giới hạn trong khoảng hợp lý
    fat = max(6, min(fat, 30))

    return round(fat, 1)