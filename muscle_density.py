import cv2
import numpy as np

def muscle_density_score(region):

    if region is None or region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    variance = laplacian.var()

    return variance

