import cv2
import numpy as np

import math

def extract_contour(mask):
    binary_mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("Contorno da lesão não encontrado")
    
    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2)
    
    return contour_points, main_contour

def compute_compactness(full_contour):
    area = cv2.contourArea(full_contour)
    perimeter = cv2.arcLength(full_contour, True)

    if perimeter > 0:
        compactness_index = (4*math.pi * area) / (perimeter**2)
    else: compactness_index = 0
    return compactness_index


def calculate_compactness(mask):
    main , _ = extract_contour(mask)
    compactness = compute_compactness(main)
    return compactness
