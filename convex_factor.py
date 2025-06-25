import cv2
import numpy as np

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



def calculate_convexity(mask):
    _, main_contour = extract_contour(mask)

    area_L = cv2.contourArea(main_contour)
    perimeter_L = cv2.arcLength(main_contour, True)

    if area_L == 0 or perimeter_L == 0:
        return 0
    hull = cv2.convexHull(main_contour)
    area_HS = cv2.contourArea(hull)
    convexity = area_L / area_HS

    return convexity

