
from scipy.fft import fft
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def extract_contour(mask):
    binary_mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("Contorno da lesão não encontrado")
    
    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2)
    
    return contour_points

def fourier_descriptors(contour_points):
    complex_contour = np.array([complex(x, y) for x, y in contour_points])
    fourier_descriptors = fft(complex_contour)
    return fourier_descriptors 

def compute_fourier_based_descriptor_shape_factor(fourier_descriptors):
    n = len(fourier_descriptors)
    magnitude = np.abs(fourier_descriptors)

    a1 = magnitude[1] 
    if a1 == 0:
        return 0.0

    numerador = 0.0
    denominador = 0.0
    for t in range(int(-n/2 + 1), int(n/2 + 1) ):
        k = t
        if k == 0:
            continue  
        if k < 0:
            e_t = magnitude[t + n] / a1
        else:
            e_t = magnitude[t]/a1

        numerador += e_t / abs(k)
        denominador += e_t

    fourier_factor = numerador / denominador if denominador > 0 else 0.0
    return fourier_factor


def measure_border_regularity(mask):
    contour_points = extract_contour(mask)
    descriptors = fourier_descriptors(contour_points)
    ff_score = compute_fourier_based_descriptor_shape_factor(descriptors)
    #aux.visualize_contour(mask)
    #aux.visualize_fourier_reconstruction(contour_points)
    #aux.display_fourier_spectrum(np.abs(contour_points))
    #plt.show()
    #print("FF Score = ", ff_score)
    return ff_score


#path = "../img_testes/teste_mask/Reg1.png" # 
path1 = "../img_testes/teste_mask/Reg2.png" # 
#path = "../img_testes/teste_mask/Reg3.png" #
#path = "../img_testes/teste_mask/Irr1.png" #
#path = "../img_testes/teste_mask/Irr2.png" #
#path = "../img_testes/teste_mask/Irr3.png" #
#path = "../img_testes/teste_mask/Irr4.png" #
path2 = "../img_testes/teste_mask/Irr5.png" #
mask_image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
mask_image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)


#print(measure_border_regularity(mask_image1))
#print(measure_border_regularity(mask_image2))
