# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:12:37 2023

@author: scott
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from glob import glob
from skimage.io import imsave


#dossier vide pour sauvegarder toutes les images segmentées
cropped_path = r'C:\Users\scott\OneDrive\Bureau\New folder (23)\cropped_peal2'

#images à segmenter
path = glob(r'C:\Users\scott\OneDrive\Bureau\New folder (23)\peale2_pick\*.jpg')


for imgg in path:
    img = cv2.imread(imgg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    binary = np.where(binary == 255, 0, 255)
    kernel = np.ones((9,9), np.uint8)
    
   
    binary = binary_fill_holes(binary).astype(np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    closing = binary_fill_holes(closing).astype(np.uint8)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    img[closing == 0] = [0, 0, 0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imsave(cropped_path+'\\'+imgg.split('\\')[-1], img)
        
