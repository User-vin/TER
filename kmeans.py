# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:47:05 2023

@author: scott
"""

import numpy as np
import cv2
from glob import glob
from ter import *
import pandas as pd

    


def train_kmeans(imgs, carac_vector):
    """
    Entrainement kmeans avec les vecteurs caracteristiques, création des .xlsx et écriture des résultats
    """
    x = []
    n = []
    for img in imgs:
        x.append(carac_vector(img))
        n.append(img.split('\\')[-1].split('.')[0])
        
    x = np.array(x).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    ret, label, center = cv2.kmeans(x,100,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    df_centers = pd.DataFrame({'label':[], 'center':[]})
    for i in range(len(center)):
        df_centers.loc[len(df_centers.index)] = np.asarray([i, center[i]], dtype=object)
        
        
    df_labels_imgs = pd.DataFrame({'images':[], 'labels':[]})
    for i in range(len(n)):
        df_labels_imgs.loc[len(df_labels_imgs)] = [n[i], label[i, 0]]
        
    with pd.ExcelWriter('classes.xlsx') as writer:
        df_labels_imgs.to_excel(writer, sheet_name='classification', index=False)
        df_centers.to_excel(writer, sheet_name='centers', index=False)
    
    return df_labels_imgs, df_centers



def predict_kmeans(img, df, carac_vector):
    """
    Calcul du centre de cluster le plus proche du vecteur de l'image donnée en paramètre
    Retourne la distance et la classe de l'image
    """
    vector = carac_vector(img)
    min_dist = np.inf
    img_class = 0
    for i in range(len(df)):
        dist = np.sqrt(np.sum(np.square(vector - df['center'][i])))
        if dist < min_dist:
            min_dist = dist
            img_class = i

    return img_class, min_dist


imgs = glob(r'coil-100\coil-100\*.png')
df_labels_imgs, df_centers = train_kmeans(imgs, caracteristic_vector_opponent_color_uint8)

df_centers = pd.read_excel('classes.xlsx', sheet_name=1)
df_centers['center'] = df_centers['center'].apply(lambda x: np.array(eval(x.replace(' ', ','))))

path = r'divided\obj88__75.png'
img_class, min_dist = predict_kmeans(path, df_centers, caracteristic_vector_opponent_color_uint8)
print(img_class, min_dist)


