import numpy as np
import cv2
from glob import glob
from ter import *
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsort_keygen

    


def train_kmeans(imgs, carac_vector, nb_clusters):
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
    ret, label, center = cv2.kmeans(x, nb_clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    df_centers = pd.DataFrame({'label':[], 'center':[]})
    for i in range(len(center)):
        df_centers.loc[len(df_centers.index)] = np.asarray([i, center[i]], dtype=object)
        
        
    df_labels_imgs = pd.DataFrame({'images':[], 'labels':[]})
    for i in range(len(n)):
        df_labels_imgs.loc[len(df_labels_imgs)] = [n[i], label[i, 0]]
    df_labels_imgs = df_labels_imgs.sort_values(by='images', key=natsort_keygen())
        
    with pd.ExcelWriter('classes.xlsx') as writer:
        df_labels_imgs.to_excel(writer, sheet_name='classification', index=False)
        df_centers.to_excel(writer, sheet_name='centers', index=False)



def predict_kmeans(img, path_xlsx, carac_vector):
    """
    Calcul du centre de cluster le plus proche du vecteur de l'image donnée en paramètre
    Retourne la distance et la classe de l'image
    """
    df = pd.read_excel(path_xlsx, sheet_name=1)
    df['center'] = df['center'].apply(lambda x: np.array(eval(x.replace(' ', ','))))
    
    vector = carac_vector(img)
    min_dist = np.inf
    img_class = 0
    for i in range(len(df)):
        dist = np.sqrt(np.sum(np.square(vector - df['center'][i])))
        if dist < min_dist:
            min_dist = dist
            img_class = i

    return img_class, min_dist



def elbow(imgs, carac_vector, lower_bound, upper_bound):
    """
    Sert à chercher le le nombre de clusters optimal pour entrainer kmeans
    """
    x = []
    for img in imgs:
        x.append(carac_vector(img))
        
    x = np.array(x).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    
    comp_list = []
    for k in range(lower_bound, upper_bound):
        print(k)
        compactness, _, _ = cv2.kmeans(x,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        comp_list.append(compactness)
        
    return comp_list


k_limit = 31 #nombre de clusters max pour la fonction elbow
imgs = glob(r'coil100\set1\*.png')

lower_bound, upper_bound = 80, 120
pl = elbow(imgs, caracteristic_vector_opponent_color_int16, lower_bound, upper_bound)
plt.plot([i for i in range(lower_bound, upper_bound)], pl)
#TODO: trouver elbow (regarder les dérivées secondes, gap statistic et silhouette coefficient)


#train_kmeans(imgs, caracteristic_vector_opponent_color_int16, 100)

path_xlsx = 'classes.xlsx'
path = r'coil100\set2\obj88__75.png'
#img_class, _ = predict_kmeans(path, path_xlsx, caracteristic_vector_opponent_color_int16)


