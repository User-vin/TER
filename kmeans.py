import numpy as np
import cv2
from glob import glob
from ter import *
import pandas as pd
from natsort import natsort_keygen
from kneed import KneeLocator
from tqdm import tqdm
from natsort import natsorted


def train_kmeans(imgs, imgs_seg, carac_vector, nb_clusters, segmented):
    """
    Clustering des vecteurs caracteristiques, création des .xlsx et écriture des résultats
    """
    x = []
    n = []
    for img, img_seg in zip(natsorted(imgs), natsorted(imgs_seg)):
        x.append(carac_vector(img, img_seg, segmented))
        n.append(img.split('\\')[-1].split('.')[0])
        
    x = np.array(x).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    _, label, center = cv2.kmeans(x, nb_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    #Création du dataframe des classe associées aux centres
    df_centers = pd.DataFrame({'label':[], 'center':[]})
    for i in range(len(center)):
        df_centers.loc[len(df_centers.index)] = np.asarray([i, center[i]], dtype=object)
        
    #Création du dataframe des cluster associés à chaque image
    df_labels_imgs = pd.DataFrame({'images':[], 'labels':[]})
    for i in range(len(n)):
        df_labels_imgs.loc[len(df_labels_imgs)] = [n[i], label[i, 0]]
    df_labels_imgs = df_labels_imgs.sort_values(by='images', key=natsort_keygen())
    
    #écriture des deux fichiers excel pour les deux dataframes
    with pd.ExcelWriter('classes.xlsx') as writer:
        df_labels_imgs.to_excel(writer, sheet_name='classification', index=False)
        df_centers.to_excel(writer, sheet_name='centers', index=False)



def predict_kmeans(img, img_seg, path_xlsx, carac_vector):
    """
    Calcul du centre de cluster le plus proche du vecteur de l'image donnée en paramètre
    Retourne la distance et la classe de l'image
    """
    df = pd.read_excel(path_xlsx, sheet_name=1)
    df['center'] = df['center'].apply(lambda x: np.array(eval(x.replace(' ', ','))))

    vector = carac_vector(img, img_seg, segmented)
    min_dist = np.inf
    img_class = 0
    for i in range(len(df)):
        dist = np.sqrt(np.sum(np.square(vector - df['center'][i])))
        if dist < min_dist:
            min_dist = dist
            img_class = i

    return img_class, min_dist



def elbow(imgs, imgs_seg, carac_vector, lower_bound, upper_bound, segmented):
    """
    Sert à chercher le nombre de clusters optimal pour entrainer kmeans
    lower_bound et upperbound : interval pour le nombre de clusters à tester
    """
    x = []
    for img, img_seg in zip(natsorted(imgs), natsorted(imgs_seg)):
        x.append(carac_vector(img, img_seg, segmented))
        
    x = np.array(x).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    
    k_values = np.arange(lower_bound, upper_bound)
    comp_list = []
    for k in tqdm(k_values):
        compactness, _, _ = cv2.kmeans(x, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        comp_list.append(compactness)
        
    kneedle = KneeLocator(k_values, comp_list, S=2.0, curve="convex", direction="decreasing")
    kneedle.plot_knee_normalized()
    
    return kneedle.elbow




if __name__ == "__main__":
    
    imgs = glob(r'.\imgs\*.png')
    imgs_seg = glob(r'.\masks\*.png')
    
    segmented = True
    
    lower_bound, upper_bound = 1, 55
    elb = elbow(imgs, imgs_seg, caracteristic_vector_rgb, lower_bound, upper_bound, segmented)
    
    print(f'elbow : {elb}')
    
    train_kmeans(imgs, imgs_seg, caracteristic_vector_rgb, elb, segmented)



