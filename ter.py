import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from glob import glob
import pandas as pd

from skimage.io import imsave
from scipy.ndimage import binary_fill_holes
#installer pip openpyxl pour la création du .xls

np.set_printoptions(threshold=np.inf)


def caracteristic_vector_opponent_color_uint8(path):
    """
    Lit une image dans le chemin 'path', convertir l'espace de couleur comme dans le papier (opponent colors) et retourne le vecteur caractéristique
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #création du nouvel espace de couleurs
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = r-g
    by = 2*b-r-g
    wb = r+g+b

    #on réassemble chaque canal pour reformer l'image
    nimg = np.dstack((rg, by, wb))
    
    #on reshape l'image en une liste de listes de pixels à 3 valeurs (plus facile à gérer qu'une image 3d pour la génération de l'histogramme)
    data = nimg.reshape((-1,3))
    
    #histogramme
    hist, edges = np.histogramdd(data, bins=(16,16,8), range=((0,256),(0,256),(0,256)))
    
    #vecteur caracteritique en linearisant l'histogramme 3d
    vec = hist.flatten()
    return vec


def caracteristic_vector_opponent_color_int16(path):
    """
    Lit une image dans le chemin 'path', convertir l'espace de couleur comme dans le papier (opponent colors) et retourne le vecteur caractéristique
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int16)
    
    #création du nouvel espace de couleurs
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = r-g
    by = 2*b-r-g
    wb = r+g+b

    #on réassemble chaque canal pour reformer l'image
    nimg = np.dstack((rg, by, wb))
    
    #on reshape l'image en une liste de listes de pixels à 3 valeurs (plus facile à gérer qu'une image 3d pour la génération de l'histogramme)
    data = nimg.reshape((-1,3))
    
    #histogramme
    hist, edges = np.histogramdd(data, bins=(16,16,8), range=((-255,256),(-2*255,2*255+1),(0,3*255+1)))
    
    #vecteur caracteritique en linearisant l'histogramme 3d
    vec = hist.flatten()
    return vec



def caracteristic_vector_rgb(path):
    """
    Même chose en restant dans l'espace rgb (pour comparer)
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.reshape((-1,3))
    hist, edges = np.histogramdd(data, bins=(16,16,16), range=((0,256),(0,256),(0,256)))
    vec = hist.flatten()
    return vec

def caracteristic_vector_gray(path):
    """
    Même chose en passant en nuances de gris
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.flatten()
    hist, edges = np.histogram(data, bins=(16), range=(0,256))
    return hist


def comparison(to_classify, imgs, carac_vector):
    """
    Compare le pourcentage d'interesection de l'image 'to_classify' avec toutes les images dans le dossier 'imgs' avec la fonction 'carac_vector' (qui est une des trois fonctions définies juste avant)
    Renvoie une liste de listes qui contient le nom de l'image qu'on compare à 'to_classify' et le score. La liste est triée en fonction du score
    """
    color_vec1 = carac_vector(to_classify)#recuperation du vecteur caractéristique
    color_vec1 = color_vec1 / sum(color_vec1)#normalisation
    
    df = pd.DataFrame({'image': [], 'score':[]})
    for img in imgs:
        color_vec2 = carac_vector(img)
        color_vec2 = color_vec2 / sum(color_vec2)
        color_intersection = sum(np.minimum(color_vec1, color_vec2))
        df.loc[len(df.index)] = [img.split('\\')[-1], round(color_intersection, 3)]
        #df = df.sort_values(['score'], ascending=False)
    return df




def segmentation(path, new_path):
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
        imsave(new_path+'\\'+imgg.split('\\')[-1], img)
        





def write_xls(imgs, path, seg):
    
    #récupération des résultats avec l'espace modifié (uint8)
    results_opp8 = comparison(path, imgs, caracteristic_vector_opponent_color_uint8)
    
    #récupération des résultats avec l'espace modifié (int16)
    results_opp16 = comparison(path, imgs, caracteristic_vector_opponent_color_int16)
    
    #résultats en rgb (sans modifier l'espace)
    results_rgb = comparison(path, imgs, caracteristic_vector_rgb)
    
    #résultats en nuances de gris
    results_gray = comparison(path, imgs, caracteristic_vector_gray)
    
    #écriture des résultats dans un fichier xlsx (créé à la volée par le code)
    with pd.ExcelWriter('results.xlsx' if seg == False else 'seg_results.xlsx') as writer:
        results_opp8.to_excel(writer, sheet_name='opponent colors uint8', index=False)
        results_opp16.to_excel(writer, sheet_name='opponent colors int16', index=False)
        results_rgb.to_excel(writer, sheet_name='rgb colors', index=False)
        results_gray.to_excel(writer, sheet_name='gray', index=False)






#dossier vide pour sauvegarder toutes les images segmentées
new_path = r'C:\Users\scott\OneDrive\Bureau\New folder (23)\cropped_peal2'
#images à segmenter
path = glob(r'C:\Users\scott\OneDrive\Bureau\New folder (23)\peale2_pick\*.jpg')
#segmentation(path, new_path)


imgs = glob(r'peale2_pick\*.jpg')#dossier contenant toutes les images
path = r'peale2_pick\05-15.jpg' #image à comparer avec toutes celles de imgs
write_xls(imgs, path, False)
