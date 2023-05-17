import cv2
import numpy as np
from glob import glob
import pandas as pd

from natsort import natsort_keygen, natsorted
from tqdm import tqdm

def caracteristic_vector_opponent_color_uint8(path, path_seg, segmented):
    """
    Lecture d'une image dans le chemin 'path', changement d'espace de couleurs en rg,by,wb et retourne le vecteur caractéristique
    Si segmented = True, on utilise seulement les pixels de l'objet grâce au masque
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
    
    data = nimg.reshape((-1,3))
    
    #lecture du masque
    mask = cv2.imread(path_seg)[:,:,0]
    indices = np.where(mask)
    object_pixels = nimg[indices]
    
    #histogramme
    hist, edges = np.histogramdd(object_pixels if segmented == True else data, bins=(16,16,8), range=((0,256),(0,256),(0,256)))
    
    #vecteur caracteritique en linearisant l'histogramme 3d
    vec = hist.flatten()
    return vec


def caracteristic_vector_rgb(path, path_seg, segmented):
    """
    Même chose en restant dans l'espace rgb
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    data = img.reshape((-1,3))
    
    mask = cv2.imread(path_seg)[:,:,0]
    indices = np.where(mask)
    object_pixels = img[indices]
    
    hist, edges = np.histogramdd(object_pixels if segmented == True else data, bins=(16,16,16), range=((0,256),(0,256),(0,256)))
    vec = hist.flatten()
    return vec

def caracteristic_vector_gray(path, path_seg, segmented):
    """
    Même chose en passant en nuances de gris
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.imread(path_seg)[:,:,0]
    indices = np.where(mask)
    object_pixels = img[indices]
    
    hist, edges = np.histogram(object_pixels if segmented == True else img.flatten(), bins=(255), range=(0,256))
    return hist


def comparison(to_classify, to_classify_seg, imgs, imgs_seg, carac_vector, col_name, segmented):
    """
    Compare le pourcentage d'interesection de l'image 'to_classify' avec toutes les images dans le dossier 'imgs' avec la fonction 'carac_vector' (qui est une des trois fonctions définies juste avant)
    Renvoie une liste de listes qui contient le nom de l'image qu'on compare à 'to_classify' et le score. La liste est triée en fonction du score
    """
    color_vec1 = carac_vector(to_classify, to_classify_seg, segmented)#vecteur caractéristique
    color_vec1 = color_vec1 / max(color_vec1)#normalisation
    
    
    name = to_classify.split('\\')[-1]
    
    df = pd.DataFrame({name: [], col_name:[]})
    for img, img_seg in zip(imgs, imgs_seg):#parcours des images
        color_vec2 = carac_vector(img, img_seg, segmented)#vec caractéristique
        color_vec2 = color_vec2 / max(color_vec2)#normalisation
        color_intersection = sum(np.minimum(color_vec1, color_vec2))/sum(np.maximum(color_vec1, color_vec2))#intersection entre les deux histogrammes
        df.loc[len(df.index)] = [img.split('\\')[-1], round(color_intersection, 3)]
    df = df.sort_values(by=name, key=natsort_keygen())#tri
    return df

def write_xls(imgs, imgs_seg, path, path_seg, xlsx, segmented):
    """
    Renvoie le dataframe contenant les résultats de comparaison de toutes les images avec toutes les images dans les 3 espaces de couleurs
    """
    
    #récupération des résultats avec l'espace modifié (uint8)
    results_opp8 = comparison(path, path_seg, imgs, imgs_seg, caracteristic_vector_opponent_color_uint8, 'opp_colors_uint8', segmented)
    
    #résultats en rgb
    results_rgb = comparison(path, path_seg, imgs, imgs_seg, caracteristic_vector_rgb, 'rgb', segmented)
    
    #résultats en nuances de gris
    results_gray = comparison(path, path_seg, imgs, imgs_seg, caracteristic_vector_gray, 'gray', segmented)
    
    
    results_opp8['rgb'] = results_rgb['rgb']
    results_opp8['gray'] = results_gray['gray']
    
    return results_opp8



if __name__ == "__main__":

    imgs_set1 = natsorted(glob(r'C:\Users\scott\OneDrive\Bureau\peale2_pick\*.jpg'))#dossier des images
    imgs_set1_seg = natsorted(glob(r'C:\Users\scott\OneDrive\Bureau\masks_pick\*.jpg'))#dossier des masques des images
    
    xlsx = 'results.xlsx'#chemin sauvegarde du fichier
    
    segmented = False#False si on veut faire les comparaisons sur les images originales, sans segmentation
    
    df = pd.DataFrame()
    
    for img, img_seg in tqdm(zip(imgs_set1, imgs_set1_seg)):
        df = pd.concat([df,  write_xls(imgs_set1, imgs_set1_seg, img, img_seg, xlsx, segmented)], axis=1)

    #écriture du fichier excel
    with pd.ExcelWriter(xlsx) as writer:
        df.to_excel(writer, index=False)
