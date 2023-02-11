import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from glob import glob
import csv

#np.set_printoptions(threshold=np.inf)

def caracteristic_vector_opponent_color(path):
    """
    Lit une image dans le chemin 'path', convertir l'espace de couleur comme dans le papier (opponent colors) et retourne le vecteur caractéristique
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint16) #conversion en uint16, sinon on a certaines valeurs (ex: r+g+b) qui dépassent 256 (qui est la limite en uint8)
    
    #création du nouvel espace de couleurs
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = np.absolute(r-g)
    by = np.absolute(2*b-r-g)
    wb = np.absolute(r+g+b)
    
    #on réassemble chaque canal pour reformer l'image
    nimg = np.dstack((rg, by, wb))
    
    #on reshape l'image en une liste de listes de pixels à 3 valeurs (plus facile à gérer qu'une image 3d pour la génération de l'histogramme)
    data = nimg.reshape((-1,3))
    
    #histogramme
    hist, edges = np.histogramdd(data, bins=(16,16,8), range=((0,256),(0,2*255+1),(0,3*255+1)))
    
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
    
    color_list = []
    for img in imgs:
        color_vec2 = carac_vector(img)
        color_vec2 = color_vec2 / sum(color_vec2)
        color_intersection = sum(np.minimum(color_vec1, color_vec2))
        color_list.append([color_intersection, img.split('\\')[-1]])
    so = sorted(color_list, key=lambda x: x[0], reverse=True)
    so.append(['end', 'end'])
    return so
    


imgs = glob(r'peale2_pick\*.jpg')#dossier contenant toutes les images
path = r'peale2_pick\05-15.jpg' #image à comparer avec toutes celles de imgs

#récupération des résultats avec l'espace modifié
results_opp = comparison(path, imgs, caracteristic_vector_opponent_color)
results_opp.insert(0, ['opponent', 'colors'])

#résultats en rgb (sans modifier l'espace)
results_rgb = comparison(path, imgs, caracteristic_vector_rgb)
results_rgb.insert(0, ['rgb', 'rgb'])

#résultats en nuances de gris
results_gray = comparison(path, imgs, caracteristic_vector_gray)
results_gray.insert(0, ['gray', 'gray'])

results = results_opp + results_rgb + results_gray

#écriture des résultats dans un fichier csv (créé à la volée par le code)
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)




#Même chose avec des images segmentée (fond en noir) voir script segmentation.py
imgs = glob(r'cropped_peal2\*.jpg')
path = r'cropped_peal2\05-15.jpg'

results_opp = comparison(path, imgs, caracteristic_vector_opponent_color)
results_opp.insert(0, ['opponent', 'colors'])

results_rgb = comparison(path, imgs, caracteristic_vector_rgb)
results_rgb.insert(0, ['rgb', 'rgb'])

results_gray = comparison(path, imgs, caracteristic_vector_gray)
results_gray.insert(0, ['gray', 'gray'])

results = results_opp + results_rgb + results_gray

with open('results_segmentation.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)

#TODO: k-means





























def f():
    #img = cv2.imread(r'C:\Users\scott\OneDrive\Bureau\projet image m1\tab1.jpeg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def display(img):
        nbins = 5
        histo3d = cv2.calcHist([img[:,:,0],img[:,:,1],img[:,:,2]], [0,1,2], None, [nbins]*3, [0,256]*3).astype(int)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection="3d")
        space = np.array([*product(range(nbins), range(nbins), range(nbins))])
        volume = histo3d.astype(np.uint8)
        ax.scatter(space[:,0], space[:,1], space[:,2], c=space/nbins, s=volume*80)
    
    
    
    #transform_matrix = np.array([[r / (r + g + b), g / (r + g + b), b / (r + g + b)] for (r, g, b) in c])