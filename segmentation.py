# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:50:47 2023

@author: scott
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#TODO: enlever le fond
#TODO: en opp colors
#TODO: elbow ???
#TODO: manque des labels
def kmeans_segmentation(img, nb_clusters):
    """
    Segmentation par kmeans
    """
    vectorized = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = nb_clusters
    attempts=10
    ret,label,centers=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    res = centers[label.flatten()]
    result_image = res.reshape((img.shape))
    
    plt.imshow(result_image)
    plt.show()
    return result_image, centers

    
def label_img(img, kmeans_segmented, centers):
    """
    Fonction qui retourne une image avec un label unique pour chaque pixel appartenant au même cluster.
    Retourne aussi une image vide qui va contenir tous les nouveaux labels combinés.
    """
    labeled = np.zeros(img.shape[:2])
    
    for i in range(len(centers)):
        indices = np.all(im == centers[i], axis=-1)
        labeled[indices] = i
    
    labeled = labeled.astype(np.uint8)
    
    stacked_labels = np.zeros_like(labeled)
    
    return labeled, stacked_labels

def label_connected_components(labeled, stacked_labels):
    """
    Pour chaque cluster de pixels, on trouve les composantes connexes pour leur affecter des labels
    On combine tous les nouveaux labels dans stacked_labels.
    """
    for region in np.unique(labeled):
        mask = np.where(labeled == region, True, False)
        _, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        max_label = np.max(labels)
        m = labels != 0
        labels[m] += max_label
        stacked_labels = np.maximum(stacked_labels, labels)
        plt.imshow(stacked_labels)
        plt.show()
        
    return stacked_labels


def region_graph(stacked_labels, img):
    """
    Création du graphe
    """
    graph = nx.Graph()
    
    #création des noeuds qui contiennent un label et le vecteur caractéristique
    for i in np.unique(stacked_labels):
        #Création d'un mask pour n'avoir que les pixels voulus
        indices = np.where(stacked_labels==i)
        object_pixels = img[indices]
        hist, edges = np.histogramdd(object_pixels, bins=(16,16,8), range=((0,256),(0,256),(0,256)))
        vec = hist.flatten()
        attributes = {'label' : i, 'vector' : vec}
        graph.add_node(i, **attributes)
        
    #Création des arêtes (dans l'image 4-connexe)
    neighbour_labels = []
    height, width = stacked_labels.shape
    for i in range(height-1):
        for j in range(width-1):
            px = stacked_labels[i,j]
            right_px = stacked_labels[i+1,j]
            bot_px = stacked_labels[i,j+1]
            if px != right_px:
                neighbour_labels.append(tuple(sorted([px,right_px])))
            if px != bot_px:
                neighbour_labels.append(tuple(sorted([px,bot_px])))

    neighbour_labels = list(set(neighbour_labels))
    for pair in neighbour_labels:
        graph.add_edge(pair[0],pair[1])
    
    return graph




if __name__ == "__main__":
    
    img = cv2.imread(r'..\papillons\peale2_pick\01-01.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    
    im, centers = kmeans_segmentation(img, 6)
    
    labeled, stacked_labels = label_img(img, im, centers)
    
    stacked_labels = label_connected_components(labeled, stacked_labels)

    graph = region_graph(stacked_labels, img)
    
    plt.figure(figsize=(50, 50))
    nx.draw(graph, with_labels=False, pos=nx.spring_layout(graph))
    plt.savefig("graph.svg")