
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import binary_fill_holes


baguette = glob(r'.\baguette*')
cannele = glob(r'.\can*')
chausson = glob(r'.\chausson*')
chocolatine = glob(r'.\painchoco*')
cookie = glob(r'.\cookie*')
croissant = glob(r'.\croissant*')
donut = glob(r'.\donut*')
eclair = glob(r'.\eclair*')
escargot = glob(r'.\painrais*')
galette = glob(r'.\galette*')
kouglof = glob(r'.\koug*')
macaron = glob(r'.\mac*')
madeleine = glob(r'.\mad*')
miche = glob(r'.\boule*')
millefeuille = glob(r'.\mille*')
palmier = glob(r'.\palmier*')
religieuse = glob(r'.\religieuse*')


def mask(glob_path):
    """
    Création des masques pour les images dans le dossier en paramètre
    """
    for p in glob_path:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU
        binary = np.bitwise_not(binary)
        #binary = binary_fill_holes(binary).astype(np.uint8)*255
        #kernel = np.ones((25,25))
        #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        to_save = np.dstack((binary, binary, binary))
        path = p.split('BDPATISBOUL')[0] + '\masks' +  p.split('BDPATISBOUL')[1]
        cv2.imwrite(path, to_save)
        plt.imshow(binary)
        plt.show()
        
        
def to_png(glob_path):
    """
    Conversion en png
    """
    for p in glob_path:
        img = cv2.imread(p)
        name = p.split('\\')[-1].split('.')[0]
        cv2.imwrite(f'./{name}.png', img)

#mask(baguette)
