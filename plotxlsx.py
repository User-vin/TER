import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ter import *
from tqdm import tqdm
    
    
def intersection_plot(path):
    """
    Lecture d'un fichier excel contenant les résultats des comparaisons du script ter.py
    Affichage de l'intersection d'histogrammes avec l'image name avec les autres dans le fichier
    """
    data = pd.read_excel(path, engine='openpyxl')
    
    x_values = data.iloc[:, 0].str.replace('.png', '')
    y1_values = data.iloc[:, 1]
    y2_values = data.iloc[:, 2]
    y3_values = data.iloc[:, 3]
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(x_values, y1_values, label='rg,by,wb')
    plt.plot(x_values, y2_values, label='rgb')
    plt.plot(x_values, y3_values, label='gray')
    
    plt.xlabel('images')
    plt.ylabel('niveau d\'intersection')
    plt.title('Comparaisons')
    plt.legend()
    
    plt.ylim(0, 1)
    plt.xticks(rotation=90)

    plt.savefig('Comparaison_espaces.png', bbox_inches='tight')
    plt.show()


def labels_kmeans(path):
    """
    Lecture d'un fichier excel contenant les résultats des comparaisons du script ter.py
    Création d'un graphique pour visualiser la répartition des classes
    """
    df = pd.read_excel(path)
    x = df['images'].values
    y = df['labels'].values
    
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    ax.stem(np.arange(x.size), y, markerfmt='')
    ax.set_xlabel('images')
    ax.set_ylabel('classes')
    ax.set_title('Résultats kmeans')
    plt.xticks(range(len(x)), x)
    ax.tick_params(axis='x', labelrotation=90, labelsize=5)
    plt.savefig('multi')
    plt.show()
    
    
    
def precision_recall_f1(path, nb_per_class):
    """
    Lecture d'un fichier excel contenant les résultats des comparaisons du script ter.py
    Calcul des courbes de précision, rappel et f1-score pour les nb_per_class-1 plus proches images, pour chaque image du dataset
    """
    df = pd.read_excel(path)
    num_columns = len(df.columns)
    
    #création d'un dataframe toutes les 4 colonnes dans le dataframe de base
    #correspond à une colonne avec toutes les images comparées à l'image courante et 3 colonnes pour les 3 espaces
    for i in range(0, num_columns, 4):
        sub_df = df.iloc[:, i:i+4]
        new_dfs = []
        
        #Tri le dataframe pour chaque espace de couleurs
        for k in range(0, num_columns, 4):
            sub_df = df.iloc[:, k:k+4]
            for j in range(3):
                new_cols = [sub_df.columns[0], sub_df.columns[j+1]]
                new_df = sub_df[new_cols]
                new_df = new_df.sort_values(by=new_cols[1], ascending=False)
                new_dfs.append(new_df)
                
              
    #arrays pour les rappels et précisions pour chaque espace
    sum_yr_gray = np.zeros(nb_per_class-1)
    sum_yr_opp = np.zeros(nb_per_class-1)
    sum_yr_rgb = np.zeros(nb_per_class-1)
    
    sum_yp_gray = np.zeros(nb_per_class-1)
    sum_yp_opp = np.zeros(nb_per_class-1)
    sum_yp_rgb = np.zeros(nb_per_class-1)
    
    #précision et rappel pour chaque sous-dataframe
    for i in tqdm(range(len(new_dfs))):
        plot_df = new_dfs[i].iloc[1:nb_per_class]#les nb_per_class premier sans compter le premier élément
        
        s = 0
        yp = np.zeros(nb_per_class-1)#calcul de la précision pour les nb_per_class permiers éléments
        name = plot_df.columns[0].split('.')[0]
        
        #précision
        for j in range(nb_per_class-1):
            classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('__')[0]#on récupère la classe dans le nom de l'image
            s += 1 if classe == name.split('__')[0] else 0#on ajoute 1 si les deux images sont de la même classe, 0 sinon
            yp[j] = s/(j+1)#on divise par le nombre d'éléments retournés
            
        s = 0
        #rappel
        yr = np.zeros(nb_per_class-1)
        for j in range(nb_per_class-1):
            classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('__')[0]
            s += 1 if classe == name.split('__')[0] else 0
            yr[j] = s/(nb_per_class-1)#division par le nombre d'éléments dans la classe
            
            
        #Ajout des rappel et précision aux arrays des espaces de couleurs correspondants pour avoir la moyenne global pour chaque espace de couleurs
        color_space = plot_df.columns[1].split('.')[0]

        if color_space == 'gray':
            sum_yr_gray += yr
            sum_yp_gray += yp
            
        elif color_space == 'opp_colors_uint8':
            sum_yr_opp += yr
            sum_yp_opp += yp
        
        elif color_space == 'rgb':
            sum_yr_rgb += yr
            sum_yp_rgb += yp
        
            
    #plot du rappel, de la précision et f1-score pour l'espace color_space    
    def plot_average_prec_rec(yr, yp, path, color_space):
        
        f1_score = 2*((yr*yp)/(yr+yp))#calcul de la courbe f1
        
        x = [x for x in range(1, nb_per_class)]
        
        plt.plot(x, f1_score, label='f1-score')
        plt.plot(x, yp, label='precision')
        plt.plot(x, yr, label='recall')
        
        
        #affichage des valeurs de chaque point dans les courbes
        for i, j in zip(x, f1_score):
            plt.text(i, j, str(round(j, 2)))
        for i, j in zip(x, yp):
            plt.text(i, j, str(round(j, 2)))
        for i, j in zip(x, yr):
            plt.text(i, j, str(round(j, 2)))
        
        
        plt.xticks(x, map(int, x))
        plt.ylim(0, 1)
        plt.legend()
        
        plt.title(f"F1-score, rappel, et précision dans l'espace {'rg,by,wb' if color_space == 'opp_colors_uint8' else color_space}")
        plt.xlabel('k plus proches images')
        plt.ylabel('f1-score recall precision')
        plt.ylim(-0.1, 1.1)
        
        plt.savefig(path + f'{color_space}/f1 precision recall.png')
        plt.show()
        print(round(f1_score[-1], 3))#print du f1-score pour la dernière valeurs de k

    
    p = path.split('results.xlsx')[0]
    
    #plot des courbes
    plot_average_prec_rec(sum_yr_gray/(len(new_dfs)/3), sum_yp_gray/(len(new_dfs)/3), p, 'gray')
    plot_average_prec_rec(sum_yr_opp/(len(new_dfs)/3), sum_yp_opp/(len(new_dfs)/3), p, 'opp_colors_uint8')
    plot_average_prec_rec(sum_yr_rgb/(len(new_dfs)/3), sum_yp_rgb/(len(new_dfs)/3), p, 'rgb')
        
        
    



def bar_hist():
    """
    Simple fonction permettant de créer un histogramme
    """
    x_labels = ['full image\ngray', 'full image\nrg,by,wb', 'full image\nrgb', 'ROI\ngray', 'ROI\nrg,by,wb', 'ROI\nrgb']
    y = [0.819, 0.875, 0.638, 0.5, 0.812, 0.762]
    classes = ['4 clusters', '4 clusters', '4 clusters', '9 clusters', '8 clusters', '9 clusters']
    
    x = [i*2 for i in range(len(y))]
    
    plt.figure(figsize=(15, 6))
    
    plt.bar(x, y, width=0.8, align='center', tick_label=x_labels)
    
    plt.ylabel('score', fontsize=14)
    plt.title('patisseries', fontsize=16)
    
    plt.ylim([0,1.05])
    
    for i, v in enumerate(y):
        plt.text(x[i], v+0.01, str(v), ha='center', fontsize=16)
        plt.text(x[i], v+0.05, classes[i], ha='center', fontsize=16)
        
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig(r'C:\Users\scott\OneDrive\Bureau\ter\kmeans_pat')

    plt.show()


def kmeans_evalutation(path_clusters, nb_per_class):
    """
    Evalutation de kmeans
    """
    
    #lecture du fichier résultant d'un clustering
    #contient les images et le label de leur cluster
    df_clusters = pd.read_excel(path_clusters)
    
    img_list = df_clusters['images'].values #listes des images
    clusters_list = df_clusters['labels'].values #liste des clusters des images
    
    same_cluster = np.zeros(len(img_list))
    
    i = 0
    for img, cluster in zip(img_list, clusters_list):#parcours des images
        img_class = img.split('-')[0]#on récupère la classe de l'image dans son nom
        for j in range(len(img_list)):# deuxième parcours des images
            #si les deux images ont la même classe et ont été associées au même cluster (on ignore l'image avec elle même)
            if img != img_list[j] and img_class == img_list[j].split('-')[0] and clusters_list[j] == cluster:
                same_cluster[i] += 1
        i += 1
        
    same_cluster /= (nb_per_class-1)
    average = round(sum(same_cluster)/len(same_cluster), 3)#moyenne
    
    print(average)
        




























