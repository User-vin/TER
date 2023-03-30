import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ter import *
from glob import glob


def all_images(path, col):#col: +0,1,2,3
    
    df = pd.read_excel(path)
    y = df.columns[::4].values
    x = df['01-01.jpg'].values
    x1 = np.arange(x.size)


    fig, ax = plt.subplots()
    i = 0
    for val in y:
        coly = df.columns.get_loc(val)
        sizes = df.iloc[:,coly+col]*50
        print(sizes)
        y1 = np.ones((y.size,), dtype=np.uint8)*(y.size-1-i)
        ax.scatter(x1, y1, s=sizes, marker='s')
        i += 1


    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Comparaisons des vecteurs caractéristiques')
    
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.tick_params(axis='x', labelrotation=70, labelsize=5)
    
    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y)
    ax.tick_params(axis='y', labelsize=5)
    
    fig.set_size_inches(20,20)


    plt.savefig('comparaison_vecteurs.png')
    plt.show()


def single_image(name, path):
    df = pd.read_excel(path)
    col_idx = df.columns.get_loc(name)
    x = df[name]
    #y_uint8 = df.iloc[:,col_idx+1]
    #y_int16 = df.iloc[:,col_idx+2]
    #y_rgb = df.iloc[:,col_idx+2]
    y_gray = df.iloc[:,col_idx+2]#+3
    
    
    # Create the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    #ax.plot(x, y_uint8, label='uint8')
    #ax.plot(x, y_int16, label='int16')
    #ax.plot(x, y_rgb, label='rgb')
    ax.plot(x, y_gray, label='gray')
    
    
    ax.set_xlabel('images')
    ax.set_ylabel('distance')
    ax.set_title(f'comparaison {name}')
    ax.tick_params(axis='x', labelrotation=90, labelsize=10)
    ax.grid(True)
    
    ax.axhline(y=0.1, linewidth=0.5)
    ax.axhline(y=0.3, linewidth=0.5)
    ax.axhline(y=0.5, linewidth=0.5)
    ax.axhline(y=0.7, linewidth=0.5)
    ax.axhline(y=0.9, linewidth=0.5)
    
    ax.legend()
    
    plt.savefig(name)
    plt.show()


def labels_kmeans(path):
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
    
    
    
def precision_recall_f1(path):
    df = pd.read_excel(path)
    num_columns = len(df.columns)
    
    for i in range(0, num_columns, 4):
        
        sub_df = df.iloc[:, i:i+4]
        new_dfs = []
        for k in range(0, num_columns, 4):
            sub_df = df.iloc[:, k:k+4]
            
            for j in range(3):
                new_cols = [sub_df.columns[0], sub_df.columns[j+1]]
                new_df = sub_df[new_cols]
                new_df = new_df.sort_values(by=new_cols[1], ascending=False)
                new_dfs.append(new_df)
                
    sum_color_space_precision_gray = 0
    sum_color_space_recall_gray = 0
    
    sum_color_space_precision_opp = 0
    sum_color_space_recall_opp = 0
    
    sum_color_space_precision_rgb = 0
    sum_color_space_recall_rgb = 0
    
                
    for i in range(len(new_dfs)):
        plot_df = new_dfs[i].iloc[1:6]
        s = 0
        yp = []
        name = plot_df.columns[0].split('.')[0]
        
        for j in range(5):
            #classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('-')[0]
            #s += 1 if classe == name.split('-')[0] else 0
            classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('__')[0]
            s += 1 if classe == name.split('__')[0] else 0
            yp.append(s/(j+1))
            
        s = 0
        yr = []
        for j in range(5):
            #classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('-')[0]
            #s += 1 if classe == name.split('-')[0] else 0
            classe = plot_df.iloc[:, 0].values[j].split('.')[0].split('__')[0]
            s += 1 if classe == name.split('__')[0] else 0
            yr.append(s/5)
            
        color_space = plot_df.columns[1].split('.')[0]    
        
        if color_space == 'gray':
            sum_color_space_precision_gray += yp[4]
            sum_color_space_recall_gray += yr[4]
            
        elif color_space == 'opp_colors_uint8':
            sum_color_space_precision_opp += yp[4]
            sum_color_space_recall_opp += yr[4]
        
        elif color_space == 'rgb':
            sum_color_space_precision_rgb += yp[4]
            sum_color_space_recall_rgb += yr[4]
        
            
        plt.plot(plot_df.iloc[:, 0], yp, label='precision')
        plt.plot(plot_df.iloc[:, 0], yr, label='recall')
        plt.legend()
        
        
        plt.title(f"Plot {name} {color_space}")
        text = 'precision: {:.2f}\nrecall: {:.2f}'.format(yp[4], yr[4])
        plt.text(1, 1, text, transform=plt.gca().transAxes, va='top', ha='right')
        plt.xlabel('Images')
        plt.ylabel('precision / recall')
        plt.ylim(-0.1, 1.1)
        
        plt.savefig(f'C:/Users/scott/OneDrive/Bureau/prec_rec/coil_100/new/sub_dataset/normal/{color_space}/{name}')

        plt.show()
        
        
    print(f'gray average precison: {sum_color_space_precision_gray/(len(new_dfs)/3)}\naverage recall: {sum_color_space_recall_gray/(len(new_dfs)/3)}')
    print(f'opp average precison: {sum_color_space_precision_opp/(len(new_dfs)/3)}\naverage recall: {sum_color_space_recall_opp/(len(new_dfs)/3)}')
    print(f'rgb average precison: {sum_color_space_precision_rgb/(len(new_dfs)/3)}\naverage recall: {sum_color_space_recall_rgb/(len(new_dfs)/3)}')
    

    
    
    
classes = r'C:\Users\scott\OneDrive\Bureau\papillons\gray\kmeans\classes.xlsx'
#labels_kmeans(classes)

path = r'C:\Users\scott\OneDrive\Bureau\ter\results.xlsx'

#for i in glob(r'C:\Users\scott\OneDrive\Bureau\imgs\*'):
#    name = i.split('\\')[-1]
#    single_image(name, path)
#path = r'C:\Users\scott\OneDrive\Bureau\papillons\results.xlsx'    
#all_images(path, 1)

path = r'C:\Users\scott\OneDrive\Bureau\ter\results.xlsx'
precision_recall_f1(path)


















