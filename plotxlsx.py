import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def all_images(path, col):#col: +0,1,2,3,4
    
    df = pd.read_excel(path)
    y = df.columns[::5].values
    x = df['01-01.jpg'].values
    x1 = np.arange(x.size)

    # Create the plot
    fig, ax = plt.subplots()
    i = 0
    for val in y:
        coly = df.columns.get_loc(val)
        sizes = df.iloc[:,coly+col]*50
        y1 = np.ones((y.size,), dtype=np.uint8)*(y.size-1-i)
        ax.scatter(x1, y1, s=sizes, marker='s')
        i += 1

    # Add labels and title
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
    y_uint8 = df.iloc[:,col_idx+1]
    y_int16 = df.iloc[:,col_idx+2]
    y_rgb = df.iloc[:,col_idx+3]
    y_gray = df.iloc[:,col_idx+4]
    
    
    # Create the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    ax.plot(x, y_uint8, label='uint8')
    ax.plot(x, y_int16, label='int16')
    ax.plot(x, y_rgb, label='rgb')
    ax.plot(x, y_gray, label='gray')
    
    # Add labels and title
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
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.savefig('comparaison_espaces.png')
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
    
classes = r'xlsx\classes.xlsx'
labels_kmeans(classes)

path = r'results.xlsx'
#single_image('obj15__345.png', path)
#all_images(path, 1)




















