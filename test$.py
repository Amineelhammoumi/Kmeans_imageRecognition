import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans

#%%
#impoter l'image qu'on va utilisé


snow = imread('snow.jpg')
plt.figure(num=None, figsize=(8, 6), dpi=80)
imshow(snow);
#%%
#une image est essentiellement une matrice tridimensionnelle, 
#chaque pixel individuel contenant une valeur pour les canaux rouge, vert et bleu.

def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(),
                       image[:,:,1].flatten(),
                       image[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df
df_snow = image_to_pandas(snow)
print(df_snow.head(5))
#%%
#Cela simplifie la manipulation de l'image car il est plus facile de la considérer comme des données 
#pouvant être introduites dans un algorithme d'apprentissage automatique. 
#Dans notre cas, nous utiliserons l'algorithme K Means pour regrouper l'image.

plt.figure(num=None, figsize=(8, 6), dpi=80)
kmeans = KMeans(n_clusters=  4, random_state = 42).fit(df_snow)
result = kmeans.labels_.reshape(snow.shape[0],snow.shape[1])
imshow(result, cmap='viridis')
plt.show()