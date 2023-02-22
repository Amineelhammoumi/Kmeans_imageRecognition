import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from PIL import Image

#%%
#impoter l'image qu'on va utilisé
image_path = "beach_doggo.PNG"
image_file = Image.open(image_path)
image_file.save("image.jpg", quality=95)

dog = imread('image.jpg')
plt.figure(num=None, figsize=(8, 6), dpi=1200)
imshow(dog);
#%%
#une image est essentiellement une matrice tridimensionnelle, 
#chaque pixel individuel contenant une valeur pour les canaux rouge, vert et bleu.

def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(),
                       image[:,:,1].flatten(),
                       image[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df
df_doggo = image_to_pandas(dog)
print(df_doggo.head(5))
#%%
#Cela simplifie la manipulation de l'image car il est plus facile de la considérer comme des données 
#pouvant être introduites dans un algorithme d'apprentissage automatique. 
#Dans notre cas, nous utiliserons l'algorithme K Means pour regrouper l'image.

plt.figure(num=None, figsize=(8, 6), dpi=1200)
kmeans = KMeans(n_clusters=  3, random_state = 0).fit(df_doggo)
result = kmeans.labels_.reshape(dog.shape[0],dog.shape[1])
imshow(result, cmap='viridis')
plt.show()
#%%
#l'image est regroupée en 3 régions distinctes.  on peut Visualiser chaque région séparément.


fig, axes = plt.subplots(1,3, figsize=(15, 12))
for n, ax in enumerate(axes.flatten()):
    dog = imread('beach_doggo.png')
    dog[:, :, 0] = dog[:, :, 0]*(result==[n])
    dog[:, :, 1] = dog[:, :, 1]*(result==[n])
    dog[:, :, 2] = dog[:, :, 2]*(result==[n])
    ax.imshow(dog);
    ax.set_axis_off()
fig.tight_layout()

#%%


