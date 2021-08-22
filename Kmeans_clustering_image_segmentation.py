#Clustering k means for image segmentation

import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import data

plt.close('all')

#We calculate the initial centroids
def cen_inicial(k):
    #k is the number of clusters
    centroide_inicial =np.array([random.randint(0, 255) for i in range(k)])
    return centroide_inicial

#Then we calculate the euclidian distance
def euclidean_distance(data, centroide_inicial):
    distancia =np.argmin((np.sqrt((data - centroide_inicial)**2)))
    return distancia

imagen = data.camera() #We import an image from skimage
#Showing original image
plt.figure()
plt.imshow(imagen, cmap='gray')
plt.show()

im = imagen.flatten()

n_cluster = int(input('Ingrese el número de clusters: ')) #Enter the number of clusters
n_iteraciones = int(input('Ingrese el número de iteraciones maximas: ')) #Number of iterations

centros_iniciales = cen_inicial(n_cluster)
for i in range(n_iteraciones):
    #we calculate the euclidian distance of each pixel
    cent_cercano = []
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            d = euclidean_distance(imagen[i,j],centros_iniciales)
            cent_cercano.append(d)
    #We calculate the new centroids
    el = {} #The diccionary allows us to average the pixels in each centroid in order to calculate the new centroid
    for k in range(n_cluster):
        n=0
        a=0
        for i in range(len(im)):
            if cent_cercano[i] == k:
                a = im[i] + a
                n = n+1
                el[k] = a,n
    #Average = new centroid
    centros_finales = np.zeros(n_cluster)
    for k in range(n_cluster):
        centros_finales[k] = el[k][0]/el[k][1]
    
    centros_iniciales = centros_finales
    print(centros_finales)
    
#Now we print the clusters in the image
imagen1 = np.zeros(im.shape)
w = 0
for s in range(n_cluster):
    for i in range(len(im)):
        if s == cent_cercano[i]:
            imagen1[i] = w
    w = w + round(255/n_cluster) #Each cluster will have a different gray level
    
imagen1 = np.array(imagen1).reshape(imagen.shape)

plt.figure()
plt.imshow(imagen1,cmap='gray')
plt.show()
