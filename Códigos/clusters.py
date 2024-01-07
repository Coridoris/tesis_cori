# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:35:44 2023

@author: corir
"""

#%% librerias
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from tqdm import tqdm

# Clase para realizar componentes principales
from sklearn.decomposition import PCA
# Estandarizador (transforma las variables en z-scores)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler() # Creamos el estandarizador para usarlo posteriormente

# importamos los algoritmos de clusterizacion
from sklearn.cluster import KMeans                    # K-means
#eleccion de k
#pip install kneed            # para el codo
from kneed import KneeLocator # importamos el paquete para detectar el codo

# importamos el puntaje de silhouette
from sklearn.metrics import silhouette_score

#jerárquico
# Paquete de scipy que tiene la clase 'dendograma' que vamos a utilizar
import scipy.cluster.hierarchy as shc
#método de clustering jerárquico (bottom-up)
from sklearn.cluster import AgglomerativeClustering
#%% data

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

condicion = temas[0]

path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv'

path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_sinautopercepcion_todas)

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df.drop(['Sujetos', 'Condición'], axis=1)


#%% primero hago PCA a los datos, ya vi que me tengo que quedar con 6 componentes para tener un 70% de la varianza

X = df.to_numpy()

print('Dimensiones de la matriz de features: {}'.format(X.shape))

# Ajustamos el estandarizador
std_scale.fit(X)

# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)

# Creación del modelo. Si el número de componentes no se específica, 
# se obtienen tantas componentes principales como features en nuestro dataset.
nro_PCA = None
pca = PCA(n_components=nro_PCA) #si queremos solo 5 poner n_components = 6

# Ajustamos el modelo a los datos escaleados
pca.fit(X_scaled)

# Obtenemos la descripción de los datos en el espacio de componentes principales
X_pca = pca.transform(X_scaled)

#%% kmeans5

# Creación del modelo KMeans con k = 5
kmeans5 = KMeans(n_clusters=5, init = "random",  n_init = 10000)

# Ajuste del modelo a los datos reducidos en componentes principales
kmeans5.fit(X_pca)
#%% kmeans3
# Creación del modelo KMeans con k = 5
kmeans3 = KMeans(n_clusters=3, init = "random",  n_init = 1000)

# Ajuste del modelo a los datos reducidos en componentes principales
kmeans3.fit(X_pca)
#%% etiquetas
'''
Para acceder a las etiquetas que le asignó el modelo a cada sample usamos 'kmeans.labels_'
'''
kmeans = kmeans3
# Nos fijamos las etiquetas asignadas a las primeras 10 muestras y los counts que recibió cada una
print("etiquetas campeones:", np.unique(kmeans.labels_[:30], return_counts=True))

#devuelve algo de la pinta
'''
devuelve algo de la pinta (array([0, 1, 2, 3]), array([19,  8,  1,  2], dtype=int64))
lo que quiere decir que en los primeros 30 datos (que corresponden a los 30 relatos de campeones) 19 fueron asignados al
cluster 0, 8 al cluster 1, 1 al cluster 2 y 2 al cluster 3
'''
#en presencial
print("etiquetas presencial:", np.unique(kmeans.labels_[30:60], return_counts=True))
#cfk
print("etiquetas cfk:", np.unique(kmeans.labels_[60:90], return_counts=True))
#arabia
print("etiquetas arabia:", np.unique(kmeans.labels_[90:120], return_counts=True))
#antes de venir
print("etiquetas filler:", np.unique(kmeans.labels_[120:], return_counts=True))

'''
los mas separados son campeones y antes venir, en particular 
campeones está 2/3 en el cluster 0 y 1/3 en el cluster 1
antes de venir esta 2/3 en el cluster 3 y 1/3 en el cluster 2
presencial se encuentra 1/3 en el cluster 2 y el resto repartido mas que nada en el 0 3 y 4
cfk 1/2 esta en el cluster 4 y después la mayoría se reparte en el cluster 0 y 2
arabia tenes 1/3 en el 4 y otro en el 2, el resto repartido en todos los demas clusters (0, 1, 3)
'''
#%% cluster en color, temas en forma

'''
Para acceder a la posición de los centroids en el espacio de 6 (o 30) PCs usamos 'kmeans.cluster_centers_ 
'''

# Guardo las posiciones de los centroids
centroids = kmeans.cluster_centers_

# Printeo las dimensiones de las posiciones
print("Shape de los centroids:",centroids.shape)
# Printeo las posiciones de las primeras 5 muestras en sus primeras dos componentes principales
print(centroids[:5,[0,1]])

# Este bloque es similar al anterior pero agregando color a cada sample en el scatter plot según la etiqueta asignada

fig, ax = plt.subplots(figsize = (20, 7))

# Hacemos un scatter plot de cada uno de los datos
ax.scatter(X_pca[0:30, 0], X_pca[0:30, 1], marker = "o", c=kmeans.labels_[0:30], label = temas[0])
ax.scatter(X_pca[30:60, 0], X_pca[30:60, 1], marker = "v", c=kmeans.labels_[30:60], label = temas[1])
ax.scatter(X_pca[60:90, 0], X_pca[60:90, 1], marker = "s", c=kmeans.labels_[60:90], label = temas[2])
ax.scatter(X_pca[90:120, 0], X_pca[90:120, 1],  marker = "*", c=kmeans.labels_[90:120], label = temas[3])
ax.scatter(X_pca[120:, 0], X_pca[120:, 1], marker = "d",  c=kmeans.labels_[120:], label = temas[4])
ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, linewidths=1,
            c=np.unique(kmeans.labels_), edgecolors='black')

## Por cada dato escribimos a qué instancia corresponde
#for i in range(data.shape[0]):
#  ax.text(X_pca[i, 0], X_pca[i, 1], s = i)

ax.set_xlabel('Primer componente principal', fontsize = 20)
ax.set_ylabel('Segunda componente principal', fontsize = 20)
ax.legend(fontsize = 15)
ax.tick_params(axis='x', labelsize=15)  
ax.tick_params(axis='y', labelsize=15)

path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
plt.savefig(path_imagenes + f'/6PCA_{kmeans}_markerstemas_colorclusters.png')        

#%% cluster en formas, temas en color

# Guardo las posiciones de los centroids
centroids = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(20, 7))

# Definir colores y marcadores
colores = ['red', 'blue', 'green', 'purple', 'orange']
marcadores = ['o', 'v', 's', '*', 'd']

# Suponiendo que kmeans.labels_ contiene las etiquetas de clústeres
tema_contador = 0
for i in range(len(X_pca)):
    cluster_index = kmeans.labels_[i]
    if i < 30:
        color_index = 0
    elif i < 60:
        color_index = 1
    elif i < 90:
        color_index = 2
    elif i < 120:
        color_index = 3
    else:
        color_index = 4

    if i in [0, 30, 60, 90, 120]:
        ax.scatter(X_pca[i, 0], X_pca[i, 1], marker=marcadores[cluster_index], c=colores[color_index], label = temas[tema_contador])
        tema_contador = tema_contador + 1
    else:
        ax.scatter(X_pca[i, 0], X_pca[i, 1], marker=marcadores[cluster_index], c=colores[color_index])

for i, centroid in enumerate(centroids):
    ax.scatter(centroid[0], centroid[1], marker=marcadores[i], s=200, linewidths=1,
                c='black', edgecolors='black')

ax.set_xlabel('Primer componente principal', fontsize = 20)
ax.set_ylabel('Segunda componente principal', fontsize = 20)
ax.legend(fontsize = 15)
ax.tick_params(axis='x', labelsize=15)  
ax.tick_params(axis='y', labelsize=15)

path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
plt.savefig(path_imagenes + f'/6PCA_{kmeans}_markersclusters_colortemas.png')   

plt.show()

#%% eleccion de k
'''
Vamos a aplicar el método KMeans al dataset pero cambiando el número de clusters k y guardaremos el puntaje de la 
función objetivo, SSE (suma de los cuadrados de la distancia euclidea de cada cluster), en una lista
'''

sse = [] # acá vamos a guardar el puntaje de la función objetivo

for k in tqdm(range(2, 30)):
  kkmeans = KMeans(n_clusters=k)
  kkmeans.fit(X_pca)
  sse.append(kkmeans.inertia_)
#%% figura codo
fig, ax = plt.subplots(figsize = (20, 7))

# esta dos lineas las agrego para que se vea la elección de KneeLocator para el codo en este gráfico
ax.scatter(13, sse[11], color='red', s=200) # agregamos un punto rojo al plot de tamaño s=200 en el lugar donde se encuentra el codo
ax.text(13-0.5, sse[11]-100, s="codo", fontsize = 15)       # agregamos un texto abajo para indicar qué representa el punto

# estas lineas son el grafico de SSEvsK
ax.scatter(range(2, 30), sse)            
ax.set_xticks(range(2, 30))
ax.set_xlabel("Número de clusters", fontsize = 20)
ax.set_ylabel("SSE", fontsize = 20)

ax.tick_params(axis='x', labelsize=15)  
ax.tick_params(axis='y', labelsize=15)


'''
A ojo distinguimos un codo entre k=10 y k=13 porque al agregar más clusters aumentamos la complejidad del modelo pero SEE 
disminuye en menor proporción.
Usamos la función 'KneeLocator' para detectar el codo. Para ello le tenemos que pasar los valores de K, SEE, la forma 
de la fución (cóncava o convexa) y la dirección (creciente o decreciente). No siempre conviene usar lo que devuelve
este método
'''

kl = KneeLocator(range(2, 30), sse, curve="convex", direction="decreasing")

print("El codo está en k =", kl.elbow)

path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
plt.savefig(path_imagenes + f'/metodocodo_6PCA_{kmeans}.png') 

#%% clusters con k óptimo
'''
ver el modelo con el k óptimo
'''

# Creación del modelo KMeans con k = 8
kmeans13 = KMeans(n_clusters=13)

# Ajuste del modelo a los datos reducidos en componentes principales
kmeans13.fit(X_pca)

# Guardamos la posición de los centroids
centroids13 = kmeans13.cluster_centers_

# Plot
fig, ax = plt.subplots(figsize = (20, 7))

# Hacemos un scatter plot de cada uno de los datos
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans13.labels_)
ax.scatter(centroids13[:, 0], centroids13[:, 1], marker="X", s=200, linewidths=2,
            c=np.unique(kmeans13.labels_),edgecolors='black')
ax.legend()

## Por cada dato escribimos a qué instancia corresponde
#for i in range(data.shape[0]):
#  ax.text(X_pca[i, 0], X_pca[i, 1], s = i)

ax.set_xlabel('Primer componente principal', fontsize = 20)
ax.set_ylabel('Segunda componente principal', fontsize = 20)

path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
plt.savefig(path_imagenes + f'/metodocodo_6PCA_{kmeans}_clusters.png') 

#%% silhouette
'''
El coeficiente de Silhouette de cada sample la podemos obtener con la clase 'silhouette_samples' de sklearn.metrics

El puntaje de Silhouette es el promedio de los coeficientes de Silhouette de todas las samples y se computa con la clase 
'silhouette_score' de sklearn.metrics. Hay que pasarle a la función los datos y sus etiquetas.

'''

# Creamos una lista para guardar de los coeficientes de silhouette para cada valor de k
silhouette_coefficients = []

# Se necesita tener al menos 2 clusters y a los sumo N-1 (con N el numero de muestras) para obtener coeficientes de Silohuette
for k in range(2, 20):
     kkkmeans = KMeans(n_clusters=k)
     kkkmeans.fit(X_pca)
     score = silhouette_score(X_pca, kkkmeans.labels_)
     silhouette_coefficients.append(score)
     
fig, ax = plt.subplots(figsize = (24, 7))

# estas lineas son el grafico de SSEvsK
ax.scatter(range(2, 20), silhouette_coefficients)            
ax.set_xticks(range(2, 20))
ax.set_xlabel("Número de clusters", fontsize = 20)
ax.set_ylabel("Promedio coeficientes de Silhouette", fontsize = 20)

kklabels = kkkmeans.labels_
kkklabels = kklabels.copy()

np.random.shuffle(kkklabels)

silhouette_score(X_pca, kkklabels)

ax.tick_params(axis='x', labelsize=15)  
ax.tick_params(axis='y', labelsize=15)

path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
plt.savefig(path_imagenes + f'/silhouette_6PCA_{kmeans}.png') 


'''
En este plot vemos que el puntaje de Silhouette < 0.2 y tiene poca desviación. 

Se puede decir que **no se encuentran estructuras fuertes en los datos**
'''

#%% jerárquico
'''
Los métodos de clustering jerárquico buscan agrupar las samples más similares para formar grupos con características 
similares.

1. Agglomerative: cada sample es un cluster y en cada paso va agrupando los clusters más similares hasta quedarse con 
un solo cluster
2. Divisive: todas las samples comienzan en el mismo cluster y en cada paso va cortando las samples menos similares 
hasta que todas las samples sean un cluster distinto
'''

# Plot del dendograma
distancia = ["single", "complete", "average", "centroid", "ward"]
distancia_label = ["min", "max", "promedio", "centroid", "ward"]

for i, dist in enumerate(distancia):
    plt.figure(figsize=(10, 7))
    plt.title("Dendograma", fontsize = 20)
    plt.ylabel(f"Distancia {distancia_label[i]}", fontsize = 20)
    
    # Con la función 'dendogram' graficamos el dendograma. 
    dend = shc.dendrogram(shc.linkage(X_pca, method=dist))  # El input de esta función es la función 'linkage' 
                                                    #donde se especifica la distancia para utlizar en cada paso del método
                                                    
    path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
    plt.savefig(path_imagenes + f'/{nro_PCA}PCA_dendograma_{distancia_label[i]}.png')                                               
'''
single: min
complete: max
centroid: centroid
average: average
ward: ward
'''
#%%definimos k                                               
'''
Lo que buscamos en el dendograma es la mayor distancia vertical sin que haya una línea horizontal para hacerle un corte 
(representado como una linea horizontal que cruza todos los datos) y quedarnos con k clusters (donde k es el número de 
lineas verticales que intersectan el corte. 
'''

# Plot del dendograma del dataset de clientes
plt.figure(figsize=(10, 7))
plt.title("Dendograma")
plt.ylabel("Distancia ward")

# Con la función 'dendogram' graficamos el dendograma. 
dend = shc.dendrogram(shc.linkage(X_pca, method='ward'))  # El input de esta función es la función 'linkage' donde se especifica la distancia para utlizar en cada paso del método
plt.axhline(17.48, c='r', label = "Cinco clusters")
plt.axhline(22, c='k', label = "Tres clusters")
plt.legend()

# Plot del dendograma del dataset de clientes
plt.figure(figsize=(10, 7))
plt.title("Dendograma")
plt.ylabel("Distancia min")

# Con la función 'dendogram' graficamos el dendograma. 
dend = shc.dendrogram(shc.linkage(X_pca, method='single'))  # El input de esta función es la función 'linkage' donde se especifica la distancia para utlizar en cada paso del método
plt.axhline(5.77, c='r', label = "Cinco clusters")
plt.axhline(7, c='k', label = "Tres clusters")
plt.legend()
#%% clusterización jerárquica con k óptimo
'''
Ahora sí aplicamos el método de clusterización jerárquica (bottom-up) con 5 clusters, la distancia euclidea para la afinidad y la distancia ward para el linkage
'''
nro_clusters = 5
path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7' 
distancia = ["single", "complete", "average", "ward"]#, "centroid"]
distancia_label = ["min", "max", "promedio", "ward"]#, "centroid"]
for j, dist in enumerate(distancia):
    # Creamos el modelo
    cluster = AgglomerativeClustering(n_clusters = nro_clusters, affinity='euclidean', linkage=dist)
    
    # Lo ajustamos con los datos
    cluster.fit_predict(X_pca)  # fit_predict hace lo mismo que fit pero devuelve el vector de etiquetas de las samples
    
    
    labels = cluster.labels_ 
    
    centroids_jerarquico = []
    for cluster_label in np.unique(labels):
        cluster_points = X_pca[labels == cluster_label]
        centroid_jerarquico = cluster_points.mean(axis=0)
        centroids_jerarquico.append(centroid_jerarquico)
        
    centroids_jerarquico = np.array(centroids_jerarquico)
    
    plt.figure(figsize=(10, 7))
    plt.title(f"Distancia {distancia_label[j]}")
    plt.scatter(X_pca[0:30, 0], X_pca[0:30, 1], marker = "o", c=cluster.labels_[0:30], label = temas[0])
    plt.scatter(X_pca[30:60, 0], X_pca[30:60, 1], marker = "v", c=cluster.labels_[30:60], label = temas[1])
    plt.scatter(X_pca[60:90, 0], X_pca[60:90, 1], marker = "s", c=cluster.labels_[60:90], label = temas[2])
    plt.scatter(X_pca[90:120, 0], X_pca[90:120, 1],  marker = "*", c=cluster.labels_[90:120], label = temas[3])
    plt.scatter(X_pca[120:, 0], X_pca[120:, 1], marker = "d",  c=cluster.labels_[120:], label = temas[4])
    plt.scatter(centroids_jerarquico[:, 0], centroids_jerarquico[:, 1], marker="X", s=200, linewidths=1,
                c=np.unique(cluster.labels_), edgecolors='black')
    plt.xlabel('Primer componente principal', fontsize = 20)
    plt.ylabel('Segunda componente principal', fontsize = 20)
    plt.legend(fontsize = 15)
    plt.tick_params(axis='x', labelsize=15)  
    plt.tick_params(axis='y', labelsize=15)
                                               
    plt.savefig(path_imagenes + f'/{nro_PCA}PCA_clusters_markerstemas_colorclusters_{distancia_label[j]}_k{nro_clusters}.png')       

#%%


nro_clusters = 3
path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7' 
distancia = ["single", "complete", "average", "ward"]#, "centroid"]
distancia_label = ["min", "max", "promedio", "ward"]#, "centroid"]
for j, dist in enumerate(distancia):
    # Creamos el modelo
    cluster = AgglomerativeClustering(n_clusters = nro_clusters, affinity='euclidean', linkage=dist)
    
    # Lo ajustamos con los datos
    cluster.fit_predict(X_pca)  # fit_predict hace lo mismo que fit pero devuelve el vector de etiquetas de las samples
    
    
    labels = cluster.labels_ 
    
    centroids_jerarquico = []
    for cluster_label in np.unique(labels):
        cluster_points = X[labels == cluster_label]
        centroid_jerarquico = cluster_points.mean(axis=0)
        centroids_jerarquico.append(centroid_jerarquico)
        
    centroids_jerarquico = np.array(centroids_jerarquico)
    
    plt.figure(figsize=(10, 7))
    plt.title(f"Distancia {distancia_label[j]}")
    # Definir colores y marcadores
    colores = ['red', 'blue', 'green', 'purple', 'orange']
    marcadores = ['o', 'v', 's', '*', 'd']
    
    # Suponiendo que kmeans.labels_ contiene las etiquetas de clústeres
    tema_contador = 0
    for i in range(len(X_pca)):
        cluster_index = cluster.labels_[i]
        if i < 30:
            color_index = 0
        elif i < 60:
            color_index = 1
        elif i < 90:
            color_index = 2
        elif i < 120:
            color_index = 3
        else:
            color_index = 4
    
        if i in [0, 30, 60, 90, 120]:
            ax.scatter(X_pca[i, 0], X_pca[i, 1], marker=marcadores[cluster_index], c=colores[color_index], label = temas[tema_contador])
            tema_contador = tema_contador + 1
        else:
            ax.scatter(X_pca[i, 0], X_pca[i, 1], marker=marcadores[cluster_index], c=colores[color_index])
    
    for i, centroid in enumerate(centroids_jerarquico):
        ax.scatter(centroids_jerarquico[0], centroids_jerarquico[1], marker=marcadores[i], s=200, linewidths=1,
                    c='black', edgecolors='black')

    plt.xlabel('Primer componente principal', fontsize = 20)
    plt.ylabel('Segunda componente principal', fontsize = 20)
    ax.legend(fontsize = 15)
    plt.tick_params(axis='x', labelsize=15)  
    plt.tick_params(axis='y', labelsize=15)
                                               
    plt.savefig(path_imagenes + f'/{nro_PCA}PCA_clusters_markersclusters_colortemas_{distancia_label[j]}_k{nro_clusters}.png')  