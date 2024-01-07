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
#from sklearn.cluster import KMedoids
#eleccion de k
#pip install kneed            # para el codo
from kneed import KneeLocator # importamos el paquete para detectar el codo

#kmemoids
import kmedoids
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn_extra.cluster import KMedoids #pip install scikit-learn-extra

# importamos el puntaje de silhouette
from sklearn.metrics import silhouette_score
#para el perfil de silhouette necesitamos
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

#jerárquico
# Paquete de scipy que tiene la clase 'dendograma' que vamos a utilizar
import scipy.cluster.hierarchy as shc
#método de clustering jerárquico (bottom-up)
from sklearn.cluster import AgglomerativeClustering

#R index
from sklearn.metrics.cluster import adjusted_rand_score #si da cerca de 0 es azar, si da cerca de 1 buen match

#DBScan
from sklearn.cluster import DBSCAN

#TSNE
from sklearn.manifold import TSNE

#%% el santo trial

entrevista = 'Primera'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]
#%% data

condicion = temas[0]

path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv'

path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_conautopercepcion_todas)
 
mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}

#tiro por PCA
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm'], axis = 1)
#nro_pca_todo = 7 #70% se alcanza con 7
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop'], axis = 1)
#nro_pca_todo = 6#el 70% se explica con 6
#df_pca = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2'], axis = 1)
#nro_pca = 5#el 70% se explica con 5
#de aca para abajo funciona muy mal con kmedoids
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2', 'L3', 'k_mean', 'Comunidades_LSC', 'Detalles internos norm', 'Detalles externos norm'], axis = 1)
#nro_pca = 5#el 70% se explica con 5
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2', 'L3', 'k_mean', 'Comunidades_LSC', 'Detalles internos norm', 'Detalles externos norm', 'num_nodes_LSC','primera_persona_norm','num advs norm','Negativo pysent','diámetro'], axis = 1)
#nro_pca = 5#el 70% se explica con 3
#df_pca = df_pca.dropna()

nro_pca = 7
df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin([5,2])]

condicion_labels = list(df['Condición'])

def encontrar_intervalos(lista, valor):
    intervalos = []
    inicio = None

    for i, elemento in enumerate(lista):
        if elemento == valor:
            if inicio is None:
                inicio = i
        elif inicio is not None:
            intervalos.append((inicio, i - 1))
            inicio = None

    # Manejar el caso cuando el último elemento es igual al valor
    if inicio is not None:
        intervalos.append((inicio, len(lista) - 1))

    return intervalos

ind_camp = encontrar_intervalos(condicion_labels, 1)
ind_pres = encontrar_intervalos(condicion_labels, 2)
ind_cfk = encontrar_intervalos(condicion_labels, 3)
ind_ar = encontrar_intervalos(condicion_labels, 4)
ind_fil = encontrar_intervalos(condicion_labels, 5)


df = df.drop(['Sujetos', 'Condición'], axis=1)


#df_noanova = df.drop(['tercera_persona_norm', 'num advs norm'], axis=1) #no pasan ANOVA con 7 PCA tenes el 70%
nro_pca_noanova = 7
#Estas no tienen mas de dos grupos con diferencias significativas entre ellos (de 10 total)
#df_notukey = df_noanova.drop(['num noun norm', 'num propn norm', 'Intensidad pysent', 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3', 'diámetro'], axis=1)
#si tiras las dos lineas anteriores con 5 PCA tenes el 70%
nro_pca_notuckey = 5

df = df.drop(['num advs norm', 'selfloops', 'diámetro', 'transitivity'], axis=1)

nro_pca = 7

#%% FUNCIONES


def markerstemas_colorcluster(labels, X, temaslabels, save = None, centroids =  None, title =  None):

    fig, ax = plt.subplots(figsize = (20, 7))
    
    # Hacemos un scatter plot de cada uno de los datos
    ax.scatter(X[ind_cfk[0][0]:ind_cfk[0][1], 0], X[ind_cfk[0][0]:ind_cfk[0][1], 1], marker = "o", c=labels[ind_cfk[0][0]:ind_cfk[0][1]], label = temaslabels[0])
    ax.scatter(X[ind_camp[0][0]:ind_camp[0][1], 0], X[ind_camp[0][0]:ind_camp[0][1], 1], marker = "v", c=labels[ind_camp[0][0]:ind_camp[0][1]], label = temaslabels[1])
    ax.scatter(X[ind_fil[0][0]:ind_fil[0][1], 0], X[ind_fil[0][0]:ind_fil[0][1], 1], marker = "s", c=labels[ind_fil[0][0]:ind_fil[0][1]], label = temaslabels[2])
    ax.scatter(X[ind_pres[0][0]:ind_pres[0][1], 0], X[ind_pres[0][0]:ind_pres[0][1], 1],  marker = "*", c=labels[ind_pres[0][0]:ind_pres[0][1]], label = temaslabels[3])
    ax.scatter(X[ind_ar[0][0]:ind_ar[0][1], 0], X[ind_ar[0][0]:ind_ar[0][1], 1], marker = "d",  c=labels[ind_ar[0][0]:ind_ar[0][1]], label = temaslabels[4])
    
    
    ax.set_xlabel('Primer componente principal', fontsize = 20)
    ax.set_ylabel('Segunda componente principal', fontsize = 20)
    ax.legend(fontsize = 15)
    ax.tick_params(axis='x', labelsize=15)  
    ax.tick_params(axis='y', labelsize=15)
    
    if type(centroids) == np.ndarray:
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, linewidths=1,
                    c=np.unique(labels), edgecolors='black')
        
    if save != None:
        path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Graficos/Cluster'                                                
        plt.savefig(path_imagenes + f'/6PCA_{save}_markerstemas_colorclusters.png')
    
    if title != None:
        ax.set_title(f"{title}", fontsize = 20)
        
    return 'ok'

def markerscluster_colortemas(labels, X, temaslabels, save = None, centroids =  None, title =  None):
    fig, ax = plt.subplots(figsize=(20, 7))

    # Definir colores y marcadores
    colores = ['red', 'blue', 'green', 'purple', 'orange']
    marcadores = ['o', 'v', 's', '*', 'd']

    # Suponiendo que kmeans.labels_ contiene las etiquetas de clústeres
    tema_contador = 0
    for i in range(len(X)):
        cluster_index = labels[i]
        if i < ind_cfk[0][1]:
            color_index = 0
        elif i < ind_camp[0][1]:
            color_index = 1
        elif i < ind_fil[0][1]:
            color_index = 2
        elif i < ind_pres[0][1]:
            color_index = 3
        else:
            color_index = 4

        if i in [0, ind_cfk[0][1], ind_camp[0][1], ind_fil[0][0], ind_pres[0][1]]:
            ax.scatter(X[i, 0], X[i, 1], marker=marcadores[cluster_index], c=colores[color_index], label = temas[tema_contador])
            tema_contador = tema_contador + 1
        else:
            ax.scatter(X[i, 0], X[i, 1], marker=marcadores[cluster_index], c=colores[color_index])
    
    if type(centroids) == np.ndarray:
        for j, centroid in enumerate(centroids):
            ax.scatter(centroid[0], centroid[1], marker=marcadores[j], s=200, linewidths=1,
                        c='black', edgecolors='black')

    ax.set_xlabel('Primer componente principal', fontsize = 20)
    ax.set_ylabel('Segunda componente principal', fontsize = 20)
    ax.legend(fontsize = 15)
    ax.tick_params(axis='x', labelsize=15)  
    ax.tick_params(axis='y', labelsize=15)
    
    if save != None:
        path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Graficos/Cluster'                                                
        plt.savefig(path_imagenes + f'/6PCA_{save}_markersclusters_colortemas.png')   

    plt.show()
    
    if title != None:
        ax.set_title(f"{title}", fontsize = 20)
    
    return 'ok'

def etiquetas(labels):
    #devuelve algo de la pinta
    '''
    devuelve algo de la pinta (array([0, 1, 2, 3]), array([19,  8,  1,  2], dtype=int64))
    lo que quiere decir que en los primeros 30 datos (que corresponden a los 30 relatos de campeones) 19 fueron asignados al
    cluster 0, 8 al cluster 1, 1 al cluster 2 y 2 al cluster 3
    '''
    if ind_camp != []:
        print("etiquetas campeones:", np.unique(labels[ind_camp[0][0]:ind_camp[0][1]], return_counts=True))
    #en presencial
    if ind_pres != []:
        print("etiquetas presencial:", np.unique(labels[ind_pres[0][0]:ind_pres[0][1]], return_counts=True))
    #cfk
    if ind_cfk != []:
        print("etiquetas cfk:", np.unique(labels[ind_cfk[0][0]:ind_cfk[0][1]], return_counts=True))
    #arabia
    if ind_ar != []:
        print("etiquetas arabia:", np.unique(labels[ind_ar[0][0]:ind_ar[0][1]], return_counts=True))
    #antes de venir
    if ind_fil != []:
        print("etiquetas filler:", np.unique(labels[ind_fil[0][0]:ind_fil[0][1]], return_counts=True))
    
    return 'ok'


def metodo_codo(X, metodo, nro_clusters, save = None):
    '''
    Vamos a aplicar el método KMeans o Kmedoids al dataset pero cambiando el número de clusters k y guardaremos el puntaje de la 
    función objetivo, SSE (suma de los cuadrados de la distancia euclidea de cada cluster), en una lista
    '''
    
    sse = [] # acá vamos a guardar el puntaje de la función objetivo
    
    if metodo == 'kmeans':
        length = np.sqrt((X**2).sum(axis=1))[:,None]
        X_norm = X / length
        for k in tqdm(range(2, nro_clusters)):
          kmeans = KMeans(n_clusters=k)
          kmeans.fit(X_norm)
          sse.append(kmeans.inertia_)
          
    if metodo == 'kmedoids fast PAM':
        for k in tqdm(range(2, nro_clusters)):
            diss = euclidean_distances(X)
            fp = kmedoids.fasterpam(diss, nro_clusters)
            sse.append(fp.loss)
            
    if metodo == 'kmedoids':
        for k in range(2, nro_clusters):
             kmedoids_ = KMedoids(n_clusters=k, metric = "cosine")
             kmedoids_.fit(X)
             sse.append(kmedoids_.inertia_)  
            
    # figura codo
    
    '''
    Usamos la función 'KneeLocator' para detectar el codo. Para ello le tenemos que pasar los valores de K, SEE, la forma 
    de la fución (cóncava o convexa) y la dirección (creciente o decreciente). No siempre conviene usar lo que devuelve
    este método
    '''
    print(len(sse))
    print(len(range(2, nro_clusters)))
    kl = KneeLocator(range(2, nro_clusters), sse, curve="convex", direction="decreasing")
    
    print("El codo está en k =", kl.elbow)
    fig, ax = plt.subplots(figsize = (20, 7))
    
    # esta dos lineas las agrego para que se vea la elección de KneeLocator para el codo en este gráfico
    ax.scatter(kl.elbow, sse[kl.elbow-2], color='red', s=200) # agregamos un punto rojo al plot de tamaño s=200 en el lugar donde se encuentra el codo
    ax.text(kl.elbow-0.5, sse[kl.elbow-2]-100, s="codo", fontsize = 15)       # agregamos un texto abajo para indicar qué representa el punto
    
    # estas lineas son el grafico de SSEvsK
    ax.scatter(range(2, nro_clusters), sse)            
    ax.set_xticks(range(2, nro_clusters))
    ax.set_xlabel("Número de clusters", fontsize = 20)
    ax.set_ylabel("SSE", fontsize = 20)
    
    ax.tick_params(axis='x', labelsize=15)  
    ax.tick_params(axis='y', labelsize=15)
    
    if save != None:
        path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Graficos/Cluster'                                                
        plt.savefig(path_imagenes + f'/metodocodo_6PCA_{save}.png') 
    
    return 'ok'

def metodo_silhouette(X, metodo, nro_clusters, save = None):
    # Creamos una lista para guardar de los coeficientes de silhouette para cada valor de k
    silhouette_coefficients = []
    
    # Se necesita tener al menos 2 clusters y a los sumo N-1 (con N el numero de muestras) para obtener coeficientes de Silohuette
    if metodo == 'kmeans':
        length = np.sqrt((X**2).sum(axis=1))[:,None]
        X_norm = X / length
        for k in range(2, nro_clusters):
             kmeans = KMeans(n_clusters=k)
             kmeans.fit(X_norm)
             score = silhouette_score(X_norm, kmeans.labels_)
             silhouette_coefficients.append(score)
        kklabels = kmeans.labels_
        kkklabels = kklabels.copy()
        
        np.random.shuffle(kkklabels)
        
        nulo = silhouette_score(X_norm, kkklabels)
        
        print("Score modelo nulo: ", nulo)
             
    if metodo == 'kmedoidsfastPAM':
        for k in range(2, nro_clusters):
            diss = euclidean_distances(X)
            fp = kmedoids.fasterpam(diss, nro_clusters)
            score = silhouette_score(X, fp.labels)
            silhouette_coefficients.append(score)
            
    if metodo == 'kmedoids':
        for k in range(2, nro_clusters):
             kmedoids_ = KMedoids(n_clusters=k, metric = "cosine")
             kmedoids_.fit(X)
             score = silhouette_score(X, kmedoids_.labels_)
             silhouette_coefficients.append(score)
    
         
    fig, ax = plt.subplots(figsize = (24, 7))
    
    # estas lineas son el grafico de SSEvsK
    ax.scatter(range(2, nro_clusters), silhouette_coefficients)            
    ax.set_xticks(range(2, nro_clusters))
    ax.set_xlabel("Número de clusters", fontsize = 20)
    ax.set_ylabel("Promedio coeficientes de Silhouette", fontsize = 20)

    
    ax.tick_params(axis='x', labelsize=15)  
    ax.tick_params(axis='y', labelsize=15)
    
    if save != None:
        path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Graficos/Cluster'                                                  
        plt.savefig(path_imagenes + f'/silhouette_6PCA_{save}.png') 
        
    return 'ok'

def dendograma(X, metodo, metrica):
    dict_method = {'single': 'min', 'complete': 'max', "average": "promedio", "ward": "ward", "centroid": "centroid"}
    
    plt.figure(figsize=(10, 7))
    plt.title("Dendograma", fontsize = 20)
    plt.ylabel(f"Distancia {dict_method[metodo]}", fontsize = 20)
    
    # Con la función 'dendogram' graficamos el dendograma. 
    dend = shc.dendrogram(shc.linkage(X, method=metodo, metric = metrica))
    
    return dend
        
def cluters_jerarquico(X, nro_clusters, metodo, metrica, save = None):
    dict_method = {'single': 'min', 'complete': 'max', "average": "promedio", "ward": "ward", "centroid": "centroid"}
    
    # Creamos el modelo
    cluster = AgglomerativeClustering(n_clusters = nro_clusters, affinity = metrica, linkage = metodo)
    
    # Lo ajustamos con los datos
    cluster.fit_predict(X)  # fit_predict hace lo mismo que fit pero devuelve el vector de etiquetas de las samples
    
    
    labels = cluster.labels_ 
    
    centroids_jerarquico = []
    for cluster_label in np.unique(labels):
        cluster_points = X_pca[labels == cluster_label]
        centroid_jerarquico = cluster_points.mean(axis=0)
        centroids_jerarquico.append(centroid_jerarquico)
        
    centroids_jerarquico = np.array(centroids_jerarquico)
        
    return labels, centroids_jerarquico, dict_method[metodo]

def perfil_silhouette(X, cluster_labels, save = None):
    
    n_clusters = len(np.unique(cluster_labels))
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label = "Average")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    if save != None:
        path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Graficos/Cluster'                                                
        plt.savefig(path_imagenes + f'/perfilsilhouette_{save}.png')   
    
    return 'ok'



#%% primero hago PCA a los datos, ya vi que me tengo que quedar con 6 componentes para tener un 70% de la varianza

X = df.to_numpy()

print('Dimensiones de la matriz de features: {}'.format(X.shape))

# Ajustamos el estandarizador
std_scale.fit(X)

# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)

# Creación del modelo. Si el número de componentes no se específica, 
# se obtienen tantas componentes principales como features en nuestro dataset.
nro_PCA = nro_pca
pca = PCA(n_components=nro_PCA) #si queremos solo 5 poner n_components = 6

# Ajustamos el modelo a los datos escaleados
pca.fit(X_scaled)

# Obtenemos la descripción de los datos en el espacio de componentes principales
X_pca = pca.transform(X_scaled)


#%% TSNE 1

n_components=3
perplexity=8.0 #the number of nearest neighbors that is used in other manifold learning algorithms. Different values can result in significantly different results
early_exaggeration=10.0
learning_rate=240
n_iter=10000


tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = 44)
#si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
X_TSNE1 = tsne.fit_transform(X)


learning_rate=477

tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = 44)
#si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
X_TSNE2 = tsne.fit_transform(X)

#%% para kmeans con conseno hay que normalizar X

length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length

length_T = np.sqrt((X_TSNE1**2).sum(axis=1))[:,None]
X_TSNE1_norm = X_TSNE1 / length_T

length_T2 = np.sqrt((X_TSNE2**2).sum(axis=1))[:,None]
X_TSNE2_norm = X_TSNE2 / length_T2

#%% kmeans5

# Creación del modelo KMeans con k = 5
kmeans5 = KMeans(n_clusters=2, init = "random",  n_init = 10000, random_state = 42)

# Ajuste del modelo a los datos reducidos en componentes principales
#PCA
kmeans5.fit(X_pca_norm)
R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
print("Indice R con kmeans 2 y PCA: ", R_index5mean)
etiquetas(kmeans5.labels_)

#%%
#TSNE1
kmeans5.fit(X_TSNE1_norm)
R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
print("Indice R con kmeans 5 y TSNE1: ", R_index5mean)
etiquetas(kmeans5.labels_)
#TSNE2
kmeans5.fit(X_TSNE2_norm)
R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
print("Indice R con kmeans 5 y TSNE2: ", R_index5mean)
etiquetas(kmeans5.labels_)


#%% kmedoids5 con PAM y faster PAM abandono esto aca porque ya estoy probando demasiadas cosas y me mareo

diss = euclidean_distances(X_pca)
fp = kmedoids.fasterpam(diss, 5)
print("Loss with FasterPAM:", fp.loss)
pam = kmedoids.pam(diss, 100)
print("Loss with PAM:", pam.loss)

# grafico
markerstemas_colorcluster(fp.labels, X_pca, temas)  

# etiquetas
'''
devuelve algo de la pinta (array([0, 1, 2, 3]), array([19,  8,  1,  2], dtype=int64))
lo que quiere decir que en los primeros 30 datos (que corresponden a los 30 relatos de campeones) 19 fueron asignados al
cluster 0, 8 al cluster 1, 1 al cluster 2 y 2 al cluster 3
'''
etiquetas(fp.labels)
#%% kmedoids con Kmedoids de sklear_extra

kmedoids5 = KMedoids(n_clusters=2, metric = "cosine")
kmedoids5.fit(X_pca)

# Guardo las posiciones de los centroids
centroids = kmedoids5.cluster_centers_

#markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas, centroids = centroids)  
R_index5 = adjusted_rand_score(condicion_labels, kmedoids5.labels_) 

print("Indice R con kmedoids 2 y PCA: ", R_index5)
#etiquetas
etiquetas(kmedoids5.labels_)


#%%
kmedoids5.fit(X_TSNE1)
R_index5 = adjusted_rand_score(condicion_labels, kmedoids5.labels_) 
print("Indice R con kmedoids 5 y TSNE1: ", R_index5)
etiquetas(kmedoids5.labels_)
kmedoids5.fit(X_TSNE2)
R_index5 = adjusted_rand_score(condicion_labels, kmedoids5.labels_) 
print("Indice R con kmedoids 5 y TSNE2: ", R_index5)
etiquetas(kmedoids5.labels_)
#%% DBScan da horrible... Buscar un poco mas cómo jugar con los parámetros, pero me muevo de aca y da o cero o por ej subiendo min_samples da mejor pero solo da dos clusters...

X_ = [X_pca, X_TSNE1, X_TSNE2]
nombre = ["PCA", "TNSE1", "TSNE2"]

for j, data in enumerate(X_):
    clustering = DBSCAN(eps=0.11111, metric = "cosine", algorithm= "auto", min_samples=5).fit(data)
    # Eliminar las etiquetas predichas que son -1 y los valores correspondientes en etiquetas reales
    etiquetas_predichas = clustering.labels_
    etiquetas_reales = condicion_labels
    etiquetas_filtradas_reales = [etiquetas_reales[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
    etiquetas_filtradas_predichas = [etiquetas_predichas[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
    R_DB = adjusted_rand_score(etiquetas_filtradas_reales, etiquetas_filtradas_predichas)
    ruido = len(etiquetas_reales) - len(etiquetas_filtradas_reales)
    ruido2 = sum(1 for etiqueta in etiquetas_predichas if etiqueta == -1)
    
    print(np.unique(clustering.labels_))
    print(f"Indice R con DBscan y {nombre[j]}: ", R_DB)
    print(ruido)
    print(ruido2)
    
    etiquetas(clustering.labels_)

    markerstemas_colorcluster(clustering.labels_, data, temas, centroids = None)
    
#%%

data = X_TSNE1

clustering = DBSCAN(eps=3.5167267267267266, metric = "euclidean", algorithm= "auto", min_samples=8).fit(data)
# Eliminar las etiquetas predichas que son -1 y los valores correspondientes en etiquetas reales
etiquetas_predichas = clustering.labels_
etiquetas_reales = condicion_labels
etiquetas_filtradas_reales = [etiquetas_reales[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
etiquetas_filtradas_predichas = [etiquetas_predichas[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
R_DB = adjusted_rand_score(etiquetas_filtradas_reales, etiquetas_filtradas_predichas)
ruido = len(etiquetas_reales) - len(etiquetas_filtradas_reales)
ruido2 = sum(1 for etiqueta in etiquetas_predichas if etiqueta == -1)

print(np.unique(clustering.labels_))
print(f"Indice R con DBscan y {nombre[j]}: ", R_DB)
print(ruido)
print(ruido2)

etiquetas(clustering.labels_)

markerstemas_colorcluster(clustering.labels_, data, temas, centroids = None)

#%% barrido cosine

R = []
ruido = []
e = []
samp = []
clus = []
for eps in tqdm(np.linspace(0.00001, 0.5, 10000)): #(0.001, 0.3, 1000)
    for min_samples in np.linspace(2, 20, 19):
        clustering = DBSCAN(eps=eps, metric = "cosine", algorithm= "brute", min_samples=min_samples).fit(X_TSNE1)
        # Eliminar las etiquetas predichas que son -1 y los valores correspondientes en etiquetas reales
        etiquetas_predichas = clustering.labels_
        etiquetas_reales = condicion_labels
        etiquetas_filtradas_reales = [etiquetas_reales[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
        etiquetas_filtradas_predichas = [etiquetas_predichas[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
        R_DB = adjusted_rand_score(etiquetas_filtradas_reales, etiquetas_filtradas_predichas)
        #ruido = len(etiquetas_reales) - len(etiquetas_filtradas_reales)
        ruido2 = sum(1 for etiqueta in etiquetas_predichas if etiqueta == -1)
        if len(np.unique(clustering.labels_)) >= 3:
            if R_DB >= 0.23:
                #if ruido2 <= len(etiquetas_reales)-150: #quiero que al menos haya casi 50% de datos clasificados
                #print(np.unique(clustering.labels_), R_DB, ruido2)
                R.append(R_DB)
                ruido.append(ruido2)
                e.append(eps)
                samp.append(min_samples)
                clus.append(clustering.labels_)
#%%
print(len(R))

#plt.plot(np.linspace(0, len(R), len(R)), R, 'o')
#plt.plot(np.linspace(0, len(R), len(R)), ruido, 'o')

fig, ax1 = plt.subplots()

# Plotear en el primer eje y
ax1.plot(np.linspace(0, len(R), len(R)), R, 'o', label='R', color='b')
ax1.set_xlabel('Índice')
ax1.set_ylabel('R', color='b')
ax1.tick_params('y', colors='b')

# Crear el segundo eje y
ax2 = ax1.twinx()
ax2.plot(np.linspace(0, len(ruido), len(ruido)), np.full(len(ruido), 321) - np.asarray(ruido), 'o', label='Clasificados', color='r')
ax2.set_ylabel('Clasificados', color='r')
ax2.tick_params('y', colors='r')

# Añadir leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title("Distancia cosine algoritmo auto")

plt.show()
#%% euclidean
R_e = []
ruido_e = []
e_e = []
samp_e = []
clus_e = []
for eps in tqdm(np.linspace(0.01, 4, 1000)):
    for min_samples in np.linspace(2, 15, 14):
        clustering = DBSCAN(eps=eps, metric = "euclidean", algorithm= "auto", min_samples=min_samples).fit(X_TSNE1)
        # Eliminar las etiquetas predichas que son -1 y los valores correspondientes en etiquetas reales
        etiquetas_predichas = clustering.labels_
        etiquetas_reales = condicion_labels
        etiquetas_filtradas_reales = [etiquetas_reales[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
        etiquetas_filtradas_predichas = [etiquetas_predichas[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
        R_DB = adjusted_rand_score(etiquetas_filtradas_reales, etiquetas_filtradas_predichas)
        #ruido = len(etiquetas_reales) - len(etiquetas_filtradas_reales)
        ruido2 = sum(1 for etiqueta in etiquetas_predichas if etiqueta == -1)
        if len(np.unique(clustering.labels_)) >= 3:
            if R_DB >= 0.05:
                #print(np.unique(clustering.labels_), R_DB, ruido2)
                R_e.append(R_DB)
                ruido_e.append(ruido2)
                e_e.append(eps)
                samp_e.append(min_samples)
                clus_e.append(clustering.labels_)
#%%
print(len(R_e))

#plt.plot(np.linspace(0, len(R), len(R)), R, 'o')
#plt.plot(np.linspace(0, len(R), len(R)), ruido, 'o')

fig, ax1 = plt.subplots()

# Plotear en el primer eje y
ax1.plot(np.linspace(0, len(R_e), len(R_e)), R_e, 'o', label='R', color='b')
ax1.set_xlabel('Índice')
ax1.set_ylabel('R', color='b')
ax1.tick_params('y', colors='b')

# Crear el segundo eje y
ax2 = ax1.twinx()
ax2.plot(np.linspace(0, len(ruido_e), len(ruido_e)), np.full(len(ruido_e), 321) - np.asarray(ruido_e), 'o', label='Clasificados', color='r')
ax2.set_ylabel('Clasificados', color='r')
ax2.tick_params('y', colors='r')

# Añadir leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Distancia euclidea algorítmo auto")
plt.show()

#%%
clustering = DBSCAN(eps=2.1, metric = "euclidean", algorithm= "auto", min_samples=3).fit(X_pca)
etiquetas_predichas = clustering.labels_
etiquetas_reales = condicion_labels
etiquetas_filtradas_reales = [etiquetas_reales[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
etiquetas_filtradas_predichas = [etiquetas_predichas[i] for i in range(len(etiquetas_predichas)) if etiquetas_predichas[i] != -1]
R_DB = adjusted_rand_score(etiquetas_filtradas_reales, etiquetas_filtradas_predichas)
ruido = len(etiquetas_reales) - len(etiquetas_filtradas_reales)
ruido2 = sum(1 for etiqueta in etiquetas_predichas if etiqueta == -1)
print(np.unique(clustering.labels_))
print(R_DB)  
print(ruido)
print(ruido2)
#%% kmeans2
# Creación del modelo KMeans con k = 5
kmeans2 = KMeans(n_clusters=2, init = "random",  n_init = 1000)

# Ajuste del modelo a los datos reducidos en componentes principales
kmeans2.fit(X_pca_norm)
# etiquetas
'''
Para acceder a las etiquetas que le asignó el modelo a cada sample usamos 'kmeans.labels_'
'''
kmeans = kmeans2
# Nos fijamos las etiquetas asignadas

etiquetas(kmeans.labels_)

# cluster en color, temas en forma
'''
Para acceder a la posición de los centroids en el espacio de 6 (o 30) PCs usamos 'kmeans.cluster_centers_ 
'''

# Guardo las posiciones de los centroids
centroids = kmeans.cluster_centers_

# Printeo las dimensiones de las posiciones
print("Shape de los centroids:",centroids.shape)
# Printeo las posiciones de las primeras 5 muestras en sus primeras dos componentes principales
print(centroids[:5,[0,1]])
   

markerstemas_colorcluster(kmeans.labels_, X_pca, temas, centroids = centroids)  

# cluster en formas, temas en color

# Guardo las posiciones de los centroids
centroids = kmeans.cluster_centers_

markerscluster_colortemas(kmeans.labels_, X_pca, temas, centroids = centroids)

#%% eleccion de k
metodo_codo(X_pca, 'kmeans', 15, save = None)
metodo_silhouette(X_pca, 'kmeans', 15)
clusterer = KMeans(n_clusters=5, random_state=10)
cluster_labels = clusterer.fit_predict(X_pca_norm)

perfil_silhouette(X_pca_norm, cluster_labels)

metodo_codo(X_pca, 'kmedoids', 15, save = None)
metodo_silhouette(X_pca, 'kmedoids', 15)
clusterer = KMedoids(n_clusters=2, metric = "cosine", random_state=1)
cluster_labels = clusterer.fit_predict(X_pca)

perfil_silhouette(X_pca, cluster_labels)



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
distancia = ["single", "complete", "average", "ward"] #"centroid",
distancia_label = ["min", "max", "promedio", "ward"] # "centroid",

for i, dist in enumerate(distancia):
    plt.figure(figsize=(10, 7))
    plt.title("Dendograma", fontsize = 20)
    plt.ylabel(f"Distancia {distancia_label[i]}", fontsize = 20)
    
    # Con la función 'dendogram' graficamos el dendograma. 
    dend = shc.dendrogram(shc.linkage(X_pca, method=dist, metric = 'cosine'))  # El input de esta función es la función 'linkage' 
                                                    #donde se especifica la distancia para utlizar en cada paso del método
                                                    
    #path_imagenes = path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'                                                
    #plt.savefig(path_imagenes + f'/{nro_PCA}PCA_dendograma_{distancia_label[i]}.png')                                               
'''
single: min
complete: max
centroid: centroid
average: average
ward: ward
'''
#%%
distancias_con_cosine = ["single", "complete", "average"] #"centroid", "ward"
distancias_sin_cosine = ["ward", "centroid"]

for i, dist in enumerate(distancias_con_cosine):
    dendograma(X_pca, dist, 'cosine')
    
length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length
for i, dist in enumerate(distancias_sin_cosine):
    dendograma(X_pca_norm, dist, 'euclidean')
                                              
'''
definir k
Lo que buscamos en el dendograma es la mayor distancia vertical sin que haya una línea horizontal para hacerle un corte 
(representado como una linea horizontal que cruza todos los datos) y quedarnos con k clusters (donde k es el número de 
lineas verticales que intersectan el corte. 
'''

#%% clusterización jerárquica con k óptimo
'''
Ahora sí aplicamos el método de clusterización jerárquica (bottom-up) con 5 clusters, la distancia euclidea para la afinidad y la distancia ward para el linkage
'''

k = 3



for i, dist in enumerate(distancias_con_cosine[1:]):
    labels_j, centroids_j, metodo = cluters_jerarquico(X_pca, k, dist, 'cosine')
    #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
    #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
    print(f"Indice R usando {dist}, k = {k}", adjusted_rand_score(condicion_labels, labels_j))
    etiquetas(labels_j)
    
    
    
length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length
for i, dist in enumerate(distancias_sin_cosine[:1]):
    labels_j, centroids_j, metodo = cluters_jerarquico(X_pca_norm, k, dist, 'euclidean')
    #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
    #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
    print(f"Indice R usando {dist}, k = {k}", adjusted_rand_score(condicion_labels, labels_j))
    etiquetas(labels_j)
    


#%% perfil silhouette  

labels_j, centroids_j, metodo = cluters_jerarquico(X_pca, 5, 'complete', 'cosine')

perfil_silhouette(X_pca, labels_j)
