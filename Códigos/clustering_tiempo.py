# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:40:41 2023

@author: corir
"""

#%% librerias
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from tqdm import tqdm
import matplotlib as npl

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

#%% FUNCIONES

def PCA_estandarizando(data, n_components = None, graph_var = True, max_var = 0.7, graph_PCs = True, n_graph_PCs = 7):
    '''
    le das el dataframe al que queres que le haga PCs 
    estandariza y te da las PCs, le podes pedir el gráfico de la varianza y el de las
    primeras n PCs
    '''

    X = data.to_numpy()
    
    # Ajustamos el estandarizador
    std_scale.fit(X)

    # Aplicamos el estandarizador y obtenemos la matriz de features escaleados
    X_scaled = std_scale.transform(X)

    # Creación del modelo. Si el número de componentes no se específica, 
    # se obtienen tantas componentes principales como features en nuestro dataset.
    pca = PCA(n_components=n_components)

    # Ajustamos el modelo a los datos escaleados
    pca.fit(X_scaled)

    # Obtenemos la descripción de los datos en el espacio de componentes principales
    X_pca = pca.transform(X_scaled)

    evr = pca.explained_variance_ratio_
    
    if graph_var == True:
        # con .explained_variance_ratio_ vemos la fracción de información que aporta cada componente
    
        fig, ax = plt.subplots(1, 1, figsize = (18, 10))
    
        # Calculamos el acumulado con la función cumsum de numpy 
        varianza_acumulada = np.cumsum(evr)
        max_comp = np.where(varianza_acumulada > max_var)[0][0] + 1
    
        ax.plot(range(1, len(evr) + 1), varianza_acumulada, '.-', markersize = 20, color = color_celeste, zorder = 5)
        ax.set_ylabel('Fracción acumulada de varianza explicada')
        ax.set_xlabel('Cantidad de componentes principales')
        ax.axhline(y=max_var, color=color_gris, linestyle='--', linewidth = 4, label=f'{max_var*100}%')
        ax.axvline(x = max_comp, color=color_gris, linestyle='--', linewidth = 4)
        ax.grid(True)
        plt.legend()

    if graph_PCs == True:

        npl.rcParams["axes.labelsize"] = 20
        npl.rcParams['xtick.labelsize'] = 20
        npl.rcParams['ytick.labelsize'] = 20


        variables = list(data.columns)
        
        componentes_principales = [pca.components_[i] for i in range(0, n_graph_PCs)]
    
        # Crea un diccionario con las componentes principales y las variables
        data_pcs = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}
    
        # Crea el DataFrame
        df_vars_1 = pd.DataFrame(data_pcs, index=variables)
    
        #df_vars_1.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/primeras_6_componentes.csv')
    
        center_value= 0
        plt.figure(figsize = (30, len(variables)))
        sns.heatmap(df_vars_1, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, linecolor="black", center = center_value) #cbar_kws={"shrink": 0.75}) #"YlGnBu,  annot=True
    
    
        plt.yticks(rotation=0) #plt.yticks(variables)
        plt.xticks(rotation=0)
        #plt.xlabel("Componentes Principales")
    
    
        # Muestra el gráfico
        plt.show()
    return X_pca, pca, evr


def R_clausterizacion(data, k, condicion_labels, indices, kmeans = False, etiquetas_print = False, graph_clusters = False, temas_graph_clusters = None):
    '''
    le das la data ya pasada por PCA o por donde quieras, los labels de las condiciones y los indices
    eso lo conseguis con indices_condiciones.
    Busca clusters con kmedoids y jerarquico
    si queres que lo haga con kmeans tambien va a tardar mas pero pone kmeans = True o lo q sea
    devuelve una lista con los R en este orden: kmedoids, jerarquico max, jerarquico average
    jerarquico ward, si agregas kmeans va a ser el primer
    si queres ver las etiquetas de los clusters poner etiquetas = True
    '''
    #para usar distancia coseno aunque el método de clausterización no lo de como opción,
    #es solo hacer una transformación de los datos
    length = np.sqrt((data**2).sum(axis=1))[:,None]
    data_norm = data / length
    data_jer = data
    
    R = []
    if kmeans != False:
        #Creación del modelo KMeans 
        kmeans5 = KMeans(n_clusters=k, init = "random",  n_init = 10000, random_state = 42)
        
        #Ajuste del modelo a los datos reducidos en componentes principales PCA
        kmeans5.fit(data_norm)
        centroids_kmeans = kmeans5.cluster_centers_
        R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
        if etiquetas_print != False:
            print(f"Indice R con kmeans {k} y PCA: ", R_index5mean)
            etiquetas(kmeans5.labels_, indices)
            R.append(R_index5mean)
    
    # kmedoids con Kmedoids de sklear_extra
    kmedoids5 = KMedoids(n_clusters=k, metric = "cosine")
    kmedoids5.fit(data)
    
    # Guardo las posiciones de los centroids
    centroids_kmedoids = kmedoids5.cluster_centers_
    
    #markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas, centroids = centroids_kmedoids)  
    R_index5 = adjusted_rand_score(condicion_labels, kmedoids5.labels_) 
    R.append(R_index5)
    if etiquetas_print != False:
        print(f"Indice R con kmedoids {k} y PCA: ", R_index5)
        #etiquetas
        etiquetas(kmedoids5.labels_, indices)
    
    # clusterización jerárquica con k óptimo
    '''
    Aplicamos el método de clusterización jerárquica (bottom-up) con 5 clusters, la distancia euclidea para la afinidad y la distancia ward para el linkage
    '''
    
    distancias_con_cosine = ["single", "complete", "average"] #"centroid", "ward"
    distancias_sin_cosine = ["ward", "centroid"]
    
    
    labels_jerarquico = []
    centroids_jerarquico = []
    for i, dist in enumerate(distancias_con_cosine[1:]):
        labels_j, centroids_j, metodo = cluters_jerarquico(data_jer, k, dist, 'cosine')
        #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
        #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
        R_jer = adjusted_rand_score(condicion_labels, labels_j)
        labels_jerarquico.append(labels_j)
        centroids_jerarquico.append(centroids_j)
        if etiquetas_print != False:
            print(f"Indice R usando {dist}, k = {k}", R_jer)
            etiquetas(labels_j, indices)
        R.append(R_jer)
        
        
    for i, dist in enumerate(distancias_sin_cosine[:1]):
        labels_j, centroids_j, metodo = cluters_jerarquico(data_norm, k, dist, 'euclidean')
        #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
        #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
        R_jer = adjusted_rand_score(condicion_labels, labels_j)
        labels_jerarquico.append(labels_j)
        centroids_jerarquico.append(centroids_j)
        if etiquetas_print != False:
            print(f"Indice R usando {dist}, k = {k}", R_jer)
            etiquetas(labels_j, indices)
        R.append(R_jer)
        
    r_max_pos = np.where(np.asarray(R) == max(R))[0][0]
    
    if graph_clusters != False:
        if kmeans != False:
            if r_max_pos == 0:
                markerstemas_colorcluster(kmeans5.labels_, X_pca, temas_graph_clusters, indices = indices, centroids = centroids_kmeans)
            if r_max_pos == 1:
                markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas_graph_clusters, indices = indices, centroids = centroids_kmedoids)  
            if r_max_pos > 1:
                markerstemas_colorcluster(labels_jerarquico[r_max_pos - 2], X_pca, temas_graph_clusters, indices = indices, centroids =  centroids_jerarquico[r_max_pos - 2])
        else:
            if r_max_pos == 0:
                markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas_graph_clusters, indices = indices, centroids = centroids_kmedoids)  
            if r_max_pos > 0:
                markerstemas_colorcluster(labels_jerarquico[r_max_pos - 1], X_pca, temas_graph_clusters, indices = indices, centroids =  centroids_jerarquico[r_max_pos - 1])
    return R
    
    
    
def markerstemas_colorcluster(labels, X, temaslabels, indices, save = None, centroids =  None, title =  None):

    fig, ax = plt.subplots(figsize = (20, 7))
    
    ind_camp_t1 = indices[0]
    ind_camp_t2 = indices[1]
    ind_pres_t1 = indices[2]
    ind_pres_t2 = indices[3]
    ind_cfk_t1 = indices[4]
    ind_cfk_t2 = indices[5]
    ind_ar_t1 = indices[6]
    ind_ar_t2 = indices[7]
    ind_fil_t1 = indices[8]
    ind_fil_t2 = indices[9]
    
    
    indices1 = [ind_cfk_t1, ind_camp_t1, ind_fil_t1, ind_pres_t1, ind_ar_t1]
    indices2 = [ind_cfk_t2, ind_camp_t2, ind_fil_t2, ind_pres_t2, ind_ar_t2]
    
    scatter1 = ["o", "v", "s", "*", "d"]
    scatter2 = ["^", "p", "H", "+", "x"]
    
    # Hacemos un scatter plot de cada uno de los datos
    for i, ind in enumerate(indices1):
        if ind != (None, None):
            ax.scatter(X[ind[0][0]:ind[0][1], 0], X[ind[0][0]:ind[0][1], 1], marker = scatter1[i], c=labels[ind[0][0]:ind[0][1]], label = temaslabels[i] + "t1")
    for i, ind in enumerate(indices2):
        if ind != (None, None):
            ax.scatter(X[ind[0][0]:ind[0][1], 0], X[ind[0][0]:ind[0][1], 1], marker = scatter1[i], c=labels[ind[0][0]:ind[0][1]], label = temaslabels[i] + "t2")

    
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

def markerscluster_colortemas(labels, X, temaslabels, indices, save = None, centroids =  None, title =  None):
    fig, ax = plt.subplots(figsize=(20, 7))

    ind_camp = indices[0]
    ind_pres = indices[1]
    ind_cfk = indices[2]
    ind_ar = indices[3]
    ind_fil = indices[4]
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

def encontrar_posiciones_tupla(lista_tuplas, tupla_objetivo):
    inicio = None
    fin = None

    for i, tupla in enumerate(lista_tuplas):
        if tupla == tupla_objetivo:
            if inicio is None:
                inicio = i
            fin = i

    return inicio, fin


def indices_condiciones(path, condiciones = None, tiempo = None, drop = None):
    '''
    tenes que darle el path de donde estas poniendo tus condiciones
    y en condiciones pones una lista con los números de las condiciones que queres que tire
    en tiempo podes poner el tiempo que queres que tire [1] [2]
    '''
    df = pd.read_csv(path)
     
    df = df.dropna()
    
    mapping = {
        'antesdevenir': 5,
        'arabia': 4,
        'campeones_del_mundo': 1,
        'presencial': 2,
        'cfk': 3, #3
    }

    # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
    df['Condición'] = df['Condición'].map(mapping)

    if condiciones != None:
        
        df = df[~df['Condición'].isin(condiciones)]
        
    if tiempo != None:
        
        df = df[~df['Tiempo'].isin(tiempo)]

        
    if drop != None:
        
        df = df.drop(drop, axis = 1)

    condicion_labels = list(df['Condición'])
    
    tiempo_labels = list(df['Tiempo'])
    
    condicion_y_tiempo_labels = list(zip(condicion_labels, tiempo_labels))
    
    condicion_y_tiempo_labels1D = [int(str(tupla[0]) + str(tupla[1])) for tupla in condicion_y_tiempo_labels]


    indices = []
    ind_camp_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (1,1))
    indices.append(ind_camp_t1)
    ind_camp_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (1,2))
    indices.append(ind_camp_t2)
        
    ind_pres_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (2,1))
    indices.append(ind_pres_t1)

    ind_pres_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (2,2))
    indices.append(ind_pres_t2)

    ind_cfk_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (3,1))
    indices.append(ind_cfk_t1)
    ind_cfk_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (3,2))
    indices.append(ind_cfk_t2)

    ind_ar_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (4,1))
    indices.append(ind_ar_t1)
    ind_ar_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (4,2))
    indices.append(ind_ar_t2)


    ind_fil_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (5,1))
    indices.append(ind_fil_t1)
    ind_fil_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (5,2))
    indices.append(ind_fil_t2)
        
    return condicion_y_tiempo_labels1D, indices 

# def indices_condiciones_tiempo(path, condiciones = None, tiempo = None, drop = None):
#     '''
#     tenes que darle el path de donde estas poniendo tus condiciones
#     y en condiciones pones una lista con los números de las condiciones que queres que tire
#     en tiempo podes poner el tiempo que queres que tire [1] [2]
#     '''
#     df = pd.read_csv(path)
     
#     df = df.dropna()
    
#     mapping = {
#         'antesdevenir': 5,
#         'arabia': 4,
#         'campeones_del_mundo': 1,
#         'presencial': 2,
#         'cfk': 3, #3
#     }

#     # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
#     df['Condición'] = df['Condición'].map(mapping)

#     if condiciones != None:
        
#         df = df[~df['Condición'].isin(condiciones)]
        
#     if tiempo != None:
        
#         df = df[~df['Tiempo'].isin(tiempo)]

        
#     if drop != None:
        
#         df = df.drop(drop, axis = 1)

#     condicion_labels = list(df['Condición'])
    
#     tiempo_labels = list(df['Tiempo'])
    
#     condicion_y_tiempo_labels = list(zip(condicion_labels, tiempo_labels))

#     indices = []
#     if 1 not in condiciones:
#         if 1 not in tiempo:
#             ind_camp_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (1,1))
#             indices.append(ind_camp_t1)
#         if 2 not in tiempo:
#             ind_camp_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (1,2))
#             indices.append(ind_camp_t2)
        
#     if 2 not in condiciones:
#         if 1 not in tiempo:
#             ind_pres_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (2,1))
#             indices.append(ind_pres_t1)
#         if 2 not in tiempo:
#             ind_pres_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (2,2))
#             indices.append(ind_pres_t2)
        
#     if 3 not in condiciones:
#         if 1 not in tiempo:
#             ind_cfk_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (3,1))
#             indices.append(ind_cfk_t1)
#         if 2 not in tiempo:
#             ind_cfk_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (3,2))
#             indices.append(ind_cfk_t2)
        
#     if 4 not in condiciones:
#         if 1 not in tiempo:
#             ind_ar_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (4,1))
#             indices.append(ind_ar_t1)
#         if 2 not in tiempo:
#             ind_ar_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (4,2))
#             indices.append(ind_ar_t2)
        
#     if 5 not in condiciones:
#         if 1 not in tiempo:
#             ind_fil_t1 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (5,1))
#             indices.append(ind_fil_t1)
#         if 2 not in tiempo:
#             ind_fil_t2 = encontrar_posiciones_tupla(condicion_y_tiempo_labels, (5,2))
#             indices.append(ind_fil_t2)
        
#     return condicion_y_tiempo_labels, indices 

def etiquetas(labels, indices):
    #devuelve algo de la pinta
    '''
    indices es una lista de los indices primero esta el de campeones, después presencial, después
    cfk arabia y filler. Si los haces con la funcion indices_condiciones ya lo devuelve asi
    devuelve algo de la pinta (array([0, 1, 2, 3]), array([19,  8,  1,  2], dtype=int64))
    lo que quiere decir que en los primeros 30 datos (que corresponden a los 30 relatos de campeones) 19 fueron asignados al
    cluster 0, 8 al cluster 1, 1 al cluster 2 y 2 al cluster 3
    '''
    ind_camp_t1 = indices[0]
    ind_camp_t2 = indices[1]
    ind_pres_t1 = indices[2]
    ind_pres_t2 = indices[3]
    ind_cfk_t1 = indices[4]
    ind_cfk_t2 = indices[5]
    ind_ar_t1 = indices[6]
    ind_ar_t2 = indices[7]
    ind_fil_t1 = indices[8]
    ind_fil_t2 = indices[9]
    
    
    if ind_camp_t1 != (None, None):
        print("etiquetas campeones t1:", np.unique(labels[ind_camp_t1[0]:ind_camp_t1[1]], return_counts=True))
    if ind_camp_t2 != (None, None):
        print("etiquetas campeones t2:", np.unique(labels[ind_camp_t2[0]:ind_camp_t2[1]], return_counts=True))

    #en presencial
    if ind_pres_t1 != (None, None):
        print("etiquetas presencial t1:", np.unique(labels[ind_pres_t1[0]:ind_pres_t1[1]], return_counts=True))
    if ind_pres_t2 != (None, None):
        print("etiquetas presencial t2:", np.unique(labels[ind_pres_t2[0]:ind_pres_t2[1]], return_counts=True))

    #cfk
    if ind_cfk_t1 != (None, None):
        print("etiquetas cfk t1:", np.unique(labels[ind_cfk_t1[0]:ind_cfk_t1[1]], return_counts=True))
    if ind_cfk_t2 != (None, None):
        print("etiquetas cfk t2:", np.unique(labels[ind_cfk_t2[0]:ind_cfk_t2[1]], return_counts=True))
    #arabia
    if ind_ar_t1 != (None, None):
        print("etiquetas arabia t1:", np.unique(labels[ind_ar_t1[0]:ind_ar_t1[1]], return_counts=True))
    if ind_ar_t2 != (None, None):
        print("etiquetas arabia t2:", np.unique(labels[ind_ar_t2[0]:ind_ar_t2[1]], return_counts=True))
    #antes de venir
    if ind_fil_t1 != (None, None):
        print("etiquetas filler t1:", np.unique(labels[ind_fil_t1[0]:ind_fil_t1[1]], return_counts=True))
    if ind_fil_t2 != (None, None):
        print("etiquetas filler t2", np.unique(labels[ind_fil_t2[0]:ind_fil_t2[1]], return_counts=True))
    
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

#%% el santo trial y colores

entrevista = 'dos tiempos' #'Primera' 'Segunda' o 'dos tiempos'

no_autop = True #pone false si queres que las tenga en cuenta para el análisis

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

color_celeste = "#79b4b7ff"
color_gris = "#9fa0a4ff"

eliminando_outliers = True

#%% path data 

if entrevista == 'dos tiempos':
    path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_dostiempos.csv'
    
    path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_dostiempos.csv'
else:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'
    
    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

if eliminando_outliers == True:
    if entrevista == 'dos tiempos':
        path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_dostiempos.csv'
        
        path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_dostiempos.csv'

    

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3 
}

eliminamos_pysent = ['Valencia pysent', 'Valencia e intensidad pysent']#, 'Valencia2 pysent', 'Valencia e intensidad2 pysent']


#%% con las variables de anova significativas (variables crudas, NO LAS PCs)
#valencia2 pysent 12 PCs
vars_no_imp = ['primera_persona_norm', 'transitivity', 'num adj norm']

df = pd.read_csv(path_conautopercepcion_todas)

if no_autop == True:
    df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 

df = df.drop(eliminamos_pysent, axis=1)

variables_dependientes = list(df.columns)[3:] #sacamos sujeto condicion y tiempo

#campeones = 1
#vars_anova_sig = ['num propn norm', 'Comunidades_LSC', 'k_mean', 'ASP', 'density'] 
#presencial = 2
#vars_anova_sig = ['Negativo pysent', 'num_nodes_LSC', 'Comunidades_LSC', 'k_mean', 'ASP', 'average_CC', 'L2']
#cfk = 3
#vars_anova_sig = ['num_nodes_LSC', 'Comunidades_LSC', 'k_mean', 'ASP', 'selfloops', 'density']
#arabia = 4
#vars_anova_sig = ['num_palabras_unicas_norm','primera_persona_norm', 'num_nodes_LSC', 'Comunidades_LSC', 'k_mean', 'ASP', 'average_CC', 'L2', 'L3', 'density']
#filler = 5
#vars_anova_sig = ['Valencia_autop', 'ValeInt_autop', 'num_palabras_unicas_norm', 'Negativo pysent', 'cohe_norm_d=2', 'cohe_norm_d=3', 'Detalles internos norm', 'Detalles externos norm']
#cfk ar y camp  1, 3 y 4
#vars_anova_sig = ['num_palabras_unicas_norm', 'num propn norm', 'num_nodes_LSC', 'Comunidades_LSC', 'diámetro', 'k_mean', 'ASP', 'average_CC', 'selfloops', 'density']
#pres y filler 5 y 2
#vars_anova_sig = ['Valencia_autop', 'ValeInt_autop', 'num_palabras_unicas_norm', 'Negativo pysent', 'Comunidades_LSC', 'k_mean', 'ASP', 'Detalles internos norm', 'Detalles externos norm']
#todass
#vars_anova_sig = ['Recuerdo_autop', 'num noun norm', 'num propn norm', 'Negativo pysent', 'cohe_norm_d=2', 'num_nodes_LSC', 'Comunidades_LSC', 'diámetro', 'k_mean', 'ASP', 'average_CC', 'selfloops', 'L2', 'density', 'Detalles internos norm', 'Detalles externos norm']

#vars_no_imp = [variable for variable in variables_dependientes if variable not in vars_anova_sig]

nro_pcs = 12

k = 2

cond_elim = [1,2,3,4]

valor_a_buscar = 5

nro_vars_dep = len(variables_dependientes)
nro_vars_elim = len(vars_no_imp)

pcs_a_recorrer = range(1,nro_vars_dep-nro_vars_elim+1)

R_pcs = []
for nro_pcs in pcs_a_recorrer:
    df = pd.read_csv(path_conautopercepcion_todas)
    
    #df = df.dropna()
    
    # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
    df['Condición'] = df['Condición'].map(mapping)
    
    df = df[~df['Condición'].isin(cond_elim)]
    
    condicion_labels, indices_ = indices_condiciones(path_conautopercepcion_todas, condiciones = cond_elim)
                   
    df = df.drop(['Sujetos', 'Condición', 'Tiempo'] + eliminamos_pysent, axis=1)
    
    if no_autop == True:
        df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 
        condicion_labels, indices_ = indices_condiciones(path_sinautopercepcion_todas, condiciones = cond_elim)
    
        
    df = df.dropna()
          
    df = df.drop(vars_no_imp, axis = 1)
    
    X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  nro_pcs, graph_var = False, graph_PCs = False)
    
    R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = False)
    
    for clave, valor in mapping.items():
        if valor == valor_a_buscar:
            clave_correspondiente = clave
            break
    R_pcs.append(max(R))

print(f"para {clave_correspondiente}", max(R_pcs))
indice_pcs = np.where(np.asarray(R_pcs) == max(R_pcs))[0][0]
mejor_pcs = pcs_a_recorrer[indice_pcs]
print(f"para {mejor_pcs} PCs")
X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  mejor_pcs, graph_var = False, graph_PCs = False)

R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = True,  graph_clusters= True, temas_graph_clusters = temas)

#%% con vars y PCs de los métodos para separar la primera entrevista
'''
esto hay que correrlo dsp de correr las PCs del método de primera entrevista, asi usa esa transformación
'''

#valencia2 pysent kmeans
nro_pcs_kmeans = 4 
vars_no_imp_kmeans = ['primera_persona_norm', 'num verb norm', 'cohe_norm_d=3', 'diámetro', 'transitivity', 'average_CC', 'selfloops']
#valencia 2 average jerarquico
nro_pcs_average = 12
vars_no_imp_average = ['primera_persona_norm', 'tercera_persona_norm', 'num noun norm', 'num verb norm', 'num numeral norm', 'num propn norm', 'Intensidad pysent', 'cohe_norm_d=3', 'selfloops']
#valencia 2 kmedoids
vars_no_imp_kmedoids = ['primera_persona_norm']
nro_psc_kmedoids = 13
k = 2

cond_elim = [2,3,4,5]

valor_a_buscar = 1


df = pd.read_csv(path_conautopercepcion_todas)

#df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin(cond_elim)]

condicion_labels, indices_ = indices_condiciones(path_conautopercepcion_todas, condiciones = cond_elim)
               
df = df.drop(['Sujetos', 'Condición', 'Tiempo'] + eliminamos_pysent, axis=1)

if no_autop == True:
    df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 
    condicion_labels, indices_ = indices_condiciones(path_sinautopercepcion_todas, condiciones = cond_elim)

    
df = df.dropna()
      
df = df.drop(vars_no_imp, axis = 1)

X = df.to_numpy()

# Ajustamos el estandarizador
std_scale.fit(X)

# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)
X_pca = pca1.transform(X_scaled)

R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = kmeans_TF, etiquetas_print = True)

for clave, valor in mapping.items():
    if valor == valor_a_buscar:
        clave_correspondiente = clave
        break

print(f"para {clave_correspondiente}", R)
