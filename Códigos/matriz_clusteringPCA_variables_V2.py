# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:41:48 2023

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

def R_clausterizacion(data, k, condicion_labels, indices, kmeans = False, etiquetas_print = False):
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
        R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
        R.append(R_index5mean)
        if etiquetas_print != False:
            print(f"Indice R con kmeans {k} y PCA: ", R_index5mean)
            etiquetas(kmeans5.labels_, indices)
            
    
    # kmedoids con Kmedoids de sklear_extra
    kmedoids5 = KMedoids(n_clusters=k, metric = "cosine")
    kmedoids5.fit(data)
    
    # Guardo las posiciones de los centroids
    #centroids = kmedoids5.cluster_centers_
    
    #markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas, centroids = centroids)  
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
    
    
    for i, dist in enumerate(distancias_con_cosine[1:]):
        labels_j, centroids_j, metodo = cluters_jerarquico(data_jer, k, dist, 'cosine')
        #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
        #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
        R_jer = adjusted_rand_score(condicion_labels, labels_j)
        if etiquetas_print != False:
            print(f"Indice R usando {dist}, k = {k}", R_jer)
            etiquetas(labels_j, indices)
        R.append(R_jer)
        
        
    for i, dist in enumerate(distancias_sin_cosine[:1]):
        labels_j, centroids_j, metodo = cluters_jerarquico(data_norm, k, dist, 'euclidean')
        #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
        #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
        R_jer = adjusted_rand_score(condicion_labels, labels_j)
        if etiquetas_print != False:
            print(f"Indice R usando {dist}, k = {k}", R_jer)
            etiquetas(labels_j, indices)
        R.append(R_jer)
        
    return R
    
    
    
def markerstemas_colorcluster(labels, X, temaslabels, indices, save = None, centroids =  None, title =  None):

    fig, ax = plt.subplots(figsize = (20, 7))
    
    ind_camp = indices[0]
    ind_pres = indices[1]
    ind_cfk = indices[2]
    ind_ar = indices[3]
    ind_fil = indices[4]
    
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

def indices_condiciones(path, condiciones = None, drop = None):
    '''
    tenes que darle el path de donde estas poniendo tus condiciones
    y en condiciones pones una lista con los números de las condiciones que queres que tire
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

        
    if drop != None:
        
        df = df.drop(drop, axis = 1)

    condicion_labels = list(df['Condición'])


    ind_camp = encontrar_intervalos(condicion_labels, 1)
    ind_pres = encontrar_intervalos(condicion_labels, 2)
    ind_cfk = encontrar_intervalos(condicion_labels, 3)
    ind_ar = encontrar_intervalos(condicion_labels, 4)
    ind_fil = encontrar_intervalos(condicion_labels, 5)
    
    indices = [ind_camp, ind_pres, ind_cfk, ind_ar, ind_fil]
    
    return condicion_labels, indices 
    

def etiquetas(labels, indices):
    #devuelve algo de la pinta
    '''
    indices es una lista de los indices primero esta el de campeones, después presencial, después
    cfk arabia y filler. Si los haces con la funcion indices_condiciones ya lo devuelve asi
    devuelve algo de la pinta (array([0, 1, 2, 3]), array([19,  8,  1,  2], dtype=int64))
    lo que quiere decir que en los primeros 30 datos (que corresponden a los 30 relatos de campeones) 19 fueron asignados al
    cluster 0, 8 al cluster 1, 1 al cluster 2 y 2 al cluster 3
    '''
    ind_camp = indices[0]
    ind_pres = indices[1]
    ind_cfk = indices[2]
    ind_ar = indices[3]
    ind_fil = indices[4]
    
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

#%% el santo trial y colores

entrevista = 'Primera'

no_autop = True #pone false si queres que las tenga en cuenta para el análisis

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

color_celeste = "#79b4b7ff"
color_gris = "#9fa0a4ff"

drop_12 = True

eliminando_outliers = True

metodo = 0 #0 kmedoids, 1 complete, 2 average, 3 ward
kmeans_TF = False
#%% path data 
path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

if eliminando_outliers == True:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv'

    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv'

    
#path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_{condicion}.csv'

#path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_{condicion}.csv'


mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3 
}

eliminamos_pysent = ['Valencia pysent', 'Valencia e intensidad pysent']#, 'Valencia2 pysent', 'Valencia e intensidad2 pysent']


#%% data PCA para eleccion de vars
#df_vars = pd.read_csv(path_conautopercepcion_todas)
#condicion_labels, indices_pres_cfk = indices_condiciones(path_conautopercepcion_todas, condiciones = [4,3,1])
if no_autop == True:
    df_vars = pd.read_csv(path_sinautopercepcion_todas)
    condicion_labels, indices_pres_cfk = indices_condiciones(path_sinautopercepcion_todas, condiciones = [4,3,1])

df_vars['Condición'] = df_vars['Condición'].map(mapping)

df_vars = df_vars[~df_vars['Condición'].isin([4, 1,3])]

df_vars = df_vars.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)

df_vars = df_vars.dropna()
#%% PCA
X_pca, pca, evr = PCA_estandarizando(df_vars)
varianza_acumulada = np.cumsum(evr)

#%% elección nro PCs para filler vs presencial (tarda si kmeans = True)
R_pcs = []
#R_pcs_kmeans = []
R_pcs_metodo = []
k = 2
nro_pcs_prueba = np.linspace(2,len(X_pca[0]), len(X_pca[0])-1).astype(int)

for i in tqdm(nro_pcs_prueba):
    R = R_clausterizacion(X_pca[:,:i], k, condicion_labels, indices_pres_cfk, kmeans = kmeans_TF, etiquetas_print = False)
    R_pcs.append(max(R))
    #R_pcs_kmeans.append(R[0])
    R_pcs_metodo.append(R[metodo])
    
#%% elección nro PCs para filler vs presencial parte 2
# R_pcs_con_kmeans_primera_entrevista_cfk_pres = [0.14214860417880368, 0.11871164799004963, 0.11855552406147644, 0.18111928577333222, 0.15467305905877538, 0.1812288690956465, 0.2100815541314072, 0.14205742135368893, 0.18116763497236787, 0.18116763497236787, 0.22463431531545716, 0.18116763497236787, 0.15481278581942115, 0.19519328035393074, 0.19513942466659318, 0.15462314544731673, 0.19513942466659318, 0.18116763497236787, 0.18116763497236787, 0.18116763497236787, 0.18116763497236787, 0.19513942466659318, 0.1549025857760673, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318]
#R_pcs_kmeans_sin_elim_outliers = [0.13012671382515145, 0.11871164799004963, 0.11855552406147644, 0.01956561962388344,0.15467305905877538, 0.1812288690956465, 0.13005479969765685, 0.13012671382515145, 0.18116763497236787, 0.18116763497236787,0.22463431531545716, 0.18116763497236787, 0.14207430852445851, 0.19513942466659318, 0.19513942466659318, 0.14205742135368893, 0.19513942466659318, 0.18116763497236787, 0.18116763497236787, 0.18116763497236787, 0.18116763497236787, 0.19513942466659318, 0.13012671382515145, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318, 0.19513942466659318]
plt.figure(4), plt.clf()
plt.plot(nro_pcs_prueba, R_pcs_metodo, 'o-')
plt.xlabel("Nro pcs")
plt.ylabel("R kmeans index")
plt.grid()
plt.plot   
#%%
# si uso valencia pysent max(R_pcs) = 0.256 se da con 10 11 y 20 PCs
# en todos los casos solo kmedoids da alto
# con 10 PCs
# Indice R con kmedoids 2 y PCA:  0.25604979194305894
# etiquetas presencial: (array([0, 1], dtype=int64), array([46, 16], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([14, 49], dtype=int64))
# con 11 PCs
# Indice R con kmedoids 2 y PCA:  0.25604979194305894
# etiquetas presencial: (array([0, 1], dtype=int64), array([16, 46], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([49, 14], dtype=int64))
# con 20 PCs tambien ward
# Indice R con kmedoids 2 y PCA:  0.19513942466659318
# etiquetas presencial: (array([0, 1], dtype=int64), array([41, 21], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([13, 50], dtype=int64))14, 49], dtype=int64))
# Indice R usando ward, k = 2 0.2560907920686845
# etiquetas presencial: (array([0, 1], dtype=int64), array([18, 44], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([51, 12], dtype=int64))

# si uso valencia pysent2 max(R_pcs) = 0.2246 se da con 12 PCs, 8 PCs tiene 0.21 
# 12 PCs se tiene con kmedoids y con complete algo cercano
# Indice R con kmedoids 2 y PCA:  0.22463431531545716
# etiquetas presencial: (array([0, 1], dtype=int64), array([20, 42], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([51, 12], dtype=int64))
# Indice R usando complete, k = 2 0.22454884731670446
# etiquetas presencial: (array([0, 1], dtype=int64), array([46, 16], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([16, 47], dtype=int64))
# 8 PCs se tiene solo con complete
# Indice R usando complete, k = 2 0.2100815541314072
# etiquetas presencial: (array([0, 1], dtype=int64), array([26, 36], dtype=int64))
# etiquetas filler: (array([0, 1], dtype=int64), array([55,  8], dtype=int64))

# print("con 10")
# R_clausterizacion(X_pca[:,:10], k, condicion_labels, indices_pres_cfk, kmeans = False, etiquetas_print = True)
# print("con 11")
# R_clausterizacion(X_pca[:,:11], k, condicion_labels, indices_pres_cfk, kmeans = False, etiquetas_print = True)
# print("con 20")
# R_clausterizacion(X_pca[:,:20], k, condicion_labels, indices_pres_cfk, kmeans = False, etiquetas_print = True)


#_clausterizacion(X_pca[:,:8], k, condicion_labels, indices_pres_cfk, kmeans = True, etiquetas_print = True)
#print("con 12")
#R_clausterizacion(X_pca[:,:12], k, condicion_labels, indices_pres_cfk, kmeans = False, etiquetas_print = True)

#print("con 6")
#R_clausterizacion(X_pca[:,:6], k, condicion_labels, indices_pres_cfk, kmeans = False, etiquetas_print = True)


#%%
#si uso valencia2 pysent
#max_posicion = np.where(np.asarray(R_pcs) == max(R_pcs))[0][0]
#nro_pcs = nro_pcs_prueba[max_posicion]
max_posicion = np.where(np.asarray(R_pcs_metodo) == max(R_pcs_metodo))[0][0]
nro_pcs = nro_pcs_prueba[max_posicion]
print(f"con {nro_pcs} PCs maximiza filler vs presencial")
R_clausterizacion(X_pca[:,:nro_pcs], k, condicion_labels, indices_pres_cfk, kmeans = kmeans_TF, etiquetas_print = True)

#nro_pcs = 12
#nro_pcs = 10 #si uso valencia pysent
variables = list(df_vars.columns)
componentes_principales = [pca.components_[i] for i in range(0, nro_pcs)]
# Crea un diccionario con las componentes principales y las variables
data_pcs = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}

# Crea el DataFrame
df_vars_1 = pd.DataFrame(data_pcs, index=variables)

#%%eleccion de variables

vars_no_imp_n = []
k = 2
R_n = []
R_n_metodo = []
importancia_pca = evr*100
pca_importan = importancia_pca[0]
pca1 = importancia_pca[1]
pca2 = importancia_pca[2]
pca3 = importancia_pca[3]
pca4 = importancia_pca[4]
pca5 = importancia_pca[5]
pca6 = importancia_pca[6]
pca7 = importancia_pca[7]
pca8 = importancia_pca[8]
pca9 = importancia_pca[9]
pca10 = importancia_pca[10]
pca11 = importancia_pca[11]
pca12 = importancia_pca[12]
pca13 = importancia_pca[13]
pca14 = importancia_pca[14]
max_comp = np.where(varianza_acumulada*100 > 70)[0][0]
ns_a_recorrer =[3, 3.5, 4, 4.5,  5, 5.5,  6, 6.5, 7, 7.5,  8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15]
for n in ns_a_recorrer:
    a = set(np.argsort(abs(pca.components_[0]))[-round(n):]) #n componentes mas importantes de la pca0 --> explica el 37
    b = []
    for i in range(1, nro_pcs):
        if round(n*importancia_pca[i]/importancia_pca[0]) != 0:
            b.append(set(np.argsort(abs(pca.components_[i]))[-round(n*importancia_pca[i]/importancia_pca[0]):])) #n componentes mas importantes de la pca1 --> explica el 9
    c = set().union(*b)
    elem_com =  list(a | c)
   
    complemento_elem_com = set(range(0,len(df_vars.columns))) - set(elem_com)
    #print(len(complemento_elem_com))
    vars_no_imp = [df_vars_1.index[indice] for indice in complemento_elem_com]
    vars_no_imp_n.append(vars_no_imp)

#vars_no_imp_n = [['Intensidad_autop', 'primera_persona_norm', 'num adj norm', 'num advs norm', 'num_nodes_LSC', 'Comunidades_LSC', 'k_mean', 'transitivity', 'ASP', 'selfloops', 'L2', 'L3', 'density']]

for n in tqdm(range(len(vars_no_imp_n))):
    df = pd.read_csv(path_conautopercepcion_todas)

    df = df.dropna()

    # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
    df['Condición'] = df['Condición'].map(mapping)

    df = df[~df['Condición'].isin([5,2])]

    condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_conautopercepcion_todas, condiciones = [5, 2])
                   
    #df = df.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)
    
    if no_autop == True:
        df = pd.read_csv(path_conautopercepcion_todas)
        
        if drop_12 == True:
            df = df.dropna()
        
        df['Condición'] = df['Condición'].map(mapping)

        df = df[~df['Condición'].isin([5,2])]

        df = df.drop(['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'] + eliminamos_pysent, axis = 1) 

        if drop_12 != True:
            df = df.dropna()
        
            condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_sinautopercepcion_todas, condiciones = [5, 2], drop = vars_no_imp_n[n])
        
    df = df.drop(vars_no_imp_n[n], axis = 1)

    #% primero hago PCA a los datos, ya vi que me tengo que quedar con 6 componentes para tener un 70% de la varianza
    R_pca = []
    R_pca_metodo = []
    pcs_recorridas = [4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18]
    for nro_pca in pcs_recorridas:
        try:
            
            X_pca, pca1, evr1 = PCA_estandarizando(df, n_components = nro_pca, graph_var = False, graph_PCs = False)
         
            # clausterizacion
            
            R = R_clausterizacion(X_pca, k, condicion_labels, indices_camp_ar_cfk, kmeans = kmeans_TF, etiquetas_print = False)
            
            lista_redondeada = [round(numero, 3) for numero in R]
            #print(lista_redondeada)
            R_pca.append(max(lista_redondeada))
            R_pca_metodo.append(lista_redondeada[metodo])
        except ValueError as e:
             print(e)
             R_pca.append(np.nan)
             R_pca_metodo.append(np.nan)
    R_n.append(R_pca)
    R_n_metodo.append(R_pca_metodo)
    
#print(R_n)
print(np.nanmax(R_n))
R_n = np.array(R_n)
indice_maximo = np.unravel_index(np.nanargmax(R_n), R_n.shape)
print("El número de PCs que maximiza es", pcs_recorridas[indice_maximo[1]])
print("El n que maximiza es", ns_a_recorrer[indice_maximo[0]])
print(R_n[indice_maximo[0]])
print(vars_no_imp_n[indice_maximo[0]])

print(np.nanmax(R_n_metodo))
R_n_metodo = np.array(R_n_metodo)
indice_maximo = np.unravel_index(np.nanargmax(R_n_metodo), R_n_metodo.shape)
print("El número de PCs que maximiza el método es", pcs_recorridas[indice_maximo[1]])
print("El n que maximiza el método es", ns_a_recorrer[indice_maximo[0]])
print(R_n_metodo[indice_maximo[0]])
print(vars_no_imp_n[indice_maximo[0]])


#%%
df = pd.read_csv(path_conautopercepcion_todas)

df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin([5,2])]

condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_conautopercepcion_todas, condiciones = [5, 2])
               
#df = df.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)

if no_autop == True:
    df = pd.read_csv(path_conautopercepcion_todas)
    
    if drop_12 == True:
        df = df.dropna()
    
    df['Condición'] = df['Condición'].map(mapping)

    df = df[~df['Condición'].isin([5,2])]

    df = df.drop(['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'] + eliminamos_pysent, axis = 1) 

    if drop_12 != True:
        df = df.dropna()
    
        condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_sinautopercepcion_todas, condiciones = [5, 2], drop = vars_no_imp_n[n])
    
df = df.drop(vars_no_imp_n[indice_maximo[0]], axis = 1)

X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  pcs_recorridas[indice_maximo[1]], max_var = 0.6, graph_var = True, graph_PCs = True, n_graph_PCs = pcs_recorridas[indice_maximo[1]])

k = 2
R = R_clausterizacion(X_pca, k, condicion_labels, indices_camp_ar_cfk, kmeans = kmeans_TF, etiquetas_print = True)

#%%
entrevista = "Segunda"
drop_12 = False

path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

if eliminando_outliers == True:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv'

    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv'

df = pd.read_csv(path_conautopercepcion_todas)

df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin([5,2])]

condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_conautopercepcion_todas, condiciones = [5, 2])
               
#df = df.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)

if no_autop == True:
    df = pd.read_csv(path_conautopercepcion_todas)
    
    if drop_12 == True:
        df = df.dropna()
    
    df['Condición'] = df['Condición'].map(mapping)

    df = df[~df['Condición'].isin([5,2])]

    df = df.drop(['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'] + eliminamos_pysent, axis = 1) 

    if drop_12 != True:
        df = df.dropna()
    
        condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_sinautopercepcion_todas, condiciones = [5, 2], drop = vars_no_imp_n[n])
    
df = df.drop(vars_no_imp_n[indice_maximo[0]], axis = 1)

X = df.to_numpy()

# Ajustamos el estandarizador
std_scale.fit(X)

# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)
X_pca = pca1.transform(X_scaled)
k = 2
print("PARA LA SEGUNDAAAAAAAAAAAAAAAAA")
R = R_clausterizacion(X_pca, k, condicion_labels, indices_camp_ar_cfk, kmeans = kmeans_TF, etiquetas_print = True)

