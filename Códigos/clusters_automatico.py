# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:49:47 2023

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
from sklearn.neighbors import NearestCentroid #para los centroids

#R index
from sklearn.metrics.cluster import adjusted_rand_score #si da cerca de 0 es azar, si da cerca de 1 buen match

#DBScan
from sklearn.cluster import DBSCAN

#TSNE
from sklearn.manifold import TSNE

#%% el santo trial y colores

entrevista = 'Primera'

no_autop = True #si queres que haga el analisis con las variables de autopercepcion poné False

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

color_celeste = "#79b4b7ff"
color_gris = "#9fa0a4ff"
#%% data

condicion = temas[0]

path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

#path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

#path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_conautopercepcion_todas)
 
mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3
}

eliminamos_pysent = ['Valencia pysent', 'Valencia e intensidad pysent']


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

vars_autop = 0
if no_autop == True:
    df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'] + eliminamos_pysent, axis = 1)
    vars_autop = 4

#df = df.drop(['Recuerdo_autop', 'num_palabras_unicas_norm', 'num advs norm', 'diámetro', 'transitivity', 'selfloops',  'ASP', 'L2', 'L3', 'density'], axis=1)

#drop = ['Recuerdo_autop', 'Valencia_autop', 'num_palabras_unicas_norm', 'primera_persona_norm', 'tercera_persona_norm', 'num adj norm', 'num advs norm', 'num propn norm', 'cohe_norm_d=1', 'cohe_norm_d=2', 'num_nodes_LSC', 'Comunidades_LSC', 'diámetro', 'transitivity', 'ASP', 'average_CC', 'selfloops', 'L2', 'density']
#drop = ['Intensidad_autop', 'primera_persona_norm', 'num adj norm', 'num advs norm', 'num_nodes_LSC', 'Comunidades_LSC', 'k_mean', 'transitivity', 'ASP', 'selfloops', 'L2', 'L3', 'density']

#drop = ['primera_persona_norm', 'num verb norm', 'num adj norm', 'num_nodes_LSC', 'Comunidades_LSC', 'L2', 'L3', 'density']

#drop =  ['num_palabras_unicas_norm', 'primera_persona_norm', 'num adj norm', 'cohe_norm_d=1', 'Comunidades_LSC', 'diámetro', 'ASP', 'average_CC', 'L2', 'L3', 'density', 'Detalles internos norm']

#drop = ['primera_persona_norm', 'tercera_persona_norm', 'num adj norm', 'num advs norm', 'Intensidad pysent', 'diámetro', 'transitivity', 'selfloops']

drop = ['num adj norm'] #'primera_persona_norm', 'transitivity', 

df = df.drop(drop, axis = 1)

#%% defino nro_pca
X = df.to_numpy()

# Ajustamos el estandarizador
std_scale.fit(X)
# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)

# Creación del modelo. Si el número de componentes no se específica, 
# se obtienen tantas componentes principales como features en nuestro dataset.
pca = PCA(n_components=None) #si queremos solo 5 poner n_components = 6
# Ajustamos el modelo a los datos escaleados
pca.fit(X_scaled)
# Obtenemos la descripción de los datos en el espacio de componentes principales
X_pca = pca.transform(X_scaled)

# con .explained_variance_ratio_ vemos la fracción de información que aporta cada componente
evr = pca.explained_variance_ratio_

fig, ax = plt.subplots(1, 1, figsize = (18, 10))

# Calculamos el acumulado con la función cumsum de numpy 
varianza_acumulada = np.cumsum(evr)

ax.plot(range(1, len(evr) + 1), varianza_acumulada, '.-', markersize = 20, color = color_celeste, zorder = 5)
ax.set_ylabel('Fracción acumulada de varianza explicada', fontsize = 20)
ax.set_xlabel('Cantidad de componentes principales', fontsize = 20)
ax.axhline(y=0.8, color=color_gris, linestyle='--', linewidth = 4, label='70%')
ax.axvline(x = 6, color=color_gris, linestyle='--', linewidth = 4)
ax.grid(True)
ax.tick_params(axis='both', which='both', labelsize=16)
plt.legend(fontsize=15)
nro_pca = 4

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

def markerstemas_colorcluster3(labels, X, temaslabels, save = None, centroids =  None, title =  None):

    fig, ax = plt.subplots(figsize = (20, 7))
    
    # Hacemos un scatter plot de cada uno de los datos
    ax.scatter(X[ind_cfk[0][0]:ind_cfk[0][1], 0], X[ind_cfk[0][0]:ind_cfk[0][1], 1], marker = "o", c=labels[ind_cfk[0][0]:ind_cfk[0][1]], label = temaslabels[0])
    ax.scatter(X[ind_camp[0][0]:ind_camp[0][1], 0], X[ind_camp[0][0]:ind_camp[0][1], 1], marker = "v", c=labels[ind_camp[0][0]:ind_camp[0][1]], label = temaslabels[1])
    #ax.scatter(X[ind_fil[0][0]:ind_fil[0][1], 0], X[ind_fil[0][0]:ind_fil[0][1], 1], marker = "s", c=labels[ind_fil[0][0]:ind_fil[0][1]], label = temaslabels[2])
    #ax.scatter(X[ind_pres[0][0]:ind_pres[0][1], 0], X[ind_pres[0][0]:ind_pres[0][1], 1],  marker = "*", c=labels[ind_pres[0][0]:ind_pres[0][1]], label = temaslabels[3])
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
    
    cent = NearestCentroid()
    cent.fit(X, cluster.labels_)

    
    #centroids_jerarquico = []
    # for cluster_label in np.unique(labels):
    #     cluster_points = X_pca[labels == cluster_label]
    #     centroid_jerarquico = cluster_points.mean(axis=0)
    #     centroids_jerarquico.append(centroid_jerarquico)
        
    # centroids_jerarquico = np.array(centroids_jerarquico)
    
    centroids_jerarquico = cent.centroids_
        
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



#%% primero hago PCA a los datos

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

# para kmeans con conseno hay que normalizar X

length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length

#%% clausterizacion

R = []

k = 2

# Creación del modelo KMeans con k = 2
kmeans5 = KMeans(n_clusters=k, init = "random",  n_init = 10000, random_state = 42)

# Ajuste del modelo a los datos reducidos en componentes principales
#PCA
kmeans5.fit(X_pca_norm)
R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
print(f"Indice R con kmeans {k} y PCA: ", R_index5mean)
etiquetas(kmeans5.labels_)
R.append(R_index5mean)

# kmedoids con Kmedoids de sklear_extra

kmedoids5 = KMedoids(n_clusters=k, metric = "cosine")
kmedoids5.fit(X_pca)

# Guardo las posiciones de los centroids
centroids = kmedoids5.cluster_centers_

#markerstemas_colorcluster(kmedoids5.labels_, X_pca, temas, centroids = centroids)  
R_index5 = adjusted_rand_score(condicion_labels, kmedoids5.labels_) 
R.append(R_index5)
print(f"Indice R con kmedoids {k} y PCA: ", R_index5)
#etiquetas
etiquetas(kmedoids5.labels_)

# clusterización jerárquica con k óptimo
'''
Ahora sí aplicamos el método de clusterización jerárquica (bottom-up) con 5 clusters, la distancia euclidea para la afinidad y la distancia ward para el linkage
'''

distancias_con_cosine = ["single", "complete", "average"] #"centroid", "ward"
distancias_sin_cosine = ["ward", "centroid"]

for i, dist in enumerate(distancias_con_cosine):
     #d, lab = dendograma(X_pca, dist, 'cosine')
     matriz = shc.linkage(X_pca_norm, method=dist, metric = 'cosine')
     etiquet = shc.cut_tree(matriz, n_clusters=k).flatten()
     
length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length
for i, dist in enumerate(distancias_sin_cosine):
     #dendograma(X_pca_norm, dist, 'euclidean')
     matriz = shc.linkage(X_pca_norm, method=dist, metric = 'euclidean')
     etiquet = shc.cut_tree(matriz, n_clusters=k).flatten()

for i, dist in enumerate(distancias_con_cosine[1:]):
    labels_j, centroids_j, metodo = cluters_jerarquico(X_pca, k, dist, 'cosine')
    #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
    #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
    R_jer = adjusted_rand_score(condicion_labels, labels_j)
    print(f"Indice R usando {dist}, k = {k}", R_jer)
    etiquetas(labels_j)
    R.append(R_jer)
    
    
length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length
for i, dist in enumerate(distancias_sin_cosine[:1]):
    labels_j, centroids_j, metodo = cluters_jerarquico(X_pca_norm, k, dist, 'euclidean')
    #markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
    #markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
    R_jer = adjusted_rand_score(condicion_labels, labels_j)
    print(f"Indice R usando {dist}, k = {k}", R_jer)
    etiquetas(labels_j)
    R.append(R_jer)


lista_redondeada = [round(numero, 3) for numero in R]
print(lista_redondeada)
#R_3 = R
    
#[0.326, 0.335, 0.475, 0.381, 0.418]

#%%

length = np.sqrt((X_pca**2).sum(axis=1))[:,None]
X_pca_norm = X_pca / length

labels_j, centroids_j, metodo = cluters_jerarquico(X_pca_norm, k, 'ward', 'euclidean')
#markerscluster_colortemas(labels_j, X_pca, temas, save = None, centroids =  centroids_j, title = metodo)
#markerstemas_colorcluster(labels_j, X_pca, temas, save = None, centroids =  centroids_j)
R_jer = adjusted_rand_score(condicion_labels, labels_j)
print(f"Indice R usando {dist}, k = {k}", R_jer)
etiquetas(labels_j)
R.append(R_jer)

#markerstemas_colorcluster3(labels_j, X_pca_norm, temas, save = None, centroids =  centroids_j, title =  "Jerárquico ward")