# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:23:47 2023

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

#%%FUNCIONES
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
    
    R = []
    if kmeans != False:
        #Creación del modelo KMeans 
        kmeans5 = KMeans(n_clusters=k, init = "random",  n_init = 10000, random_state = 42)
        
        #Ajuste del modelo a los datos reducidos en componentes principales PCA
        kmeans5.fit(data_norm)
        R_index5mean = adjusted_rand_score(condicion_labels, kmeans5.labels_) 
        centroids_kmeans = kmeans5.cluster_centers_
        if etiquetas_print != False:
            print(f"Indice R con kmeans {k} y PCA: ", R_index5mean)
            etiquetas(kmeans5.labels_, indices)
            R.append(R_index5mean)
    
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
        labels_j, centroids_j, metodo = cluters_jerarquico(data, k, dist, 'cosine')
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

def encontrar_posiciones_tupla(lista_tuplas, tupla_objetivo):
    inicio = None
    fin = None

    for i, tupla in enumerate(lista_tuplas):
        if tupla == tupla_objetivo:
            if inicio is None:
                inicio = i
            fin = i

    return inicio, fin

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

#%% path data 

if entrevista == 'dos tiempos':
    path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_dostiempos.csv'
    
    path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_dostiempos.csv'
else:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'
    
    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'


mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3 
}

eliminamos_pysent = ['Valencia pysent', 'Valencia e intensidad pysent']#, 'Valencia2 pysent', 'Valencia e intensidad2 pysent']
#%% eleccion vars
nro_pcs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

k = 2

cond_elim = [1,2,3,5]

valor_a_buscar = 4

df_vars = pd.read_csv(path_conautopercepcion_todas)

#df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df_vars['Condición'] = df_vars['Condición'].map(mapping)

df_vars = df_vars[~df_vars['Condición'].isin(cond_elim)]

condicion_labels, indices_ = indices_condiciones(path_conautopercepcion_todas, condiciones = cond_elim)
               
df_vars = df_vars.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)

if no_autop == True:
    df_vars = df_vars.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 
    condicion_labels, indices_ = indices_condiciones(path_sinautopercepcion_todas, condiciones = cond_elim)

    
df_vars = df_vars.dropna()
      
df_vars = df_vars.drop(['Tiempo'], axis = 1) 

R_s = []
for nro_pc in nro_pcs:
    X_pca, pca1, evr1 = PCA_estandarizando(df_vars, n_components =  nro_pc, graph_var = False, graph_PCs = False)
    
    R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = False)
    R_s.append(max(R))
    for clave, valor in mapping.items():
        if valor == valor_a_buscar:
            clave_correspondiente = clave
            break
        
plt.figure(1), plt.clf()
plt.plot(nro_pcs, R_s, 'o-')
plt.xlabel("Nro pcs")
plt.ylabel("R index")
plt.grid()
plt.plot   
        
mejor_pcs = nro_pcs[np.where(np.asarray(R_s) == max(R_s))[0][0]]
print(f"para {clave_correspondiente}", max(R_s))
print("nro PCs", mejor_pcs)
#COMO ES BASICAMENTE 0 TODOS LOS R, TOMO EL NRO DE PCs QUE ME DEJAN EL 70% DE LA VAR
mejor_pcs = 8
X_pca, pca, evr = PCA_estandarizando(df_vars, n_components =  mejor_pcs, graph_var = False, graph_PCs = False)
varianza_acumulada = np.cumsum(evr)
R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = True)
R_s.append(max(R))
for clave, valor in mapping.items():
    if valor == valor_a_buscar:
        clave_correspondiente = clave
        break
variables = list(df_vars.columns)
componentes_principales = [pca.components_[i] for i in range(0, mejor_pcs)]
# Crea un diccionario con las componentes principales y las variables
data_pcs = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}

# Crea el DataFrame
df_vars_1 = pd.DataFrame(data_pcs, index=variables)
#%%   
vars_no_imp_n = []
k = 2
R_n = []
importancia_pca = evr*100

max_comp = np.where(varianza_acumulada*100 > 70)[0][0]
ns_a_recorrer =[3, 3.5, 4, 4.5,  5, 5.5,  6, 6.5, 7, 7.5,  8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15]
for n in ns_a_recorrer:
    a = set(np.argsort(abs(pca.components_[0]))[-round(n):]) #n componentes mas importantes de la pca0 --> explica el 37
    b = []
    for i in range(1, mejor_pcs):
        if round(n*importancia_pca[i]/importancia_pca[0]) != 0:
            b.append(set(np.argsort(abs(pca.components_[i]))[-round(n*importancia_pca[i]/importancia_pca[0]):])) #n componentes mas importantes de la pca1 --> explica el 9
    c = set().union(*b)
    elem_com =  list(a | c)
   
    complemento_elem_com = set(range(0,len(df_vars.columns))) - set(elem_com)
    #print(len(complemento_elem_com))
    vars_no_imp = [df_vars_1.index[indice] for indice in complemento_elem_com]
    vars_no_imp_n.append(vars_no_imp)
#%%
#valencia2 pysent 12 PCs
#vars_no_imp = ['primera_persona_norm', 'transitivity', 'num adj norm']
#valencia2 pysent 8 PCs
#vars_no_imp = ['primera_persona_norm', 'num adj norm', 'num advs norm', 'cohe_norm_d=1', 'diámetro', 'transitivity', 'selfloops']

pcs_a_recorrer = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

R_n = []
for n in range(len(vars_no_imp_n)):
    df = pd.read_csv(path_conautopercepcion_todas)
    
    #df = df.dropna()
    
    # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
    df['Condición'] = df['Condición'].map(mapping)
    
    df = df[~df['Condición'].isin(cond_elim)]
    
    condicion_labels, indices_ = indices_condiciones(path_conautopercepcion_todas, condiciones = cond_elim)
                   
    df = df.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)
    
    if no_autop == True:
        df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 
        condicion_labels, indices_ = indices_condiciones(path_sinautopercepcion_todas, condiciones = cond_elim)
    
        
    df = df.dropna()
          
    df = df.drop(vars_no_imp_n[n] + ['Tiempo'], axis = 1) #
    
    R_s = []
    for nro_pc in pcs_a_recorrer:
        try: 
            X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  nro_pc, graph_var = False, graph_PCs = False)
            
            R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = False)
            R_s.append(max(R))
        except ValueError as e:
             print(e)
             R_s.append(np.nan)
    R_n.append(R_s)
    
        
for clave, valor in mapping.items():
    if valor == valor_a_buscar:
        clave_correspondiente = clave
        break

#print(R_n)
print(np.nanmax(R_n))
R_n = np.array(R_n)
indice_maximo = np.unravel_index(np.nanargmax(R_n), R_n.shape)
print(f"El número de PCs que maximiza para {clave_correspondiente} es", pcs_a_recorrer[indice_maximo[1]])
print(f"El n que maximiza para {clave_correspondiente} es", ns_a_recorrer[indice_maximo[0]])
print(R_n[indice_maximo[0]])
print(vars_no_imp_n[indice_maximo[0]])

X_pca, pca, evr = PCA_estandarizando(df, n_components =  pcs_a_recorrer[indice_maximo[1]], graph_var = False, graph_PCs = False)
varianza_acumulada = np.cumsum(evr)
R = R_clausterizacion(X_pca, k, condicion_labels, indices_, kmeans = False, etiquetas_print = True)
R_s.append(max(R))



