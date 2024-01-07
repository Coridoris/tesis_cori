# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:31:18 2023

@author: corir
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#rompí la siguiente librería por actualizar scipy y numpy, no se cómo arreglarlo.
from netgraph import Graph # pip install netgraph
import re
from collections import Counter
import igraph as ig #para buscar comunidades con Infomap en el grafo
from tqdm import tqdm

#%% red sust adj verb

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

tema = temas[0]

#solo tiene noun, verb, adj
path_3 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Categorias tres/df_red_sustverbadj_{tema}.csv'
#tiene las 6 clasificaciones (noun, verb, adj, num, propn, adv)
path_6 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Categorias seis/df_red_sustverbadjetc_{tema}_sin_meacuerdo.csv'

data_df = pd.read_csv(path_6)

# Inicializar un diccionario vacío para cada fila
fila_dict_list = []

# Iterar a través de las filas del DataFrame
for index, row in data_df.iterrows():
    fila_dict = {}  # Diccionario para esta fila
    
    # Iterar a través de las columnas que representan las opciones de tuplas
    for col in data_df.columns[22:]:  # Empezar desde la (22 para path 6) 13 ava columna (las primeras son "Sujeto" "texto" "Tuplas" etc)
        tupla = eval(col)  # Convertir el nombre de la columna en una tupla
        valor = row[col]  # Obtener el valor en esta columna para esta fila
        fila_dict[tupla] = valor  # Agregar al diccionario de la fila
    
    fila_dict_list.append(fila_dict)

# Ahora fila_dict_list contiene una lista de diccionarios, uno por fila

# Visualizar la lista de diccionarios
#for i, fila_dict in enumerate(fila_dict_list):
#    print(f"Sujeto {i + 1}: {fila_dict}")
    
#%% Hago la red de alguno

sujeto = 5

dict_sujeto = fila_dict_list[sujeto + 1]

#normalicemos dividiendo por el enlace con mayor peso
#si no se quiere normalizar comentar las dos siguientes lineas
max_valor = max(dict_sujeto.values())
dict_sujeto = {clave: valor / max_valor for clave, valor in dict_sujeto.items()}


enlaces = list(dict_sujeto.keys())

triangle = nx.DiGraph(enlaces)

# node_positions = {
#     'NOUN' : np.array([0.2, 0.2]),
#     'VERB' : np.array([0.8, 0.2]),
#     'ADJ' : np.array([0.5, 0.8]),
# }

node_positions = {
    'NOUN': np.array([0.2, 0.2]),
    'VERB': np.array([0.8, 0.2]),
    'ADJ': np.array([0.5, 0.7]),
    'ADV': np.array([0.2, 0.8]),  
    'PROPN': np.array([0.8, 0.8]),  
    'NUM': np.array([0.5, 0.3]),  
}

#edge_labels = {
#    ('a', 'b') : 3,
#    ('a', 'c') : 'Lorem ipsum',
#    ('b', 'a') : 4,
#    ('c', 'b') : 'dolor sit',
#    ('c', 'c') : r'$\pi$'
#}

edge_weights = dict_sujeto

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))

Graph(triangle, node_labels=True, #edge_labels = edge_weights,
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      node_layout=node_positions, edge_layout='straight',
      node_size=6, edge_width=edge_weights  , arrows=True, ax=ax1)

Graph(triangle, node_labels=True, 
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      node_layout=node_positions, edge_layout='curved',
      node_size=6, edge_width=edge_weights, arrows=True, ax=ax2)

plt.show()

#%% hago la red promedio

#primero me armo el diccionario

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

dict_prom_temas = []

for tema in temas:
    
    path_6 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Categorias seis/df_red_sustverbadjetc_{tema}_sin_meacuerdo.csv'

    data_df = pd.read_csv(path_6)

    # Inicializar un diccionario vacío para cada fila
    fila_dict_list = []

    # Iterar a través de las filas del DataFrame
    for index, row in data_df.iterrows():
        fila_dict = {}  # Diccionario para esta fila
        
        # Iterar a través de las columnas que representan las opciones de tuplas
        for col in data_df.columns[22:]:  # Empezar desde la (22 para path 6) 13 ava columna (las primeras son "Sujeto" "texto" "Tuplas" etc)
            tupla = eval(col)  # Convertir el nombre de la columna en una tupla
            valor = row[col]  # Obtener el valor en esta columna para esta fila
            fila_dict[tupla] = valor  # Agregar al diccionario de la fila
        
        fila_dict_list.append(fila_dict)

    fila_dict_list_norm = []
    
    for sujeto in range(0, len(fila_dict_list)):
    
        dict_sujeto = fila_dict_list[sujeto]
    
        #normalicemos dividiendo por el enlace con mayor peso
        #si no se quiere normalizar comentar las dos siguientes lineas
        max_valor = max(dict_sujeto.values())
        dict_sujeto_norm = {clave: valor / max_valor for clave, valor in dict_sujeto.items()}
    
        fila_dict_list_norm.append(dict_sujeto_norm)
        
    def mean_dict(list_of_dicts):
        # Inicializa el diccionario promedio como un diccionario vacío
        diccionario_promedio = {}
        
        # Cuenta la cantidad de diccionarios en la lista
        total_diccionarios = len(list_of_dicts)
        
        # Recorre los diccionarios en la lista
        for diccionario in list_of_dicts:
            # Recorre las claves en cada diccionario
            for clave, valor in diccionario.items():
                # Agrega el valor al diccionario promedio
                if clave in diccionario_promedio:
                    diccionario_promedio[clave] += valor
                else:
                    diccionario_promedio[clave] = valor
        
        # Calcula el promedio dividiendo cada valor por el total de diccionarios
        for clave in diccionario_promedio:
            diccionario_promedio[clave] /= total_diccionarios
        
        return diccionario_promedio
    
    
    dict_promedio = mean_dict(fila_dict_list_norm)
    
    dict_prom_temas.append(dict_promedio)
    
    #guardo el diccionario como un csv
    
    #df_dict_prom = pd.DataFrame([dict_promedio])
    
    #df_dict_prom.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Promedio sesi categorias/df_red_promedio_categorias_{tema}_sin_meacuerdo.csv')



fig, ax = plt.subplots(2, 2, figsize=(14,14))

dict_selfloops = {}

list_de_dicts_selfloops = []

for i in [0,1,3,4]:

    j = 0
    tema = i
    dict_promedio = dict_prom_temas[tema]
    
    enlaces = list(dict_promedio.keys())
    
    G = nx.DiGraph(enlaces)
    
    selfloops = {llave: valor for llave, valor in dict_promedio.items() if llave[0] == llave[1]}
    
    list_de_dicts_selfloops.append(selfloops)
   
    num_selfloops = 0
    
    for valor in selfloops.values():
        num_selfloops += valor
    
    dict_selfloops[temas[i]] = num_selfloops
    
    node_positions = {
        'NOUN': np.array([0.2, 0.2]),
        'VERB': np.array([0.8, 0.2]),
        'ADJ': np.array([0.5, 0.7]),
        'ADV': np.array([0.2, 0.8]),  
        'PROPN': np.array([0.8, 0.8]),  
        'NUM': np.array([0.5, 0.3]),  
    }
    
    
    edge_weights = {clave: valor * 2.5 for clave, valor in dict_promedio.items()}
    
    if i == 3:
        i = 0
        j = 1
    if i ==4: 
        i = 1
        j = 1
    Graph(G, node_labels=True, #edge_labels = edge_weights,
          edge_label_fontdict=dict(size=12, fontweight='bold'),
          node_layout=node_positions, edge_layout='straight', #'curved'
          node_size=6, edge_width=edge_weights, edge_color= 'blue'  , arrows=True, ax=ax[i][j])
    
    ax[i][j].set_title(f"{temas[tema]}")
    
plt.show()

#%% red part of speech

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

tema = temas[0]

path_POS = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/df_red_partofspeech_{tema}.csv'

data_df_POS = pd.read_csv(path_POS)

# Inicializar un diccionario vacío para cada fila
fila_dict_list_POS = []

# Iterar a través de las filas del DataFrame
for index, row in data_df_POS.iterrows():
    fila_dict_POS = {}  # Diccionario para esta fila
    
    # Iterar a través de las columnas que representan las opciones de tuplas
    for col in data_df_POS.columns[13:]:  # Empezar desde la 14 ava columna (las primeras son "Sujeto" "texto" "Tuplas" etc)
        tupla = eval(col)  # Convertir el nombre de la columna en una tupla
        valor = row[col]  # Obtener el valor en esta columna para esta fila
        fila_dict_POS[tupla] = valor  # Agregar al diccionario de la fila
    
    fila_dict_list_POS.append(fila_dict_POS)

# Ahora fila_dict_list contiene una lista de diccionarios, uno por fila

# Visualizar la lista de diccionarios
for i, fila_dict in enumerate(fila_dict_list_POS):
    print(f"Sujeto {i + 1}: {fila_dict}")
    
#%% Hago la red de alguno

sujeto = 5

dict_sujeto_POS = fila_dict_list_POS[sujeto + 1]

enlaces_POS = list(dict_sujeto_POS.keys())

red = nx.DiGraph(enlaces_POS)

# node_positions = {
#     'NOUN' : np.array([0.2, 0.2]),
#     'VERB' : np.array([0.8, 0.2]),
#     'ADJ' : np.array([0.5, 0.8]),
# }

#edge_labels = {
#    ('a', 'b') : 3,
#    ('a', 'c') : 'Lorem ipsum',
#    ('b', 'a') : 4,
#    ('c', 'b') : 'dolor sit',
#    ('c', 'c') : r'$\pi$'
#}

edge_weights = dict_sujeto_POS

edge_weights_norm = {clave: valor/max(edge_weights.values()) for clave, valor in edge_weights.items()}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))

Graph(red, node_labels=True, #edge_labels = edge_weights,
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      edge_layout='straight',
      node_size=6, edge_width=edge_weights_norm , arrows=True, ax=ax1)

Graph(red, node_labels=True, 
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      edge_layout='curved',
      node_size=6, edge_width=edge_weights_norm, arrows=True, ax=ax2)

plt.show()

#%% red de todo el texto o lemmatizado, llevando todo a minúscula

def grafo_text_sujeto(sujeto, tema, lemm = 'False'):
        
    if lemm == "True":

        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
        
        data_df = pd.read_csv(path)
        
        texto = data_df["transcript"][sujeto+1]
        
    else:
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/df_red_sustverbadj_{tema}.csv'

        data_df = pd.read_csv(path)
        
        texto = data_df["texto"][sujeto+1]

    # Tokenizar el texto en palabras
    palabras = re.findall(r'\w+', texto.lower()) #lista de las palabras en orden
    
    # Contar ocurrencias de palabras consecutivas
    ocurrencias = Counter(zip(palabras, palabras[1:]))
    
    dict_texto = {}
    # Agregar nodos y arcos con pesos
    for (palabra_a, palabra_b), peso in ocurrencias.items():
        tupla = (palabra_a, palabra_b)
        valor = peso
        dict_texto[tupla] = valor
        
    enlaces_texto = list(dict_texto.keys())
    
    red = nx.DiGraph(enlaces_texto)
    
    edge_weights = dict_texto
    
    edge_weights_norm = {clave: valor/max(dict_texto.values()) for clave, valor in dict_texto.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))
    
    Graph(red, node_labels=True, #edge_labels = edge_weights,
          edge_label_fontdict=dict(size=12, fontweight='bold'),
          edge_layout='straight',
          node_size=6, edge_width=edge_weights_norm , arrows=True, ax=ax1)
    
    G = nx.DiGraph()
    
    # Agregar nodos y arcos con pesos
    for (palabra_a, palabra_b), peso in ocurrencias.items():
        G.add_edge(palabra_a, palabra_b, weight=peso)
    
    # Dibujar el grafo
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.show()
    
    return 'ok'
#%%
temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

tema = temas[0]
sujeto = 12

grafo_text_sujeto(sujeto, tema)

#%% componente fuertemeente conectada de la red de todo el texto o lemmatizado, llevando todo a minúscula

def LSC_text_sujeto(sujeto, tema, lemm = 'False', close = 'True', todas = False):
        
    if lemm == "True":

        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
        
        data_df = pd.read_csv(path)
        
        texto = data_df["transcript"][sujeto]
        
    else:
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Categorias tres/df_red_sustverbadj_{tema}.csv'

        data_df = pd.read_csv(path)
        
        texto = data_df["texto"][sujeto]

    # Tokenizar el texto en palabras
    palabras = re.findall(r'\w+', texto.lower()) #lista de las palabras en orden
    
    # Contar ocurrencias de palabras consecutivas
    ocurrencias = Counter(zip(palabras, palabras[1:]))
    
    edge_weights = {}
    # Agregar nodos y arcos con pesos
    for (palabra_a, palabra_b), peso in ocurrencias.items():
        tupla = (palabra_a, palabra_b)
        valor = peso
        edge_weights[tupla] = valor
        
    enlaces_texto = list(edge_weights.keys())

    red = nx.DiGraph(enlaces_texto)

    mean_weight = np.mean(list(edge_weights.values()))

    edge_weights_norm = {clave: mean_weight/valor for clave, valor in edge_weights.items()}

    

    G = nx.DiGraph()

    # Agregar nodos y arcos con pesos
    for (palabra_a, palabra_b), peso in edge_weights_norm.items():
        G.add_edge(palabra_a, palabra_b, weight=peso)
    
    # Encontrar las componentes fuertemente conexas
    componentes_fuertemente_conexas = list(nx.strongly_connected_components(G))
    
    # Seleccionar la componente fuertemente conexa más grande (puedes elegir la que desees)
    componente_mas_grande = max(componentes_fuertemente_conexas, key=len)
    
    # Crear un subgrafo con la componente fuertemente conexa
    LSC = G.subgraph(componente_mas_grande) #LSC, the maximal subgraph in which all pairs of nodes are reachable 
                                            #from one another in the directed subgraph
    
    
    if close != "True":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))
        
        # Dibujar el grafo
        pos = nx.spring_layout(G)
        #labels = nx.get_edge_attributes(G, 'weight')
        labels = {(u, v): "{:.2f}".format(data['weight']) for u, v, data in G.edges(data=True)}
        nx.draw(G, pos, with_labels=True, node_size = 300, node_color="skyblue", font_size=11, font_color="black", arrows=True, ax = ax1)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax = ax1)
        
        # Dibujar el subgrafo
        pos = nx.spring_layout(LSC)
        labels = {(u, v): "{:.2f}".format(data['weight']) for u, v, data in LSC.edges(data=True)}
        nx.draw(LSC, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=11, font_color="black", arrows = True)#font_weight = "bold", edge_color='blue', width=2)
        nx.draw_networkx_edge_labels(LSC, pos, edge_labels=labels, ax = ax2)
        plt.show()

    networks = [G, LSC]
    
    atributos = ['diametro', 'num_edges', 'num_nodes', 'selfloops', 'adjacency_matrix', 'L2', 'L3', 'k_mean', 'density', 'ASP', 'average_CC', 'transitivity', 'nro_comunidades', 'centralidad_grado', 'centralidad_betweenes', 'centralidad_autoval', 'centralidad_cercania']


    list_atributos = {nombre: [] for nombre in atributos}
    
    list_atributos['diametro'].append('nan')
    list_atributos['ASP'].append('nan')
    for i, red in enumerate(networks):
        if i != 0:
            list_atributos['diametro'].append(nx.diameter(red, weight='weight'))
            list_atributos['ASP'].append(nx.average_shortest_path_length(red, weight='weight'))
        list_atributos['num_edges'].append(red.number_of_edges())
        list_atributos['num_nodes'].append(red.number_of_nodes())
        list_atributos['selfloops'].append(len(list(nx.nodes_with_selfloops(red))))
        list_atributos['adjacency_matrix'].append(nx.attr_matrix(red)[0])
        adjacency_matrix_squared = np.dot(list_atributos['adjacency_matrix'][i], list_atributos['adjacency_matrix'][i])
        adjacency_matrix_3 = np.dot(adjacency_matrix_squared , list_atributos['adjacency_matrix'][i])  
        trace_of_squared_adjacency = np.trace(adjacency_matrix_squared)
        trace_of_3_adjacency = np.trace(adjacency_matrix_3)
        list_atributos['L2'].append(trace_of_squared_adjacency / 2)
        list_atributos['L3'].append(trace_of_3_adjacency / 3) 
        N = list_atributos['num_nodes'][i]
        E = list_atributos['num_edges'][i]
        list_atributos['k_mean'].append(2*E/N)
        list_atributos['density'].append(E/(N*(N-1)))
        list_atributos['average_CC'].append(nx.average_clustering(red))
        list_atributos['transitivity'].append(nx.transitivity(red))
        # Infomap (comunidades)
        G_ig = ig.Graph.from_networkx(red)
        com_ip = G_ig.community_infomap() #si pones print(com_ip) te dice por qué nodos esta compuesta cada comunidad, el len(com_ip) te da el número de comunidades
        list_atributos['nro_comunidades'].append(len(com_ip))
        #centralidades
        diccionario_centralidad_grado = nx.degree_centrality(red)
        diccionario_centralidad_intermediatez = nx.betweenness_centrality(red, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        diccionario_centralidad_autovalor = nx.eigenvector_centrality(red, max_iter=100, tol=1e-06, nstart=None, weight='weight')
        diccionario_centralidad_cercania = nx.closeness_centrality(red, u=None, distance=None, wf_improved=True)
        list_atributos['centralidad_grado'].append(np.mean(list(diccionario_centralidad_grado.values())))
        list_atributos['centralidad_betweenes'].append(np.mean(list(diccionario_centralidad_intermediatez.values())))
        list_atributos['centralidad_autoval'].append(np.mean(list(diccionario_centralidad_autovalor.values())))
        list_atributos['centralidad_cercania'].append(np.mean(list(diccionario_centralidad_cercania.values())))
        
    if todas != False:
        rta = [sujeto+1, tema, list_atributos['num_nodes'][0], list_atributos['num_edges'][0], list_atributos['k_mean'][0], list_atributos['transitivity'][0], list_atributos['average_CC'][0], list_atributos['selfloops'][0], list_atributos['L2'][0], list_atributos['L3'][0], list_atributos['nro_comunidades'][0], list_atributos['centralidad_grado'][0], list_atributos['centralidad_betweenes'][0], list_atributos['centralidad_autoval'][0], list_atributos['centralidad_cercania'][0], list_atributos['num_nodes'][1], list_atributos['num_edges'][1], list_atributos['diametro'][1], list_atributos['k_mean'][1], list_atributos['transitivity'][1], list_atributos['ASP'][1], list_atributos['average_CC'][1], list_atributos['selfloops'][1], list_atributos['L2'][1], list_atributos['L3'][1], list_atributos['nro_comunidades'][1], list_atributos['centralidad_grado'][1], list_atributos['centralidad_betweenes'][1], list_atributos['centralidad_autoval'][1], list_atributos['centralidad_cercania'][1]]
    else:
        rta = [sujeto+1, tema, list_atributos['num_nodes'][1]/list_atributos['num_nodes'][0], list_atributos['nro_comunidades'][1], list_atributos['diametro'][1], list_atributos['k_mean'][1], list_atributos['transitivity'][1], list_atributos['ASP'][1], list_atributos['average_CC'][1], list_atributos['selfloops'][1], list_atributos['L2'][1], list_atributos['L3'][1], list_atributos['density'][1]]
    return rta

#num_edges_norm era la densidad... 'num_edges_norm'
#list_atributos['num_edges'][1]/(list_atributos['num_nodes'][1]*(list_atributos['num_nodes'][0]-1))
#%%

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

atrib_todas = ['Sujetos', 'Condición', 'num_nodes_LCC', 'num_edges_LCC', 'k_mean_LCC', 'transitivity_LCC', 'average_CC_LCC', 'selfloops_LCC', 'L2_LCC', 'L3_LCC', 'nro_comunidades_LCC', 'mean_cent_grado_LCC', 'mean_cent_betweenes_LCC', 'mean_cent_autoval_LCC', 'mean_cent_cercania_LCC',
             'num_nodes_LSC', 'num_edges_LSC', 'diametro_LSC', 'k_mean_LSC', 'transitivity_LSC', 'ASP_LSC', 'average_CC_LSC', 'selfloops_LSC', 'L2_LSC', 'L3_LSC', 'nro_comunidades_LSC', 'mean_cent_grado_LSC', 'mean_cent_betweenes_LSC', 'mean_cent_autoval_LSC', 'mean_cent_cercania_LSC']

atribu = ['Sujetos', 'Condición', 'num_nodes_norm', 'Comunidades_LSC', 'diámetro', 'k_mean', 'transitivity', 'ASP', 'average_CC', 'selfloops', 'L2', 'L3', 'density']

#%%
tema = temas[1]
sujeto = 21

rta = LSC_text_sujeto(sujeto, tema, close = False)

#%%

#si quiero guardar todas las medidas crudas (sin normalizar) pongo atributos = atrib_todas, sino atributos = atrib
#en función hay que cambiar todas = True ademas para guardar todas las medidas
atributos = atribu

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]


for j, tema in tqdm(enumerate(temas)):
    datos = []
    for sujeto in tqdm(range(30)):
        datos.append(LSC_text_sujeto(sujeto, tema))
      
    datos_dict = np.asarray(datos).T
    
    dict_datos = {}
    for i, atrib in enumerate(atributos):
        dict_datos[atrib] = datos_dict[i]
        
    df = pd.DataFrame(dict_datos)
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Atributos/df_red_norm_{tema}.csv'
    
    df.to_csv(path, index=False) 

