# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:36:58 2023

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

from sentence_transformers import SentenceTransformer, util
import os
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import random 

import nltk
nltk.download('punkt')  # Descargar el tokenizador de oraciones (solo se necesita ejecutar una vez)
from nltk.tokenize import sent_tokenize
import ast

#%% el santo trial

entrevista = 'Segunda'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

#%% FUNCIONES

def tokenize_if_not_nan(text):
    if not pd.isnull(text):
        return sent_tokenize(text)
    else:
        return np.nan

#coherencia

def coherence_evolution(text):

  sentence = sent_tokenize(text)
  
  numb_sentences = len(sentence)

  embeddings = []
  for i in range(len(sentence)):
      #Compute embedding 
      embeddings.append(model.encode(sentence[i], convert_to_tensor=True))

  coherence = []
  for j in range(1, numb_sentences):
      cosine_scores = []
      #Compute cosine-similarities
      for i in range(len(sentence)-j):
          cosine_scores.append(util.cos_sim(embeddings[i], embeddings[i+j]).item())
      
      coherence.append(np.mean(cosine_scores))
      
  return coherence


def shuffle_sentences(text):
    # Suponemos que las oraciones están separadas por puntos seguidos de un espacio o un salto de línea.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Mezclamos el orden de las oraciones.
    random.shuffle(sentences)

    # Unimos las oraciones en un texto mezclado.
    mixed_text = ' '.join(sentences)

    return mixed_text

def dividir_por_valor(row, valor_divisor):
    if type(row) == np.ndarray:
        return [x / valor_divisor for x in row]
    else:
        return np.nan

def ast_literal_eval_notnan(lista):
    if type(lista) == str:
        return ast.literal_eval(lista)
    else:
        return np.nan
    
def array_sinnans(lista):     
    if lista != np.nan:
        return np.array(lista)
    else:
        return np.nan    


def csv_coherence_evolution(tema):
    '''
    esta función le das le decis el tema del que queres que te busque la coherencia y devuelve una lista que en la 
    primer componente tiene la coherencia promedio del texto, midiendo la coherencia entre oraciones seguidas
    en la segunda tiene la coherencia promedio del texto, midiendo la coherencia entre dos oraciones separadas por una
    y asi
    '''
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
    
    coherence = []
    
    for j in tqdm(range(len(Sujetos))):
        
        if type(df_textos['texto_crudo'][j]) == str:
            coher = coherence_evolution(df_textos['texto_crudo'][j])
        else:
            coher = np.nan
        
        coherence.append(coher)
        
    df_textos['coherencia_evolucion'] = coherence
    
    df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion.csv', index = False)
    
    return 'ok'



#%% para HACER el modelo nulo --> proximas 2 celdas, SI YA LO TENES NO CORRER
'''
para comparar con un modelo nulo la idea es ver si baja la coherencia si la calculamos mezclando oraciones del 
mismo sujeto pero de diferentes textos, por ahora uso también el filler, no se si dejarlo.
'''

#separo en oraciones los textos en un dataframe y comodo en un único dataframe (una columna por tema)
data = dict()
for j in tqdm(range(len(temas))):
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{temas[j]}_crudo_y_lemm_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
    
    #textos_crudos = np.asarray(df_textos["texto_crudo"].apply(tokenize_if_not_nan))  #si no lo quiero separados en oraciones tendria que sacar el sent_tokenize
    
    textos_crudos = df_textos["texto_crudo"]
    
    data[f"texto_crudo_{temas[j]}"] = textos_crudos
        
df = pd.DataFrame(data)


#%% va a tardar 5 horas, correr de noche

#ahora necesitamos un código que agarre oraciones aleatorias de cada uno de los temas. La cantidad de oraciones
#al principio iba a ser el máximo de oraciones del sujeto, pero a veces ese número se va a la mierda.
#después pensé en el mínimo, pero a veces no llega ni a 5. Lo que se me ocurrió es tomar 5, una de cada tema, si 
#después se cambia hay que modificar esto.

# Función para elegir una oración al azar de una lista de oraciones
def elegir_n_oraciones_al_azar(texto, n = 2):
    if not pd.isnull(texto):
        oraciones = sent_tokenize(texto)
        oraciones_array = np.random.choice(oraciones, n, replace = False)
        oraciones_combinadas = ' '.join(oracion + '.' if not oracion.endswith('.') else oracion for oracion in oraciones_array)
        return oraciones_combinadas
    else:
        return ""
    

for i in tqdm(range(0,1000)): 
    df[f'texto_random_{i}'] = df.apply(lambda row: elegir_n_oraciones_al_azar(row['texto_crudo_cfk']) + " " + 
                                         elegir_n_oraciones_al_azar(row['texto_crudo_campeones_del_mundo']) + " " + 
                                         elegir_n_oraciones_al_azar(row['texto_crudo_antesdevenir'])
                                         + " " + elegir_n_oraciones_al_azar(row['texto_crudo_presencial'])
                                         + " " + elegir_n_oraciones_al_azar(row['texto_crudo_arabia']), axis=1)
    
    #primero vamos a desordenar las oraciones y después vamos a buscar la coherencia, sino siempre van e estar en este
    #orden los temas
    
    df[f'texto_random_{i}'] = df[f'texto_random_{i}'].apply(shuffle_sentences)
    
    #la calculamos la coherencia
    
    df[f'coherencia_evolucion_de_random_{i}'] = df[f'texto_random_{i}'].apply(coherence_evolution)
    
    df.index += 1
    
    df.index.name = 'Sujetos'

df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/coherencia_evolucion_random.csv')

#%% coherencia evolucion

for i in tqdm(range(len(temas))):
    tema =  temas[i]
    ok = csv_coherence_evolution(tema)

#%% renormalizar la coherencia del sujeto x la de su modelo nulo

'''
o sea la idea es redefinir la coherencia como coherencia/modelo nulo del sujeto
si da arriba de 1 fue coherente, si da por abajo no
'''

#modelo nulo

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)

modelo_nulo = []
q_modelo_nulo = []
for j in range(0, len(df_nulo)): #recorre sobre sujetos
    modelo_nulo_sujeto_j = []
    for i in range(0,1000): #recorre sobre las mil iteraciones de modelo nulo
        if j == 0:
            df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
        try:  
            modelo_nulo_sujeto_j.append(df_nulo[f'coherencia_evolucion_de_random_{i}'][j][0])
        except IndexError:
        # Manejar la excepción si el índice está fuera de rango
            pass
    modelo_nulo.append(modelo_nulo_sujeto_j)
    if not modelo_nulo_sujeto_j:  # Si la lista está vacía
        q_modelo_nulo_j = np.nan
    else:
        q_modelo_nulo_j = np.percentile(modelo_nulo_sujeto_j, 95)
    q_modelo_nulo.append(q_modelo_nulo_j)
#%%
   

for i in tqdm(range(len(temas))):
    tema =  temas[i]
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'
    
    df_coherence_evolution = pd.read_csv(path)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a vectores
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution['coherencia_evolucion'].apply(ast_literal_eval_notnan)
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution['coherencia_evolucion'].apply(array_sinnans)
    df_coherence_evolution['coherencia_evolucion_normalizada'] = None
    
    for i, row in df_coherence_evolution.iterrows():
        df_coherence_evolution.at[i, 'coherencia_evolucion_normalizada'] = dividir_por_valor(row['coherencia_evolucion'], q_modelo_nulo[i])


    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion_norm.csv'
    
    df_coherence_evolution.to_csv(path)

#%% SI SOLO QUERES GUARDAR COHERENCIA CORRER

for tema in temas:
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion_norm.csv'
    
    df_norm = pd.read_csv(path)
    
    #paso a lista el string 
    
    df_norm['coherencia_evolucion_normalizada'] = df_norm['coherencia_evolucion_normalizada'].apply(ast_literal_eval_notnan)
    
    # Truncar las listas a tres elementos y llenar con NaN si es necesario
    df_norm['coherencia_evolucion_normalizada'] = df_norm['coherencia_evolucion_normalizada'].apply(lambda x: x[:3] if (type(x) == list and len(x) >= 3) else x + [None]*(3-len(x)) if type(x) == list else np.nan)
    
    # Crear las columnas 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3' a partir de la columna existente
    df_norm[['cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3']] = df_norm['coherencia_evolucion_normalizada'].apply(pd.Series)
    
    # Cambiar el nombre
    #df_norm['Sujetos'] = df_norm['Sujeto']
    
    # Selecciona las columnas que queremos
    columnas_seleccionadas = ['Sujetos', 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3']
    
    #reescribe el dataframe
    df_norm = df_norm[columnas_seleccionadas]
    
    df_norm.insert(1, 'Condición', tema)
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion_norm_ELcsv.csv'
    
    #df_norm.to_csv(path, index=False)


#%% FUNCIONES REDES

def grafo_text_sujeto(sujeto, tema, entrevista, lemm = 'False'):
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
        
    if lemm == "True":
        
        texto = df_textos["lematizado"][sujeto+1]
        
    else:
        
        texto = df_textos["texto_crudo"][sujeto+1]
        
    if texto != np.nan:

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
    else:
        return np.nan
#%%

tema = temas[0]
sujeto = 12

grafo_text_sujeto(sujeto, tema, entrevista)

#%% componente fuertemeente conectada de la red de todo el texto o lemmatizado, llevando todo a minúscula

def LSC_text_sujeto(sujeto, tema, entrevista, lemm = 'False',  close = 'True', todas = False):
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
        
    if lemm == "True":
        
        texto = df_textos["lematizado"][sujeto]
        
    else:
        
        texto = df_textos["texto_crudo"][sujeto]
        
    if type(texto) == str:

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
            rta = [sujeto+1, tema, list_atributos['num_nodes'][1], list_atributos['nro_comunidades'][1], list_atributos['diametro'][1], list_atributos['k_mean'][1], list_atributos['transitivity'][1], list_atributos['ASP'][1], list_atributos['average_CC'][1], list_atributos['selfloops'][1], list_atributos['L2'][1], list_atributos['L3'][1], list_atributos['density'][1]]
        return rta
    else:
        if todas != False:
            rta = np.full(30, np.nan)
            rta[0] = sujeto+1
        else:
            rta = np.full(13, np.nan)
            rta[0] = sujeto+1
        return rta

#/list_atributos['num_nodes'][0] antes normalizaba el num de nodos por los de la red total, ahora no.
#num_edges_norm era la densidad... 'num_edges_norm'
#list_atributos['num_edges'][1]/(list_atributos['num_nodes'][1]*(list_atributos['num_nodes'][0]-1))
#%%

atrib_todas = ['Sujetos', 'Condición', 'num_nodes_LCC', 'num_edges_LCC', 'k_mean_LCC', 'transitivity_LCC', 'average_CC_LCC', 'selfloops_LCC', 'L2_LCC', 'L3_LCC', 'nro_comunidades_LCC', 'mean_cent_grado_LCC', 'mean_cent_betweenes_LCC', 'mean_cent_autoval_LCC', 'mean_cent_cercania_LCC',
             'num_nodes_LSC', 'num_edges_LSC', 'diametro_LSC', 'k_mean_LSC', 'transitivity_LSC', 'ASP_LSC', 'average_CC_LSC', 'selfloops_LSC', 'L2_LSC', 'L3_LSC', 'nro_comunidades_LSC', 'mean_cent_grado_LSC', 'mean_cent_betweenes_LSC', 'mean_cent_autoval_LSC', 'mean_cent_cercania_LSC']

atribu = ['Sujetos', 'Condición', 'num_nodes_LSC', 'Comunidades_LSC', 'diámetro', 'k_mean', 'transitivity', 'ASP', 'average_CC', 'selfloops', 'L2', 'L3', 'density']
#%%
tema = temas[3]
sujeto = 31

rta = LSC_text_sujeto(sujeto, tema, entrevista, close = False)

#%% CORRER SOLO SI QUERES GUARDAR LA DE REDES APARTE

#si quiero guardar todas las medidas crudas (sin normalizar) pongo atributos = atrib_todas, sino atributos = atrib
#en función hay que cambiar todas = True ademas para guardar todas las medidas
atributos = atribu

for j, tema in tqdm(enumerate(temas)):
    datos = []
    for sujeto in tqdm(range(nro_sujetos)):
        datos.append(LSC_text_sujeto(sujeto, tema, entrevista))
      
    datos_dict = np.asarray(datos).T
    
    dict_datos = {}
    for i, atrib in enumerate(atributos):
        dict_datos[atrib] = datos_dict[i]
        
    df = pd.DataFrame(dict_datos)
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Red/Atributos/df_red_norm_{tema}.csv'
    
    df.to_csv(path, index=False) 

#%% CSV de variables estructurales: 
    
atributos = atribu    


for j, tema in tqdm(enumerate(temas)):
              
    datos = []
    for sujeto in tqdm(range(nro_sujetos)):
        datos.append(LSC_text_sujeto(sujeto, tema, entrevista))
      
    datos_dict = np.asarray(datos).T
    
    dict_datos = {}
    for i, atrib in enumerate(atributos):
        dict_datos[atrib] = datos_dict[i]
        
    df_redes = pd.DataFrame(dict_datos)
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion_norm.csv'
    
    df_norm = pd.read_csv(path)
    
    #paso a lista el string 
    
    df_norm['coherencia_evolucion_normalizada'] = df_norm['coherencia_evolucion_normalizada'].apply(ast_literal_eval_notnan)
    
    # Truncar las listas a tres elementos y llenar con NaN si es necesario
    df_norm['coherencia_evolucion_normalizada'] = df_norm['coherencia_evolucion_normalizada'].apply(lambda x: x[:3] if (type(x) == list and len(x) >= 3) else x + [None]*(3-len(x)) if type(x) == list else np.nan)
    
    # Crear las columnas 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3' a partir de la columna existente
    df_norm[['cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3']] = df_norm['coherencia_evolucion_normalizada'].apply(pd.Series)
    
    # Cambiar el nombre
    #df_norm['Sujetos'] = df_norm['Sujeto']
    
    # Selecciona las columnas que queremos
    columnas_seleccionadas = ['Sujetos', 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3']
    
    #reescribe el dataframe
    df_norm = df_norm[columnas_seleccionadas]
    
    df_norm.insert(1, 'Condición', tema)
    
    #reescribo el sujetos de redes q esta raro
    
    df_redes["Sujetos"] = df_norm["Sujetos"]
    
    df_estructurales = df_norm.merge(df_redes, on=["Sujetos", "Condición"])
    
    df_estructurales.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables estructurales/variables_estructurales_{tema}.csv', index=False)    

