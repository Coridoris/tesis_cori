# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:16:46 2023

@author: Usuario
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

#%%

G = nx.Graph()

# Agregar nodos al grafo
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
G.add_nodes_from(nodes)

# Agregar enlaces con pesos
edges = [('B', 'C', {'weight': 14}),
         ('C', 'A', {'weight': 14}),
         ('D', 'E', {'weight': 1}),
         ('E', 'F', {'weight': 2}),
         ('F', 'G', {'weight': 2}),
         ('G', 'H', {'weight': 2}),
         ('H', 'A', {'weight': 14}),
         ('B', 'E', {'weight': 14}),
         ('H', 'I', {'weight': 1})]

# edges = [('B', 'C', {'weight': 7}),
#          ('C', 'A', {'weight': 7}),
#          ('D', 'E', {'weight': 1}),
#          ('E', 'F', {'weight': 1}),
#          ('F', 'G', {'weight': 1}),
#          ('G', 'H', {'weight': 1}),
#          ('H', 'A', {'weight': 7}),
#          ('B', 'E', {'weight': 7}),
#          ('H', 'I', {'weight': 1})]


G.add_edges_from(edges)

plt.figure(1), plt.clf()
# Dibujar el grafo con etiquetas de pesos
pos = nx.spring_layout(G)  # Distribución para el dibujo
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

diameter = nx.diameter(G, weight='weight')

print("Diámetro del grafo:", diameter)

# Calcular la distancia entre dos nodos (por ejemplo, 'B' y 'F') teniendo en cuenta el peso de los enlaces
source_node = 'D'
target_node = 'I'
distance = nx.shortest_path_length(G, source=source_node, target=target_node, weight='weight')

print(f"Distancia entre '{source_node}' y '{target_node}': {distance}")

#%%

G = nx.Graph()

# Agregar nodos al grafo
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
G.add_nodes_from(nodes)

# Agregar enlaces con pesos
edges = [('B', 'C', {'weight': 2.33333333333333}),
         ('C', 'A', {'weight': 2.33333333333333/2}),
         ('A', 'B', {'weight': 2.33333333333333/4}),
         ('E', 'B', {'weight': 2.33333333333333/2}),
         ('E', 'F', {'weight': 2.33333333333333}),
         ('B', 'D', {'weight': 2.33333333333333/4})]

# edges = [('B', 'C', {'weight': 7}),
#          ('C', 'A', {'weight': 7}),
#          ('D', 'E', {'weight': 1}),
#          ('E', 'F', {'weight': 1}),
#          ('F', 'G', {'weight': 1}),
#          ('G', 'H', {'weight': 1}),
#          ('H', 'A', {'weight': 7}),
#          ('B', 'E', {'weight': 7}),
#          ('H', 'I', {'weight': 1})]


G.add_edges_from(edges)

plt.figure(1), plt.clf()
# Dibujar el grafo con etiquetas de pesos
pos = nx.spring_layout(G)  # Distribución para el dibujo
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

diameter = nx.diameter(G, weight='weight')

ASP = nx.average_shortest_path_length(G, weight="weight")

print("Diámetro del grafo:", diameter)
print("ASP del grafo:", ASP)

# Calcular la distancia entre dos nodos (por ejemplo, 'B' y 'F') teniendo en cuenta el peso de los enlaces
source_node = 'A'
target_node = 'F'
distance = nx.shortest_path_length(G, source=source_node, target=target_node, weight='weight')

print(f"Distancia entre '{source_node}' y '{target_node}': {distance}")

#%%
temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

tema = temas[0]

sujeto = 18

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))

G = nx.DiGraph()

# Agregar nodos y arcos con pesos
for (palabra_a, palabra_b), peso in edge_weights_norm.items():
    G.add_edge(palabra_a, palabra_b, weight=peso)

# Dibujar el grafo
pos = nx.spring_layout(G)
#labels = nx.get_edge_attributes(G, 'weight')
labels = {(u, v): "{:.2f}".format(data['weight']) for u, v, data in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size = 300, node_color="skyblue", font_size=11, font_color="black", arrows=True, ax = ax1)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax = ax1)

# Encontrar las componentes fuertemente conexas
componentes_fuertemente_conexas = list(nx.strongly_connected_components(G))

# Seleccionar la componente fuertemente conexa más grande (puedes elegir la que desees)
componente_mas_grande = max(componentes_fuertemente_conexas, key=len)

# Crear un subgrafo con la componente fuertemente conexa
LSC = G.subgraph(componente_mas_grande) #LSC, the maximal subgraph in which all pairs of nodes are reachable 
                                        #from one another in the directed subgraph

# Dibujar el subgrafo
pos = nx.spring_layout(LSC)
#labels = nx.get_edge_attributes(LSC, 'weight')
labels = {(u, v): "{:.2f}".format(data['weight']) for u, v, data in LSC.edges(data=True)}
nx.draw(LSC, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=11, font_color="black", arrows = True)#font_weight = "bold", edge_color='blue', width=2)
nx.draw_networkx_edge_labels(LSC, pos, edge_labels=labels, ax = ax2)
plt.show()


