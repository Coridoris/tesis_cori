# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:30:30 2023

@author: corir
"""

import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_node("Sust")
G.add_node("Verb")
G.add_node("Adj")

G.add_edge("Sust", "Verb", weight=3)
G.add_edge("Verb", "Adj", weight=5)
G.add_edge("Adj", "Sust", weight=2)
G.add_edge("Sust", "Adj", weight=8)

pos = nx.circular_layout(G)  # Posiciones de los nodos en un círculo
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

#%%
# Obtener los pesos de las aristas como un diccionario de atributos
edge_weights = nx.get_edge_attributes(G, 'weight')

# Calcular el ancho de las líneas basado en los pesos de las aristas
edge_widths = [0.1 * edge_weights[edge] for edge in G.edges()]

# Dibujar la red con anchos de línea basados en los pesos
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', width=edge_widths, edge_color='gray')
plt.show()

#%%

# Obtener los pesos de las aristas como un diccionario de atributos
edge_weights = nx.get_edge_attributes(G, 'weight')

# Calcular el ancho de las líneas basado en los pesos de las aristas
edge_widths = [0.1 * edge_weights[edge] for edge in G.edges()]

# Dibujar la red con anchos de línea basados en los pesos y colores diferentes para aristas en direcciones opuestas
pos = nx.circular_layout(G)
edge_colors = ['gray' if edge[0] < edge[1] else 'blue' for edge in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', width=edge_widths, edge_color=edge_colors)
plt.show()

#%% una mierda
G = nx.DiGraph()

# Agregar nodos
G.add_node("A")
G.add_node("B")
G.add_node("C")

# Agregar aristas con pesos
G.add_edge("A", "B", weight=3)
G.add_edge("B", "C", weight=5)
G.add_edge("C", "A", weight=2)
G.add_edge("A", "C", weight=4)  # Arista en la otra dirección

# Obtener los pesos de las aristas como un diccionario de atributos
edge_weights = nx.get_edge_attributes(G, 'weight')

# Calcular el ancho de las líneas basado en los pesos de las aristas
edge_widths = [0.1 * edge_weights[edge] for edge in G.edges()]

# Dibujar la red con disposiciones ajustadas para las aristas en direcciones opuestas
pos = nx.circular_layout(G)
pos_A = {node: [x, y + 0.05 * (i % 2)] for i, (node, (x, y)) in enumerate(pos.items())}
pos_C = {node: [x + 0.05, y + 0.05 * (i % 2)] for i, (node, (x, y)) in enumerate(pos.items())}

edge_colors = ['gray' if edge[0] < edge[1] else 'blue' for edge in G.edges()]

nx.draw(G, pos_A, with_labels=True, node_size=1000, node_color='skyblue', width=edge_widths, edge_color=edge_colors)
nx.draw(G, pos_C, with_labels=False, node_size=1000, node_color='skyblue', width=edge_widths, edge_color=edge_colors)
nx.draw_networkx_edge_labels(G, pos_A, edge_labels=edge_weights)
plt.show()

#%%
# Crear una instancia del grafo dirigido
G = nx.DiGraph()

# Agregar nodos
G.add_node("A")
G.add_node("B")
G.add_node("C")

# Agregar aristas con pesos utilizando add_weighted_edges_from
weighted_edges = [("A", "B", 3), ("B", "C", 5), ("C", "A", 2), ("A", "C", 4)]
G.add_weighted_edges_from(weighted_edges)

# Obtener los pesos de las aristas como un diccionario de atributos
edge_weights = nx.get_edge_attributes(G, 'weight')

# Calcular el ancho de las líneas basado en los pesos de las aristas
edge_widths = [0.7 * edge_weights[edge] for edge in G.edges()]

# Dibujar la red con disposición circular y anchos de línea basados en los pesos
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', width=edge_widths, edge_color='gray')

# Agregar etiquetas de peso a las aristas
edge_labels = {(u, v): f"{w}" for (u, v, w) in edge_weights.keys()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()

#%%
import numpy as np
from netgraph import Graph # pip install netgraph

triangle = nx.DiGraph([('a', 'b'), ('a', 'c'), ('b', 'a'), ('c', 'b'), ('c', 'c')])

node_positions = {
    'a' : np.array([0.2, 0.2]),
    'b' : np.array([0.8, 0.2]),
    'c' : np.array([0.5, 0.8]),
}

edge_labels = {
    ('a', 'b') : 3,
    ('a', 'c') : 'Lorem ipsum',
    ('b', 'a') : 4,
    ('c', 'b') : 'dolor sit',
    ('c', 'c') : r'$\pi$'
}

edge_weights = {
    ('a', 'b') : 3,
    ('a', 'c') : 1,
    ('b', 'a') : 4,
    ('c', 'b') : 2,
    ('c', 'c') : 2
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))

Graph(triangle, node_labels=True, edge_labels=edge_labels,
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      node_layout=node_positions, edge_layout='straight',
      node_size=6, edge_width=edge_weights  , arrows=True, ax=ax1)

Graph(triangle, node_labels=True, edge_labels=edge_labels,
      edge_label_fontdict=dict(size=12, fontweight='bold'),
      node_layout=node_positions, edge_layout='curved',
      node_size=6, edge_width=4, arrows=True, ax=ax2)

plt.show()


