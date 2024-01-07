# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:14:48 2023

@author: corir
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#%%paleta de colores de la presentacion

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"
#%%

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Encuestas/Preentrevista/Encuesta de Memoria Autobiográfica (Respuestas) 222 - Respuestas de formulario 1.csv'

df = pd.read_csv(path)

edades = df["Edad"]

genero = df["Género"]

#%%

plt.figure(1), plt.clf()
plt.hist(edades, color = color_celeste, edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
plt.xlabel('Edades', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostrar el histograma
plt.show()

plt.savefig('C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/g1yg2_edad.png', transparent=True)

#%%

# Calcular la cantidad de cada género
conteo_generos = genero.value_counts()

# Personaliza los colores para los géneros
colores = [color_celeste, color_celestito, color_palido, color_gris, 'orange']  # Agrega más colores si es necesario

# Crea el gráfico de torta
plt.figure(2), plt.clf()
plt.pie(conteo_generos, labels = conteo_generos.index, colors=colores, autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': 14})

#plt.legend(conteo_generos.index, title="Géneros", fontsize=12)

# Muestra el gráfico
plt.show()

plt.savefig('C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/g1yg2_genero.png', transparent=True)