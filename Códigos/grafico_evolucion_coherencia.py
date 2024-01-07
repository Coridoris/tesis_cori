# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 02:18:03 2023

@author: corir
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import ast 
import random
import re

path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario'

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

color_violeta = "#856084"


def plot_two_histograms(data1, data2, titulo):
    # Crear una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Crear los histogramas en cada subgráfico
    ax1.hist(data1, edgecolor='black')
    ax2.hist(data2, edgecolor='black')


    # Etiquetas y título para cada subgráfico
    ax1.set_xlabel('Valores de coherencia')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Cada oración')
    ax1.legend()

    ax2.set_xlabel('Valores de coherencia')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Cada dos oraciones')
    ax2.legend()

    # Título general para los dos histogramas
    fig.suptitle(titulo, fontsize=16)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    
    #plt.savefig(path_imagenes + f'/{titulo}_hist_coherencia.png')

    # Mostrar los histogramas
    plt.show()
    
def plot_three_histograms(data1, data2, data3, titulo, title1, title2, title3, save, xlabel = None):
    # Crear una figura con dos subgráficos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # Crear los histogramas en cada subgráfico
    ax1.hist(data1, edgecolor='black')
    ax2.hist(data2, edgecolor='black')
    ax3.hist(data3, edgecolor='black')


    # Etiquetas y título para cada subgráfico
    ax1.set_xlabel('Num de sust')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title(title1)
    ax1.legend()

    ax2.set_xlabel('Num de verb')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(title2)
    ax2.legend()
    
    ax3.set_xlabel('Num de adj')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title(title3)
    ax3.legend()
    
    if xlabel != None:
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        ax3.set_xlabel(xlabel)        

    # Título general para los dos histogramas
    fig.suptitle(titulo, fontsize=16)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    
    #plt.savefig(path_imagenes + f'/{titulo}_hist_{save}.png')

    # Mostrar los histogramas
    plt.show()
    
def box_plot3(data1, ylabel, save, titulo):
    plt.figure()
    datos = {
        'Sust': data1[0],
        'Verb': data1[1],
        'Adj': data1[2]}   
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)
    
    # Título general para los dos histogramas
    plt.title(titulo, fontsize=16)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'
    

def box_plot(data1, ylabel, save):
    plt.figure()
    datos = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'Cfk': data1[2],
        'Arabia': data1[3]}   #Filler': data1[1]
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'

def box_plot5(data1, ylabel, save):
    plt.figure()
    datos = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'Cfk': data1[2],
        'Arabia': data1[3],
        'Filler': data1[4]}
    
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'


#%%
#algunas variables que necesito definir en general
temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
#%% acomodo los datos

coherence1 = []
coherence2 = []
coherence3 = []
coherence4 = []
coherence5 = []
coherence1_err = []
coherence2_err = []
coherence3_err = []
coherence4_err = []
coherence5_err = []

for i in range(len(temas)):
    tema =  temas[i]
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'
    
    df_coherence_evolution = pd.read_csv(path)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a listas
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution['coherencia_evolucion'].apply(ast.literal_eval)
    
    coherence1.append(sum(v[0] for v in df_coherence_evolution['coherencia_evolucion']) / len(df_coherence_evolution))
    valores1 = [v[0] for v in df_coherence_evolution['coherencia_evolucion']]
    coherence1_err.append(np.std(valores1)/np.sqrt(len(df_coherence_evolution)))
    coherence2.append(sum(v[1] for v in df_coherence_evolution['coherencia_evolucion']) / len(df_coherence_evolution))
    valores2 = [v[1] for v in df_coherence_evolution['coherencia_evolucion']]
    coherence2_err.append(np.std(valores2)/np.sqrt(len(df_coherence_evolution)))
    coherence3.append(sum(v[2] for v in df_coherence_evolution['coherencia_evolucion']) / len(df_coherence_evolution))
    valores3 = [v[2] for v in df_coherence_evolution['coherencia_evolucion']]
    coherence3_err.append(np.std(valores3)/np.sqrt(len(df_coherence_evolution)))
    valid_vectors4 = [v for v in df_coherence_evolution['coherencia_evolucion'] if len(v) >= 4]
    coherence4.append(sum(v[3] for v in valid_vectors4) / len(valid_vectors4))
    valores4 = [v[3] for v in valid_vectors4]
    coherence4_err.append(np.std(valores4)/np.sqrt(len(valid_vectors4)))
    valid_vectors5 = [v for v in df_coherence_evolution['coherencia_evolucion'] if len(v) >= 5]
    coherence5.append(sum(v[4] for v in valid_vectors5) / len(valid_vectors5))
    valores5 = [v[4] for v in valid_vectors5]
    coherence5_err.append(np.std(valores5)/np.sqrt(len(valid_vectors5)))
    
#modelo nulo

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)
#hice mil mezclas de textos de cada sujeto y le calculé la coherencia
coherence1_random = []
coherence2_random = []
coherence3_random = []
coherence4_random = []
coherence5_random = []

for i in range(0,1000):

    df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
    # Calcular el promedio del primer elemento de los vectores
    coherence1_random.append(sum(v[0] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    coherence2_random.append(sum(v[1] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    #coherence3_random.append(sum(v[2] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    #coherence4_random.append(sum(v[3] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    #coherence5_random.append(sum(v[4] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    
#buscamos los cuantiles

q_coherence_rand_1 = np.percentile(coherence1_random, 95)
q_coherence_rand_2 = np.percentile(coherence2_random, 95)

print("cuartil 95 cohe1rand:", q_coherence_rand_1)
print("cuartil 95 cohe2rand:", q_coherence_rand_2)
    
#%% modelo nulo, coherencia cada 1 y cada 2 oraciones

#plot_two_histograms(coherence1_random, coherence2_random, "Modelo nulo")

# Crear una figura con dos subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Crear los histogramas en cada subgráfico
ax1.hist(coherence1_random, edgecolor='black')
ax2.hist(coherence2_random, edgecolor='black')
ax1.axvline(x = coherence1[0], color='red', linestyle='--', label= temas[0])
ax2.axvline(x = coherence2[0], color='red', linestyle='--', label=temas[0])
ax1.axvline(x = coherence1[1], color='rebeccapurple', linestyle='--', label= temas[1])
ax2.axvline(x = coherence2[1], color='rebeccapurple', linestyle='--', label=temas[1])
ax1.axvline(x = coherence1[2], color='lightgreen', linestyle='--', label= temas[2])
ax2.axvline(x = coherence2[2], color='lightgreen', linestyle='--', label=temas[2])
ax1.axvline(x = coherence1[3], color='pink', linestyle='--', label= temas[3])
ax2.axvline(x = coherence2[3], color='pink', linestyle='--', label=temas[3])
ax1.axvline(x = coherence1[4], color='gold', linestyle='--', label= temas[4])
ax2.axvline(x = coherence2[4], color='gold', linestyle='--', label=temas[4])
ax1.axvline(x = q_coherence_rand_1, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')
ax2.axvline(x = q_coherence_rand_2, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')

coherence_camp = coherence1[0]
# Etiquetas y título para cada subgráfico
ax1.set_xlabel('Valores de coherencia')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Cada oración')
ax1.legend()

ax2.set_xlabel('Valores de coherencia')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Cada dos oraciones')
ax2.legend()

# Título general para los dos histogramas
fig.suptitle('Modelo nulo', fontsize=16)

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

plt.savefig(path_imagenes + '/ModeloNulo_hist_coherencia.png')

# Mostrar los histogramas
plt.show()

#%% modelo nulo coherencia cada 3, cada 4 y cada 5 oraciones

#plot_two_histograms(coherence1_random, coherence2_random, "Modelo nulo")

# Crear una figura con dos subgráficos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

# Crear los histogramas en cada subgráfico
ax1.hist(coherence1_random, edgecolor='black')
ax2.hist(coherence1_random, edgecolor='black')
ax3.hist(coherence1_random, edgecolor='black')

ax1.axvline(x = coherence3[0], color='red', linestyle='--', label= temas[0])
ax2.axvline(x = coherence4[0], color='red', linestyle='--', label=temas[0])
ax3.axvline(x = coherence5[0], color='red', linestyle='--', label=temas[0])

ax1.axvline(x = coherence3[1], color='rebeccapurple', linestyle='--', label= temas[1])
ax2.axvline(x = coherence4[1], color='rebeccapurple', linestyle='--', label=temas[1])
ax3.axvline(x = coherence5[1], color='rebeccapurple', linestyle='--', label=temas[1])

ax1.axvline(x = coherence3[2], color='lightgreen', linestyle='--', label= temas[2])
ax2.axvline(x = coherence4[2], color='lightgreen', linestyle='--', label=temas[2])
ax3.axvline(x = coherence5[2], color='lightgreen', linestyle='--', label=temas[2])

ax1.axvline(x = coherence3[3], color='pink', linestyle='--', label= temas[3])
ax2.axvline(x = coherence4[3], color='pink', linestyle='--', label=temas[3])
ax3.axvline(x = coherence5[3], color='pink', linestyle='--', label=temas[3])

ax1.axvline(x = coherence3[4], color='gold', linestyle='--', label= temas[4])
ax2.axvline(x = coherence4[4], color='gold', linestyle='--', label=temas[4])
ax3.axvline(x = coherence5[4], color='gold', linestyle='--', label=temas[4])

ax1.axvline(x = q_coherence_rand_1, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')
ax2.axvline(x = q_coherence_rand_1, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')
ax3.axvline(x = q_coherence_rand_1, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')

# Etiquetas y título para cada subgráfico
ax1.set_xlabel('Valores de coherencia')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Cada tres oreciones')
ax1.legend()

ax2.set_xlabel('Valores de coherencia')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Cada cuatro oraciones')
ax2.legend()

ax3.set_xlabel('Valores de coherencia')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Cada cinco oraciones')
ax3.legend()

# Título general para los dos histogramas
fig.suptitle('Modelo nulo', fontsize=16)

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

plt.savefig(path_imagenes + '/ModeloNulo_hist_coherencia_345.png')

# Mostrar los histogramas
plt.show()

#%% una mejor forma de ver esto grafico coherencia vs cada x cantidad de oraciones

# Crear una figura con dos subgráficos
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

temas_label = ["Campeones", "Presencial", "CFK", "Arabia", "Filler"]

color_campeones = color_celeste
color_presencial = color_celestito
color_cfk = color_palido
color_arabia = color_violeta
color_filler = color_gris

color_violeta_fuerte = "#6A4480"

t = np.linspace(1, 5, 5)
coherence_por_cant_oraciones = np.array([np.array(coherence1), np.array(coherence2), np.array(coherence3), np.array(coherence4), np.array(coherence5)])
#coherence1 es un vector que en el primer elemento tiene la coherencia entre oraciones contiguas de 
#para el tema 0, en el segundo elemento para el tema 1, en el tercero para el tema 2 y asi
coherence_por_tema = np.transpose(coherence_por_cant_oraciones)
#esto debería tener en el primer elemento un vector que tenga la coherencia en los distintos t del tema 0
# Crear los histogramas en cada subgráfico
ax.scatter(t, coherence_por_tema[0], s = 100, color = color_celeste, label = temas_label[0])
ax.scatter(t, coherence_por_tema[1], s = 100, color = color_celestito, label = temas_label[1])
ax.scatter(t, coherence_por_tema[2], s = 100, color = color_palido, edgecolors = 'k', label = temas_label[2], zorder = 5) 
ax.scatter(t, coherence_por_tema[3], s = 100, color = color_violeta_fuerte, label = temas_label[3])
ax.scatter(t, coherence_por_tema[4], s = 100, color = color_gris, label = temas_label[4])


plt.errorbar(t, coherence_por_tema[0], yerr = coherence1_err, fmt='o', color = color_celeste, linewidth =3)
plt.errorbar(t, coherence_por_tema[1], yerr = coherence2_err, fmt='o', color = color_celestito, linewidth = 3)
plt.errorbar(t, coherence_por_tema[2], yerr = coherence3_err, fmt='o', color = 'k', linewidth = 3, zorder = 1)
plt.errorbar(t, coherence_por_tema[3], yerr = coherence4_err, fmt='o', color = color_violeta_fuerte, linewidth = 3)
plt.errorbar(t, coherence_por_tema[4], yerr = coherence5_err, fmt='o', color = color_gris, linewidth = 3)


ax.axhline(y = q_coherence_rand_1, color='red', linestyle='--', label= 'Modelo nulo')


# Etiquetas y título para cada subgráfico
ax.set_xlabel('Distancia', fontsize = 18)
ax.set_xticks([1,2,3,4,5], )
ax.tick_params(labelsize=15)
ax.set_ylabel('Coherencia', fontsize = 18)
plt.grid(True)

ax.legend(loc='upper right',fontsize = 15, bbox_to_anchor=(1.5, 1.0))


# Título general para los dos histogramas
#fig.suptitle('Coherencia en el tiempo', fontsize=16)

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

plt.savefig(path_imagenes + '/coherencia_vs_t_conmodelonulo.png', transparent = True)

# Mostrar los histogramas
plt.show()
    
#%% ver la coherencia por tema por sujeto

coherence1 = []
coherence2 = []
coherence3 = []
coherence4 = []
coherence5 = []

for i in range(len(temas)):
    tema =  temas[i]
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'
    
    df_coherence_evolution = pd.read_csv(path)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a listas
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution['coherencia_evolucion'].apply(ast.literal_eval)
    
    coherence1.append([v[0] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence2.append([v[1] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence3.append([v[2] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence4_tema = []
    for v in df_coherence_evolution['coherencia_evolucion']:
        if len(v) >= 4:
            coherence4_tema.append(v[3])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence4_tema.append(np.nan)
    coherence4.append(coherence4_tema)
    coherence5_tema = []
    for v in df_coherence_evolution['coherencia_evolucion']:
        if len(v) >= 5:
            coherence5_tema.append(v[4])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence5_tema.append(np.nan)
    coherence5.append(coherence5_tema)

#modelo nulo

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)

modelo_nulo = []
q_modelo_nulo = []
for j in range(0, len(df_nulo)): #recorre sobre sujetos
    modelo_nulo_sujeto_j = []
    for i in range(0,1000): #recorre sobre las mil iteraciones de modelo nulo
        if j == 0 :
            df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
        modelo_nulo_sujeto_j.append(df_nulo[f'coherencia_evolucion_de_random_{i}'][j][0])
    modelo_nulo.append(modelo_nulo_sujeto_j)
    q_modelo_nulo_j  = np.percentile(modelo_nulo_sujeto_j, 95)
    q_modelo_nulo.append(q_modelo_nulo_j)

nro_inicial_sujeto = 5
nro_sujetos = 10
colors = np.linspace(0.1, 1, nro_sujetos-nro_inicial_sujeto)  # Genera valores de color entre 0 y 1
vector_de_unos = np.ones( nro_sujetos-nro_inicial_sujeto)
vector_de_dos = np.full( nro_sujetos-nro_inicial_sujeto, 2)
vector_de_tres = np.full( nro_sujetos-nro_inicial_sujeto, 3)
vector_de_cuatros = np.full( nro_sujetos-nro_inicial_sujeto, 4)
vector_de_cinco = np.full( nro_sujetos-nro_inicial_sujeto, 5)
# Crear un scatter plot
plt.figure()
plt.scatter(vector_de_unos, coherence1[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab20')
plt.scatter(vector_de_dos, coherence2[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab20')
plt.scatter(vector_de_tres, coherence3[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab20')
plt.scatter(vector_de_cuatros, coherence4[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab20')
plt.scatter(vector_de_cinco, coherence5[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab20')


q_nulo = q_modelo_nulo[0:nro_sujetos]
    
# Generar un colormap "tab20" y las posiciones de colores
cmap = plt.get_cmap("tab20")
color_positions = np.linspace(0.1, 1, nro_sujetos)

# Dibujar líneas horizontales con colores del colormap
for i, position in enumerate(q_nulo):
    color_index = i % len(color_positions)  # Ciclar a través de las posiciones de colores
    color = cmap(color_positions[color_index])
    plt.hlines(y = position, xmin = 0.5, xmax = 5.5, color = color, linestyle='--')



ax.set_xlabel('Temas')
ax.set_ylabel('Coherencia')
plt.show()

#for i in range(len(nro_sujetos)):
#    ax.axhline(y = q_coherence_rand_1, color='red', linestyle='--', label= 'Modelo nulo')


#%%  ver la coherencia por tema por sujeto

'''
esta es la nomeclatura
coherence{cada_cuantas_oraciones}[nro_tema][nro_sujeto]
q_modelo_nulo[nro_sujeto]
'''   


coherence1 = []
coherence2 = []
coherence3 = []
coherence4 = []
coherence5 = []

coherence1_norm = []
coherence2_norm = []
coherence3_norm = []
coherence4_norm = []
coherence5_norm = []

for i in range(len(temas)):
    tema =  temas[i]
    path1 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'
    path2 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion_norm.csv'
    
    df_coherence_evolution = pd.read_csv(path2)
    df_coherence_evolution_ = pd.read_csv(path1)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a listas
    df_coherence_evolution_['coherencia_evolucion'] = df_coherence_evolution_['coherencia_evolucion'].apply(ast.literal_eval)
    df_coherence_evolution['coherencia_evolucion_normalizada'] = df_coherence_evolution['coherencia_evolucion_normalizada'].apply(ast.literal_eval)
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution_['coherencia_evolucion']
  
    coherence1.append([v[0] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence2.append([v[1] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence3.append([v[2] for v in df_coherence_evolution['coherencia_evolucion']])
    coherence4_tema = []
    for v in df_coherence_evolution['coherencia_evolucion']:
        if len(v) >= 4:
            coherence4_tema.append(v[3])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence4_tema.append(np.nan)
    coherence4.append(coherence4_tema)
    coherence5_tema = []
    for v in df_coherence_evolution['coherencia_evolucion']:
        if len(v) >= 5:
            coherence5_tema.append(v[4])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence5_tema.append(np.nan)
    coherence5.append(coherence5_tema)

    coherence1_norm.append([v[0] for v in df_coherence_evolution['coherencia_evolucion_normalizada']])
    coherence2_norm.append([v[1] for v in df_coherence_evolution['coherencia_evolucion_normalizada']])
    coherence3_norm.append([v[2] for v in df_coherence_evolution['coherencia_evolucion_normalizada']])
    coherence4_tema_norm = []
    for v in df_coherence_evolution['coherencia_evolucion_normalizada']:
        if len(v) >= 4:
            coherence4_tema_norm.append(v[3])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence4_tema_norm.append(np.nan)
    coherence4_norm.append(coherence4_tema_norm)
    coherence5_tema = []
    for v in df_coherence_evolution['coherencia_evolucion_normalizada']:
        if len(v) >= 5:
            coherence5_tema.append(v[4])
        else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
            coherence5_tema.append(np.nan)
    coherence5_norm.append(coherence5_tema)
    
#modelo nulo

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)

modelo_nulo = []
q_modelo_nulo = []
for j in range(0, len(df_nulo)): #recorre sobre sujetos
    modelo_nulo_sujeto_j = []
    for i in range(0,1000): #recorre sobre las mil iteraciones de modelo nulo
        if j == 0:
            df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
        modelo_nulo_sujeto_j.append(df_nulo[f'coherencia_evolucion_de_random_{i}'][j][0])
    modelo_nulo.append(modelo_nulo_sujeto_j)
    q_modelo_nulo_j  = np.percentile(modelo_nulo_sujeto_j, 95)
    q_modelo_nulo.append(q_modelo_nulo_j)
    
nro_inicial_sujeto = 4
nro_sujetos = 5
colors = np.linspace(0.1, 1, nro_sujetos-nro_inicial_sujeto)  # Genera valores de color entre 0 y 1
vector_de_unos = np.ones( nro_sujetos-nro_inicial_sujeto)
vector_de_dos = np.full( nro_sujetos-nro_inicial_sujeto, 2)
vector_de_tres = np.full( nro_sujetos-nro_inicial_sujeto, 3)
vector_de_cuatros = np.full( nro_sujetos-nro_inicial_sujeto, 4)
vector_de_cinco = np.full( nro_sujetos-nro_inicial_sujeto, 5)
# Crear un scatter plot
plt.figure()
plt.scatter(vector_de_unos, coherence1[0][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab10') #coherencia del primer tema (cada una oracion)
plt.scatter(vector_de_dos, coherence1[1][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab10') #coherencia del segundo tema 
plt.scatter(vector_de_tres, coherence1[2][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab10')
plt.scatter(vector_de_cuatros, coherence1[3][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab10')
plt.scatter(vector_de_cinco, coherence1[4][nro_inicial_sujeto:nro_sujetos], c=colors, cmap='tab10')


q_nulo = q_modelo_nulo[nro_inicial_sujeto:nro_sujetos]
    
# Generar un colormap "tab20" y las posiciones de colores
cmap = plt.get_cmap("tab10")
color_positions = np.linspace(0, 1, nro_sujetos)

# Dibujar líneas horizontales con colores del colormap
for i, position in enumerate(q_nulo):
    color_index = i % len(color_positions)  # Ciclar a través de las posiciones de colores
    color = cmap(color_positions[color_index])
    plt.hlines(y = position, xmin = 0.5, xmax = 5.5, color = color, linestyle='--')



plt.xlabel('Temas')
plt.ylabel('Coherencia')
plt.show()

#%% grafico de coherencia normalizada por sujeto en función de los temas

#de un único sujeto
sujeto = 1
temas_ticks = ["Camp", "Pres", "CFK", "Arabia", "Filler"]
# Crear un scatter plot
x = np.array([1,2,3,4,5])
y = np.array([coherence1_norm[0][sujeto], coherence1_norm[1][sujeto], coherence1_norm[2][sujeto], coherence1_norm[3][sujeto], coherence1_norm[4][sujeto]])
plt.figure()
plt.scatter
plt.scatter(x, y) 
plt.hlines(y = 1, xmin = 0.5, xmax = 5.5, color = color, linestyle='--')
plt.xticks(x, temas_ticks)
plt.xlabel('Temas')
plt.ylabel('Coherencia')
plt.show()

#%% gráfico de varios sujetos

nro_inicial_sujeto = 0
nro_sujetos = 30
colors = np.linspace(0.1, 1, nro_sujetos-nro_inicial_sujeto)  # Genera valores de color entre 0 y 1
colores_tab20 = plt.cm.get_cmap('tab20', 20)
colores_tab20b = plt.cm.get_cmap('tab20b', 20)

# Combina los colores de ambas paletas
colors = [colores_tab20(i) for i in range(20)] + [colores_tab20b(i) for i in range(10)]

vector_de_unos = np.ones( nro_sujetos-nro_inicial_sujeto)
vector_de_dos = np.full( nro_sujetos-nro_inicial_sujeto, 2)
vector_de_tres = np.full( nro_sujetos-nro_inicial_sujeto, 3)
vector_de_cuatros = np.full( nro_sujetos-nro_inicial_sujeto, 4)
vector_de_cinco = np.full( nro_sujetos-nro_inicial_sujeto, 5)
# Crear un scatter plot
plt.figure()
plt.scatter(vector_de_unos, coherence1_norm[0][nro_inicial_sujeto:nro_sujetos],s=50, c=colors)#, cmap='tab10') #coherencia del primer tema (cada una oracion)
plt.scatter(vector_de_dos, coherence1_norm[1][nro_inicial_sujeto:nro_sujetos],s=50, c=colors)#, cmap='tab10') #coherencia del segundo tema 
plt.scatter(vector_de_tres, coherence1_norm[2][nro_inicial_sujeto:nro_sujetos],s=50, c=colors)#, cmap='tab10')
plt.scatter(vector_de_cuatros, coherence1_norm[3][nro_inicial_sujeto:nro_sujetos],s=50, c=colors)#, cmap='tab10')
#plt.scatter(vector_de_cinco, coherence1_norm[4][nro_inicial_sujeto:nro_sujetos],s=50, c=colors, cmap='tab10')

plt.hlines(y = 1, xmin = 0.5, xmax = 4.5, color ="black", linestyle='--')

plt.xticks(x[0:4], temas_ticks[0:4])
plt.xlabel('Temas', fontsize = 18)
plt.ylabel('Coherencia', fontsize = 18)
plt.tick_params(labelsize=15)

plt.savefig(path_imagenes + '/coherencia_por_tema_por_sujeto_21a30.png')
plt.show()

        
#%% box plot a d = 1

import matplotlib as npl

temas_ticks = ["Camp", "Pres", "CFK", "Arabia", "Filler"]


color_campeones = color_celeste
color_presencial = color_celestito
color_cfk = color_palido
color_arabia = color_violeta
color_filler = color_gris

color_violeta_fuerte = "#6A4480"


coherence1_norm[0] #campeones

coherence1_norm[1] #presencial

coherence1_norm[2] #cfk

coherence1_norm[3] #arabia

coherence1_norm[4] #filler

ylabel = "Coherencia normalizada a d = 1"

nombre_guardado = "coherencia_dist1_temas_seminario"

npl.rcParams['figure.figsize'] = [10, 15]
npl.rcParams["axes.labelsize"] = 16
npl.rcParams['xtick.labelsize'] = 16
npl.rcParams['ytick.labelsize'] = 16
data = {
    'Campeones': coherence1_norm[0],
    'Presencial': coherence1_norm[1],
    'CFK': coherence1_norm[2],
    'Arabia': coherence1_norm[3],
   'Filler': coherence1_norm[4]}

df_ = pd.DataFrame(data)

colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]
#plt.figure(figsize=(15,10)), plt.clf()
# Crear el violin plot utilizando catplot
catplot = sns.catplot(data=df_, kind='violin', palette=colors, height=6, aspect=1.5)

catplot.set_ylabels(ylabel)

plt.hlines(y = 1, xmin = -0.5, xmax = 4.5, color ="black", linestyle='--', zorder = -1)

# Mostrar el gráfico
plt.show()

plt.gcf().subplots_adjust(bottom=0.1)
plt.subplots_adjust(left=0.1) 

plt.savefig(path_imagenes+'/' + nombre_guardado +'violin', transparent=True)

#%%
# Crear un boxplot en lugar de un violin plot
plt.figure(figsize=(10, 12))

df = pd.DataFrame(data)

colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]

sns.boxplot(data=df, palette=colors)

plt.ylabel(ylabel, fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.ylim((0.66, 1.15))

# Agregar línea de referencia
plt.hlines(y=1, xmin=-0.25, xmax=len(df.columns)-0.45, color="black", linestyle='--', zorder = -1)

plt.subplots_adjust(bottom=0.1)

# Guardar la figura
plt.savefig(path_imagenes + '/' + nombre_guardado+ 'boxplot', transparent=True)
    
#%% 
# Crear una figura con dos subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axvline(x =q_modelo_nulo[13], color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')
ax2.axvline(x = q_modelo_nulo[14], color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')

ax1.hist(modelo_nulo[13], edgecolor='black')
ax2.hist(modelo_nulo[14], edgecolor='black')

ax1.axvline(x = coherence1[0][13], color='red', linestyle='--', label= temas[0])
ax2.axvline(x = coherence1[0][14], color='red', linestyle='--', label=temas[0])
ax1.axvline(x = coherence1[1][13], color='rebeccapurple', linestyle='--', label= temas[1])
ax2.axvline(x = coherence1[1][14], color='rebeccapurple', linestyle='--', label=temas[1])
ax1.axvline(x = coherence1[2][13], color='lightgreen', linestyle='--', label= temas[2])
ax2.axvline(x = coherence1[2][14], color='lightgreen', linestyle='--', label=temas[2])
ax1.axvline(x = coherence1[3][13], color='pink', linestyle='--', label= temas[3])
ax2.axvline(x = coherence1[3][14], color='pink', linestyle='--', label=temas[3])
ax1.axvline(x = coherence1[4][13], color='gold', linestyle='--', label= temas[4])
ax2.axvline(x = coherence1[4][14], color='gold', linestyle='--', label=temas[4])

plt.show()

#%% modelo nulo promedio y el de un sujeto juntos



# Crear una figura con dos subgráficos
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))

ax1.axvline(x =q_modelo_nulo[12], color='black', linestyle='--', linewidth = 4.5)
ax1.axvline(x = q_coherence_rand_1, color='black', linestyle='-', linewidth = 4.5, label = 'cuantil 95%')

ax1.hist(modelo_nulo[12], edgecolor='black', color = "mediumseagreen", alpha=0.5, label = "Modelo nulo de un sujeto")
ax1.hist(coherence1_random, edgecolor='black', color = "sandybrown", alpha=0.5, label = "Modelo nulo promedio")
ax1.tick_params(labelsize =15)

ax1.axvline(x = coherence1[0][12], color='midnightblue', linestyle='--', linewidth = 4.5)

ax1.axvline(x = coherence_camp , color='midnightblue', linestyle='-', linewidth = 4.5, label= "Coherencia campeones")
plt.ylabel("Frecuencia", fontsize = 18)
plt.xlabel("Coherencia", fontsize = 18)
plt.legend(fontsize=15)

plt.show()

#%% coherencia en el t para cada sujeto 

sujeto = 4

coherence_en_t = []
coherence = [coherence1, coherence2, coherence3, coherence4, coherence5]
for j, tema in enumerate(temas):
    coherence_en_t_por_tema =  []
    for i in range(len(coherence)):
        coherence_en_t_por_tema.append(coherence[i][j][sujeto])
        
    coherence_en_t.append(coherence_en_t_por_tema)

plt.figure()
t = np.linspace(1, 5, 5)
for i, tema in enumerate(temas):
    plt.scatter(t, coherence_en_t[i], label = tema)
# Colocar la leyenda fuera de la figura
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Tiempo')
plt.ylabel('Coherencia')
plt.show()

#%%contador de cuantas veces un sujeto supera su umbral y en que tema

coherence1 

count = 0
count_camp = 0
sujetos_camp = []
count_pres = 0
sujetos_pres = []
count_cfk = 0
sujetos_cfk = []
count_arab = 0
sujetos_arab = []
count_antes = 0
sujetos_antes = []
sujeto_coherente = []
for j in range(len(Sujetos)):
    for i in range(len(temas)):
        if coherence1[i][j] > q_modelo_nulo[j]:
            count += 1
            sujeto_coherente.append(j)
            if i == 0:
                count_camp += 1
                sujetos_camp.append(j)
            elif i == 1:
                count_pres += 1
                sujetos_pres.append(j)
            elif i == 2:
                count_cfk += 1
                sujetos_cfk.append(j)
            elif i == 3:
                count_arab += 1
                sujetos_arab.append(j)
            elif i == 4:
                count_antes += 1
                sujetos_antes.append(j)
                
            
count_tema = {"campeones_del_mundo": count_camp, "presencial": count_pres, "cfk": count_cfk, "arabia": count_arab, "antesdevenir": count_antes}

sujetos_tema = {"campeones_del_mundo": sujetos_camp, "presencial": sujetos_pres, "cfk": sujetos_cfk, "arabia": sujetos_arab, "antesdevenir": sujetos_antes}

#%% por consistencia veo que normalizado de lo mismo --> da lo mismo 

coherence1 

count = 0
count_camp = 0
sujetos_camp = []
count_pres = 0
sujetos_pres = []
count_cfk = 0
sujetos_cfk = []
count_arab = 0
sujetos_arab = []
count_antes = 0
sujetos_antes = []
sujeto_coherente = []
for j in range(len(Sujetos)):
    for i in range(len(temas)):
        if coherence1_norm[i][j] > 1:
            count += 1
            sujeto_coherente.append(j)
            if i == 0:
                count_camp += 1
                sujetos_camp.append(j)
            elif i == 1:
                count_pres += 1
                sujetos_pres.append(j)
            elif i == 2:
                count_cfk += 1
                sujetos_cfk.append(j)
            elif i == 3:
                count_arab += 1
                sujetos_arab.append(j)
            elif i == 4:
                count_antes += 1
                sujetos_antes.append(j)
                
            
count_tema_norm = {"campeones_del_mundo": count_camp, "presencial": count_pres, "cfk": count_cfk, "arabia": count_arab, "antesdevenir": count_antes}

sujetos_tema_norm = {"campeones_del_mundo": sujetos_camp, "presencial": sujetos_pres, "cfk": sujetos_cfk, "arabia": sujetos_arab, "antesdevenir": sujetos_antes}

