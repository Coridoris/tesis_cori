# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:36:28 2023

@author: corir
"""
#%% librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as npl

#%%colores y cosas imagenes
path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario'

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

color_violeta = "#856084"

color_campeones = color_celeste
color_presencial = color_celestito
color_cfk = color_palido
color_arabia = color_violeta
color_filler = color_gris

color_violeta_fuerte = "#6A4480"

#%%

#algunas variables que necesito definir en general
temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"]#, "antesdevenir"]

temas_labels = ["Campeones", "Presencial", "CFK", "Arabia"]#, "Filler"]

Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    

df_del_tema = []

for tema in temas:    

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{tema}.csv'
    
    df_del_tema.append(pd.read_csv(path))
    
def box_plot4(data1, ylabel, save):
    
    npl.rcParams['figure.figsize'] = [10, 15]
    npl.rcParams["axes.labelsize"] = 20
    npl.rcParams['xtick.labelsize'] = 20
    npl.rcParams['ytick.labelsize'] = 20
    
    plt.figure()
    datos = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'CFK': data1[2],
        'Arabia': data1[3]}
    
    
    df = pd.DataFrame.from_dict(datos)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df, palette = colors)
    
    plt.ylabel(ylabel)#, fontsize = 15)
    
    #plt.xticks(fontsize=12) 
    #plt.yticks(fontsize=12) 

    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = True)
    
    #plt.close()
    
    return 'ok'


# temas_ticks = ["Camp", "Pres", "CFK", "Arabia", "Filler"]


# catplot = sns.catplot(data=df_, kind='violin', palette=colors, height=6, aspect=1.5)

# catplot.set_ylabels(ylabel)

# plt.hlines(y = 1, xmin = -0.5, xmax = 4.5, color ="black", linestyle='--', zorder = -1)

# # Mostrar el gráfico
# plt.show()

# plt.gcf().subplots_adjust(bottom=0.1)
# plt.subplots_adjust(left=0.1) 

# plt.savefig(path_imagenes+'/' + nombre_guardado +'violin', transparent=True)
    
#%%
    
palabras_unicas = []
nro_sust = []
nro_adj = []
nro_verb = []
nro_num = []
nro_adv = []
nro_propn = []
detalles_int = []
detalles_ext = []
total_wc = []
comunidades = []
L1 = []
L2 = []
L3 = []
densidad = []
diametro = []
grado_medio = []
nro_enlaces = []
asp = []

for i, tema in enumerate(temas_labels):
    #palabras_unicas.append(df_del_tema[i]['Nro palabras únicas'])
    #nro_sust.append(df_del_tema[i]['Nro sust'])
    #nro_verb.append(df_del_tema[i]['Nro verb'])
    #nro_adj.append(df_del_tema[i]['Nro adj'])
    #nro_num.append(df_del_tema[i]['Nro numeral'])
    #nro_propn.append(df_del_tema[i]['Nro propn'])
    #nro_adv.append(df_del_tema[i]['Nro advs'])
    detalles_int.append(df_del_tema[i]['Detalles internos norm'])
    detalles_ext.append(df_del_tema[i]['Detalles externos norm'])
    #detalles_ext.append(df_del_tema[i]['Detalles externos']/df_del_tema[i]['Total word count ruben'])
    comunidades.append(df_del_tema[i]['Comunidades_LSC'])
    L1.append(df_del_tema[i]['selfloops'])
    L2.append(df_del_tema[i]['L2'])
    L3.append(df_del_tema[i]['L3'])
    densidad.append(df_del_tema[i]['density'])
    diametro.append(df_del_tema[i]['diámetro'])
    grado_medio.append(df_del_tema[i]['k_mean'])
    nro_enlaces.append(df_del_tema[i]['num_edges_norm'])
    asp.append(df_del_tema[i]['ASP'])
    
    
#%%

box_plot4(palabras_unicas, '$Z_{score}$ del núm de palabras', 'nro_palabras_unicas_boxplot')
#%%
box_plot4(diametro, 'Diámetro', 'red_diametro')
box_plot4(grado_medio, 'Grado medio', 'red_grado_medio')

#%%
box_plot4(comunidades, 'comunidades', 'red_comunidades')
box_plot4(L1, 'L1', 'red_L1')
box_plot4(L2, 'L2', 'red_L2')
box_plot4(L3, 'L3', 'red_L3')
box_plot4(densidad, 'densidad', 'red_densidad')

box_plot4(nro_enlaces, 'nro enlaces', 'red_enlaces')
box_plot4(asp, 'ASP', 'red_ASP')

#%%

box_plot4(nro_sust, 'Núm. de sust', 'nro_sust')
box_plot4(nro_verb, 'Núm. de verb', 'nro_verb')
box_plot4(nro_adj, 'Núm. de adj', 'nro_adj')
box_plot4(nro_num, 'Núm. de numeral', 'nro_num')
box_plot4(nro_adv, 'Núm. de adv', 'nro_adv')
box_plot4(nro_propn, 'Núm. de nomb. prop.', 'nro_propn')

import matplotlib.image as mpimg

# Crear una figura con 2 filas y 3 columnas de subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Rutas de las imágenes guardadas
rutas_imagenes = ['/nro_sust', '/nro_verb', '/nro_adj', "/nro_num", "/nro_adv", "/nro_propn"]

# Iterar a través de las rutas de las imágenes y mostrarlas en los subplots
for i, ruta in enumerate(rutas_imagenes):
    img = mpimg.imread(path_imagenes + ruta + '_boxplot.png')  # Cargar la imagen
    fila = i // 3  # Calcular la fila del subplot
    columna = i % 3  # Calcular la columna del subplot
    axs[fila, columna].imshow(img)
    axs[fila, columna].axis('off')  # Desactivar ejes
    #axs[fila, columna].set_title(f'{categoria[i]}')  # Establecer título

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar la figura con las imágenes
plt.show()



#%%

colores = ["#13505B", "#9FA0A4"]

detalles_int_x_tema = np.mean(detalles_int, axis = 1)
detalles_ext_x_tema = np.mean(detalles_ext, axis = 1)

detalles_int_x_tema_std = np.std(detalles_int, axis = 1)/np.sqrt(len(detalles_int))
detalles_ext_x_tema_std = np.std(detalles_ext, axis = 1)/np.sqrt(len(detalles_int))

condicion = ("Campeones", "Presencial", "CFK", "Arabia")#, "Filler")
detalles = {
    'Detalles internos': tuple(detalles_int_x_tema),
    'Detalles externos': tuple(detalles_ext_x_tema),
}

x = np.arange(len(condicion))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (11,6))

# for (attribute, measurement, std_dev) in zip(detalles.keys(), detalles.values(), [detalles_int_x_tema_std, detalles_ext_x_tema_std]):
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=std_dev, capsize=5)
#     multiplier += 1
    
for i, (attribute, measurement, std_dev, color) in enumerate(zip(detalles.keys(), detalles.values(), [detalles_int_x_tema_std, detalles_ext_x_tema_std], colores)):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=std_dev, capsize=5, color=color)  # Asigna el color personalizado
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Número de detalles normalizado', fontsize =15)
ax.set_xticks(x + width, condicion)
ax.tick_params(labelsize = 15)
ax.legend(loc='upper right', fontsize = 13)
plt.savefig(path_imagenes+'/' + 'detalles_int_ext', transparent=True)
plt.show()
#%%

edades1 = np.concatenate((np.ones(28)*24, np.ones(10)*23))
edades2= np.concatenate((np.ones(5)*26, np.array([25,25, 27,27,27, 28,28, 30,31,32,18,19,19,20,20, 21,21,21,21,22,22,22])))

edades = np.concatenate((edades1,edades2))

print(len(edades), np.mean(edades), np.std(edades))