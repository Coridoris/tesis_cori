# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:14:48 2023

@author: corir
"""

#%%librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats

#%% funciones

def encontrar_intervalos(lista, valor):
    intervalos = []
    inicio = None

    for i, elemento in enumerate(lista):
        if elemento == valor:
            if inicio is None:
                inicio = lista.index[i]
        elif inicio is not None:
            intervalos.append((inicio, lista.index[i-1]))
            inicio = None

    # Manejar el caso cuando el último elemento es igual al valor
    if inicio is not None:
        intervalos.append((inicio, lista.index[-1]))

    return intervalos

def rgb_to_hex(rgb):
    # Asegurarse de que los valores estén en el rango [0, 1]
    rgb = tuple(max(0, min(1, x)) for x in rgb)

    # Convertir los valores RGB a enteros en el rango [0, 255]
    rgb_int = tuple(int(x * 255) for x in rgb)

    # Formatear el color en formato hexadecimal
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_int)

    return hex_color

def darken_color(color, factor=0.6):
    """
    Oscurece un color ajustando manualmente sus componentes RGB.

    Parameters:
    - color (str): Código hexadecimal del color.
    - factor (float): Factor de oscurecimiento (0 a 1).

    Returns:
    - str: Código hexadecimal del color oscurecido.
    """
    # Obtener componentes RGB
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # Aplicar el factor de oscurecimiento
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Asegurarse de que los valores estén en el rango correcto (0-255)
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))

    # Convertir los componentes oscurecidos de nuevo a código hexadecimal
    darkened_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

    return darkened_color


#%%santo trial

entrevista = 'Segunda' #'Primera'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3
}


#%%paleta de colores y path imagenes
color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

palette = sns.color_palette("PuBu", n_colors=5) #

#palette = sns.color_palette("autumn_r", n_colors=2)

# Asignar colores a las variables
color_1 = rgb_to_hex(palette[2])
color_2 = rgb_to_hex(palette[3])
color_3 = rgb_to_hex(palette[4])

color= [color_1, color_2, color_3]

rainbow_palette = sns.color_palette("rainbow", n_colors=7)

#rainbow_palette = sns.color_palette("autumn_r", n_colors=2)



# Asignar colores a las variables
color_1 = rgb_to_hex(rainbow_palette[0])
color_2 = rgb_to_hex(rainbow_palette[2])
color_3 = rgb_to_hex(rainbow_palette[4])
color_4 = rgb_to_hex(rainbow_palette[5])
color_5 = rgb_to_hex(rainbow_palette[6])

color_iv = [color_1, color_2, color_3, color_4, color_5]

color_iv_dark = [darken_color(color_1), darken_color(color_2), darken_color(color_3), darken_color(color_4)]

path_imagenes = "C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Figuras_finales/Encuestas"
#%%data edades y genero

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Encuestas/Preentrevista/Encuesta de Memoria Autobiográfica (Respuestas) 222 - Respuestas de formulario 1.csv'

df = pd.read_csv(path)

edades = df["Edad"]

genero = df["Género"]

genero[0] = "Masculino"
genero[64] = "Femenino"
genero[38] = "Femenino"



#%% Edades

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(edades, color = color_celeste, edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
ax1.set_xlabel('Edades', fontsize = 20)
ax1.set_ylabel('Cuentas', fontsize = 20)
ax1.tick_params(axis='both', labelsize=18)
ax1.axvline(x=np.mean(edades), ymin = 0, ymax= 0.947, color=darken_color(color[0]), linestyle='-', linewidth=5, label = "Media")
ax1.legend(fontsize = 20)

plt.tight_layout()
# Mostrar el histograma
plt.show()


plt.savefig(path_imagenes + '/edades_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/edades.png') 
plt.savefig(path_imagenes + '/edades.pdf') 

#%% encuesta SAM

path_SAM = "C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Encuesta de Memoria Autobiográfica (Respuestas) 222.csv"

df_SAM = pd.read_csv(path_SAM)

SAM_total = df_SAM["suma total"]

fig, ax2 = plt.subplots(figsize=(8, 8))
ax2.hist(SAM_total, color = color[0], edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
ax2.set_xlabel('Puntuación total SAM', fontsize = 20)
ax2.set_ylabel('Cuentas', fontsize = 20)
ax2.tick_params(axis='both', labelsize=18)
ax2.axvline(x=np.mean(SAM_total), ymin = 0, ymax= 0.876, color=darken_color(color[0]), linestyle='-', linewidth=5, label = "Media")
ax2.legend(fontsize = 20)

plt.tight_layout()
# Mostrar el histograma
plt.show()

plt.savefig(path_imagenes + '/SAMhist_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/SAMhist.png') 
plt.savefig(path_imagenes + '/SAMhist.pdf') 

#%% SAM y edad

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12, 5))
    
ax1.hist(edades, color = color[0], edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
ax1.set_xlabel('Edades', fontsize = 20)
ax1.set_ylabel('Cuentas', fontsize = 20)
ax1.tick_params(axis='both', labelsize=18)
ax1.axvline(x=np.mean(edades), ymin = 0, ymax= 0.947, color=darken_color(color[0]), linestyle='-', linewidth=5, label = "Media")
ax1.legend(fontsize = 20)

ax1.text(0.13, 0.97, '(a)', transform=ax1.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.text(0.13, 0.97, '(b)', transform=ax2.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.hist(SAM_total, color = color[0], edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
ax2.set_xlabel('Puntuación total SAM', fontsize = 20)
ax2.set_ylabel('Cuentas', fontsize = 20)
ax2.tick_params(axis='both', labelsize=18)
ax2.axvline(x=np.mean(SAM_total), ymin = 0, ymax= 0.874, color=darken_color(color[0]), linestyle='-', linewidth=5, label = "Media")
ax2.legend(fontsize = 20)

plt.tight_layout()
# Mostrar el histograma
plt.show()

plt.savefig(path_imagenes + '/SAMyedadhist_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/SAMyedadhist.png') 
plt.savefig(path_imagenes + '/SAMyedadhist.pdf') 
    

#%% genero q no va

# Calcular la cantidad de cada género
conteo_generos = genero.value_counts()

# Personaliza los colores para los géneros
colores = [color_celeste, color_celestito, color_palido, color_gris, 'orange']  # Agrega más colores si es necesario
colores = [color[0], color[2], color_palido]
# Crea el gráfico de torta
plt.figure(2), plt.clf()
plt.pie(conteo_generos, labels = conteo_generos.index, colors=colores, autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': 14})

#plt.legend(conteo_generos.index, title="Géneros", fontsize=12)

# Muestra el gráfico
plt.show()
#%% genero la q va
plt.figure(figsize=(8, 8))

# Configuración del tamaño de los textos y posición fuera del pie
textprops = {'fontsize': 20}
radius = 1.2

print(conteo_generos.index)

label = ["Masculino 54,5%", "Femenino 43,9%", "No binario 1,5%"]

# Crear el gráfico de pastel
plt.pie(conteo_generos, labels=label, colors=colores, startangle=140,
        pctdistance=0.85, textprops=textprops, radius=radius) #autopct='%1.1f%%'

# Ajustar la posición del texto
plt.gca().set_aspect('equal')
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + '/genero_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/genero.png') 
plt.savefig(path_imagenes + '/genero.pdf') 



#%% data post encuesta

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

df_postencuesta = pd.read_csv(path_conautopercepcion_todas)

vars_que_quiero = ['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop']

df_postencuesta = df_postencuesta[vars_que_quiero]

df_postencuesta['Condición'] = df_postencuesta['Condición'].map(mapping)
#elimino control
df_postencuesta = df_postencuesta[~df_postencuesta['Condición'].isin([5])]

df_postencuesta = df_postencuesta.dropna()

valencia = df_postencuesta['Valencia_autop']

intensidad = df_postencuesta['Intensidad_autop']

recuerdo = df_postencuesta['Recuerdo_autop']


ind_camp = encontrar_intervalos(df_postencuesta['Condición'], 1)
ind_pres = encontrar_intervalos(df_postencuesta['Condición'], 2)
ind_cfk = encontrar_intervalos(df_postencuesta['Condición'], 3)
ind_ar = encontrar_intervalos(df_postencuesta['Condición'], 4)
ind_control = encontrar_intervalos(df_postencuesta['Condición'], 5)
#%%figura post encuesta mean val e int
fig, ax = plt.subplots(figsize = (20, 7))

# # Hacemos un scatter plot de cada uno de los datos
# if ind_cfk != []:
#     ax.scatter(np.mean(intensidad[ind_cfk[0][0]:ind_cfk[0][1]]), np.mean(valencia[ind_cfk[0][0]:ind_cfk[0][1]]), marker = "o", s = 100, c=color_iv[0], label = "CFK")
# if ind_camp != []:
#     ax.scatter(np.mean(intensidad[ind_camp[0][0]:ind_camp[0][1]]), np.mean(valencia[ind_camp[0][0]:ind_camp[0][1]]), marker = "o", s = 100, c=color_iv[1], label = "Campeones")
# if ind_pres !=[]:
#     ax.scatter(np.mean(intensidad[ind_pres[0][0]:ind_pres[0][1]]), np.mean(valencia[ind_pres[0][0]:ind_pres[0][1]]),  marker = "o", s = 100, c=color_iv[2], label = "Presencial")
# if ind_ar != []:
#     ax.scatter(np.mean(intensidad[ind_ar[0][0]:ind_ar[0][1]]), np.mean(valencia[ind_ar[0][0]:ind_ar[0][1]]), marker = "o", s = 100,  c=color_iv[3], label = "Arabia")

cond = ["CFK", "Campeones", "Presencial", "Arabia"]
for i, ind in enumerate([ind_cfk, ind_camp, ind_pres, ind_ar]):
    if ind != []:
        mean_intensidad = np.mean(intensidad.loc[ind[0][0]:ind[0][1]])
        mean_valencia = np.mean(valencia.loc[ind[0][0]:ind[0][1]])

        std_intensidad = stats.sem(intensidad.loc[ind[0][0]:ind[0][1]])
        std_valencia = stats.sem(valencia.loc[ind[0][0]:ind[0][1]])

        ax.scatter(mean_intensidad, mean_valencia, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)
        ax.errorbar(mean_intensidad, mean_valencia, xerr=std_intensidad, yerr=std_valencia, fmt='none', ecolor=color_iv[i], elinewidth=2, capsize=5, alpha = 0.7)

ax.set_xlabel('Media intensidad', fontsize = 20)
ax.set_ylabel('Media valencia', fontsize = 20)
ax.legend(fontsize = 18)
ax.tick_params(axis='x', labelsize=18)  
ax.tick_params(axis='y', labelsize=18)

plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad_media_transparente.png', transparent = True) 
plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad_media.png') 
plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad_media.pdf') 
#%% post encuesta val e int

fig, ax = plt.subplots(figsize = (20, 7))
cond = ["CFK", "Arabia", "Campeones", "Presencial"]

for i, ind in enumerate([ind_cfk,  ind_ar, ind_camp, ind_pres]):
    if ind != []:
        tema_intensidad = intensidad.loc[ind[0][0]:ind[0][1]]
        tema_valencia = valencia.loc[ind[0][0]:ind[0][1]]

        dispersión_x = np.random.normal(loc=0, scale=0.1, size=len(tema_intensidad))
        dispersión_y = np.random.normal(loc=0, scale=0.1, size=len(tema_valencia))

        ax.scatter(tema_intensidad + dispersión_x, tema_valencia + dispersión_y, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)

ax.set_xlabel('Intensidad', fontsize = 20)
ax.set_ylabel('Valencia', fontsize = 20)
ax.legend(fontsize = 18)
ax.tick_params(axis='x', labelsize=18)  
ax.tick_params(axis='y', labelsize=18)

plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad_transparente.png', transparent = True) 
plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad.png') 
plt.savefig(path_imagenes + f'/{entrevista}_valenciavsintensidad.pdf') 
#%% recuerdo

cond = ["CFK", "Arabia", "Campeones", "Presencial"]
recuerdo_tema = []
mean_recuerdo_tema = []
sem_recuerdo_tema = []
for i, ind in enumerate([ind_cfk,  ind_ar, ind_camp, ind_pres]):
    recuerdo_tema.append(recuerdo.loc[ind[0][0]:ind[0][1]])
    mean_recuerdo_tema.append(np.mean(recuerdo.loc[ind[0][0]:ind[0][1]]))
    sem_recuerdo_tema.append(stats.sem(recuerdo.loc[ind[0][0]:ind[0][1]]))
    
fig, ax = plt.subplots(figsize=(10, 6))

# Configurar barras de error
error_bars_recuerdo = [sem * 1.96 for sem in sem_recuerdo_tema]  # Intervalo de confianza del 95%

# Crear el gráfico de barras con barras de error
bar_width = 0.35
index = np.arange(len(cond))
bars = ax.bar(index, mean_recuerdo_tema, bar_width, yerr=error_bars_recuerdo, capsize=5, color=color_iv, alpha = 0.7)

# Configurar etiquetas y leyenda
ax.set_xlabel('Condiciones', fontsize=20)
ax.set_ylabel('Media de Recuerdo', fontsize=20)
ax.set_xticks(index)
ax.set_xticklabels(cond)
ax.tick_params(axis='x', labelsize=18)  
ax.tick_params(axis='y', labelsize=18)
#ax.legend(fontsize = 18)

plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + f'/{entrevista}_recuerdo_transparente.png', transparent = True) 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdo.png') 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdo.pdf') 

#%% recuerdo e intensidad vs valencia juntas    
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

error_bars_recuerdo = [sem * 1.96 for sem in sem_recuerdo_tema]  # Intervalo de confianza del 95%

# Crear el gráfico de barras con barras de error
bar_width = 0.35
index = np.arange(len(cond))
bars = ax1.bar(index, mean_recuerdo_tema, bar_width, yerr=error_bars_recuerdo, capsize=5, color=color_iv, alpha = 0.7)

# Configurar etiquetas y leyenda
ax1.set_xlabel('Condiciones', fontsize=20)
ax1.set_ylabel('Media de Recuerdo', fontsize=20)
ax1.set_xticks(index)
ax1.set_xticklabels(cond)
ax1.tick_params(axis='x', labelsize=18)  
ax1.tick_params(axis='y', labelsize=18)


cond = ["CFK", "Campeones", "Presencial", "Arabia"]
for i, ind in enumerate([ind_cfk, ind_camp, ind_pres, ind_ar]):
    if ind != []:
        mean_intensidad = np.mean(intensidad.loc[ind[0][0]:ind[0][1]])
        mean_valencia = np.mean(valencia.loc[ind[0][0]:ind[0][1]])

        std_intensidad = stats.sem(intensidad.loc[ind[0][0]:ind[0][1]])
        std_valencia = stats.sem(valencia.loc[ind[0][0]:ind[0][1]])

        ax2.scatter(mean_intensidad, mean_valencia, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)
        ax2.errorbar(mean_intensidad, mean_valencia, xerr=std_intensidad, yerr=std_valencia, fmt='none', ecolor=color_iv[i], elinewidth=2, capsize=5, alpha = 0.7)

ax2.set_xlabel('Media intensidad', fontsize = 20)
ax2.set_ylabel('Media valencia', fontsize = 20)
ax2.legend(fontsize = 18, loc = "lower right")
ax2.tick_params(axis='x', labelsize=18)  
ax2.tick_params(axis='y', labelsize=18)

ax1.text(0.13, 0.97, '(a)', transform=ax1.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.text(0.13, 0.97, '(b)', transform=ax2.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + f'/{entrevista}_recuerdoymediaIntvsVal_transparente.png', transparent = True) 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdoymediaIntvsVal.png') 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdoymediaIntvsVal.pdf') 


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

error_bars_recuerdo = [sem * 1.96 for sem in sem_recuerdo_tema]  # Intervalo de confianza del 95%

# Crear el gráfico de barras con barras de error
bar_width = 0.35
index = np.arange(len(cond))
bars = ax1.bar(index, mean_recuerdo_tema, bar_width, yerr=error_bars_recuerdo, capsize=5, color=color_iv, alpha = 0.7)

# Configurar etiquetas y leyenda
ax1.set_xlabel('Condiciones', fontsize=20)
ax1.set_ylabel('Media de Recuerdo', fontsize=20)
ax1.set_xticks(index)
ax1.set_xticklabels(cond)
ax1.tick_params(axis='x', labelsize=18)  
ax1.tick_params(axis='y', labelsize=18)


cond = ["CFK", "Arabia", "Campeones", "Presencial"]

for i, ind in enumerate([ind_cfk,  ind_ar, ind_camp, ind_pres]):
    if ind != []:
        tema_intensidad = intensidad.loc[ind[0][0]:ind[0][1]]
        tema_valencia = valencia.loc[ind[0][0]:ind[0][1]]

        dispersión_x = np.random.normal(loc=0, scale=0.1, size=len(tema_intensidad))
        dispersión_y = np.random.normal(loc=0, scale=0.1, size=len(tema_valencia))

        ax2.scatter(tema_intensidad + dispersión_x, tema_valencia + dispersión_y, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)

ax2.set_xlabel('Intensidad', fontsize = 20)
ax2.set_ylabel('Valencia', fontsize = 20)
#ax2.legend(fontsize = 18)
ax2.tick_params(axis='x', labelsize=18)  
ax2.tick_params(axis='y', labelsize=18)

ax1.text(0.13, 0.97, '(a)', transform=ax1.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.text(0.13, 0.97, '(b)', transform=ax2.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + f'/{entrevista}_recuerdoyIntvsVal_transparente.png', transparent = True) 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdoyIntvsVal.png') 
plt.savefig(path_imagenes + f'/{entrevista}_recuerdoyIntvsVal.pdf') 
#%% recuerdo e intensidad vs valencia media con 1t y 2t juntas    

cond = ["CFK", "Arabia", "Campeones", "Presencial"]
path_conautopercepcion_todas1 = 'C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Primera_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'
path_conautopercepcion_todas2 = 'C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Segunda_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'


df_postencuesta1 = pd.read_csv(path_conautopercepcion_todas1)
df_postencuesta2 = pd.read_csv(path_conautopercepcion_todas2)

vars_que_quiero = ['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop']

df_postencuesta1 = df_postencuesta1[vars_que_quiero]
df_postencuesta2 = df_postencuesta2[vars_que_quiero]

df_postencuesta1['Condición'] = df_postencuesta1['Condición'].map(mapping)
df_postencuesta1 = df_postencuesta1[~df_postencuesta1['Condición'].isin([5])]
df_postencuesta1 = df_postencuesta1.dropna()
valencia1 = df_postencuesta1['Valencia_autop']
intensidad1 = df_postencuesta1['Intensidad_autop']
recuerdo1 = df_postencuesta1['Recuerdo_autop']

df_postencuesta2['Condición'] = df_postencuesta2['Condición'].map(mapping)
df_postencuesta2 = df_postencuesta2[~df_postencuesta2['Condición'].isin([5])]
df_postencuesta2 = df_postencuesta2.dropna()
valencia2 = df_postencuesta2['Valencia_autop']
intensidad2 = df_postencuesta2['Intensidad_autop']
recuerdo2 = df_postencuesta2['Recuerdo_autop']

ind_camp1 = encontrar_intervalos(df_postencuesta1['Condición'], 1)
ind_pres1 = encontrar_intervalos(df_postencuesta1['Condición'], 2)
ind_cfk1 = encontrar_intervalos(df_postencuesta1['Condición'], 3)
ind_ar1 = encontrar_intervalos(df_postencuesta1['Condición'], 4)
ind_control1 = encontrar_intervalos(df_postencuesta1['Condición'], 5)

ind_camp2 = encontrar_intervalos(df_postencuesta2['Condición'], 1)
ind_pres2 = encontrar_intervalos(df_postencuesta2['Condición'], 2)
ind_cfk2 = encontrar_intervalos(df_postencuesta2['Condición'], 3)
ind_ar2 = encontrar_intervalos(df_postencuesta2['Condición'], 4)
ind_control2 = encontrar_intervalos(df_postencuesta2['Condición'], 5)

recuerdo_tema1 = []
mean_recuerdo_tema1 = []
sem_recuerdo_tema1 = []
for i, ind in enumerate([ind_cfk1,  ind_ar1, ind_camp1, ind_pres1]):
    recuerdo_tema1.append(recuerdo1.loc[ind[0][0]:ind[0][1]])
    mean_recuerdo_tema1.append(np.mean(recuerdo1.loc[ind[0][0]:ind[0][1]]))
    sem_recuerdo_tema1.append(stats.sem(recuerdo1.loc[ind[0][0]:ind[0][1]]))
    
recuerdo_tema2 = []
mean_recuerdo_tema2 = []
sem_recuerdo_tema2 = []
for i, ind in enumerate([ind_cfk2,  ind_ar2, ind_camp2, ind_pres2]):
    recuerdo_tema2.append(recuerdo2.loc[ind[0][0]:ind[0][1]])
    mean_recuerdo_tema2.append(np.mean(recuerdo2.loc[ind[0][0]:ind[0][1]]))
    sem_recuerdo_tema2.append(stats.sem(recuerdo2.loc[ind[0][0]:ind[0][1]]))


    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

error_bars_recuerdo1 = [sem for sem in sem_recuerdo_tema1] 

error_bars_recuerdo2 = [sem for sem in sem_recuerdo_tema2]  

# # Crear el gráfico de barras con barras de error
# bar_width = 0.35
# index = np.arange(len(cond))
# bars1 = ax1.bar(index, mean_recuerdo_tema1, bar_width, yerr=error_bars_recuerdo1, capsize=5, color=color_iv, alpha = 0.7)

index = np.arange(len(cond))

# Crear la primera barra con barras de error
bars1 = ax1.bar(index- bar_width/2, mean_recuerdo_tema1, bar_width, yerr=error_bars_recuerdo1, capsize=5, color=color_iv, alpha=0.7)

# Crear la segunda barra pegada a la primera
bars2 = ax1.bar(index + bar_width/2, mean_recuerdo_tema2, bar_width, yerr=error_bars_recuerdo2, capsize=5, color=color_iv_dark, alpha=0.7)


# Configurar etiquetas y leyenda
ax1.set_xlabel('Condiciones', fontsize=20)
ax1.set_ylabel('Media de recuerdo', fontsize=20)
ax1.set_xticks(index)
ax1.set_xticklabels(cond)
ax1.tick_params(axis='x', labelsize=18)  
ax1.tick_params(axis='y', labelsize=18)



for i, ind in enumerate([ind_cfk1, ind_camp1, ind_pres1, ind_ar1]):
    if ind != []:
        mean_intensidad1 = np.mean(intensidad1.loc[ind[0][0]:ind[0][1]])
        mean_valencia1 = np.mean(valencia1.loc[ind[0][0]:ind[0][1]])
        
        std_intensidad1 = stats.sem(intensidad1.loc[ind[0][0]:ind[0][1]])
        std_valencia1 = stats.sem(valencia1.loc[ind[0][0]:ind[0][1]])

        ax2.scatter(mean_intensidad1, mean_valencia1, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)
        ax2.errorbar(mean_intensidad1, mean_valencia1, xerr=std_intensidad1, yerr=std_valencia1, fmt='none', ecolor=color_iv[i], elinewidth=2, capsize=5, alpha = 0.7)

for i, ind in enumerate([ind_cfk2, ind_camp2, ind_pres2, ind_ar2]):
    if ind != []:
        mean_intensidad2 = np.mean(intensidad2.loc[ind[0][0]:ind[0][1]])
        mean_valencia2 = np.mean(valencia2.loc[ind[0][0]:ind[0][1]])

        std_intensidad2 = stats.sem(intensidad2.loc[ind[0][0]:ind[0][1]])
        std_valencia2 = stats.sem(valencia2.loc[ind[0][0]:ind[0][1]])

        ax2.scatter(mean_intensidad2, mean_valencia2, marker="o", s=100, c=darken_color(color_iv[i]), alpha = 0.7)
        ax2.errorbar(mean_intensidad2, mean_valencia2, xerr=std_intensidad2, yerr=std_valencia2, fmt='none', ecolor=darken_color(color_iv[i]), elinewidth=2, capsize=5, alpha = 0.7)


ax2.set_xlabel('Media intensidad', fontsize = 20)
ax2.set_ylabel('Media valencia', fontsize = 20)
ax2.legend(fontsize = 18, loc = "lower right")
ax2.tick_params(axis='x', labelsize=18)  
ax2.tick_params(axis='y', labelsize=18)

ax1.text(0.13, 0.97, '(a)', transform=ax1.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.text(0.13, 0.97, '(b)', transform=ax2.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + '/recuerdoymediaIntvsVal_dost_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/recuerdoymediaIntvsVal_dost.png') 
plt.savefig(path_imagenes + '/recuerdoymediaIntvsVal_dost.pdf') 
