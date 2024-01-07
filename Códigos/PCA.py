# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:37:24 2023

@author: corir
"""

#%% librerias

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
# Clase para realizar componentes principales
from sklearn.decomposition import PCA
# Estandarizador (transforma las variables en z-scores)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler() # Creamos el estandarizador para usarlo posteriormente
import seaborn as sns


#%% el santo trial

entrevista = 'Primera'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

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

#%% data
path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv'

path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_conautopercepcion_todas)

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin([4, 1,3])]

df = df.drop(['Sujetos', 'Condición'], axis=1)

#df = df.drop(['tercera_persona_norm', 'num advs norm'], axis=1) #no pasan ANOVA con 7 PCA tenes el 70%
#Estas no tienen mas de dos grupos con diferencias significativas entre ellos (de 10 total)
#df = df.drop(['num noun norm', 'num propn norm', 'Intensidad pysent', 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3', 'diámetro'], axis=1)
#si tiras las dos lineas anteriores con 5 PCA tenes el 70%
#tiro por PCA
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm'], axis = 1)
#70% se alcanza con 7
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop'], axis = 1)
#el 70% se explica con 6
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2'], axis = 1)
#el 70% se explica con 5
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2', 'L3', 'k_mean', 'Comunidades_LSC', 'Detalles internos norm', 'Detalles externos norm'], axis = 1)
#el 70% se explica con 5
#df = df.drop(['average_CC', 'Intensidad_autop', 'Positivo pysent', 'selfloops', 'num_palabras_unicas_norm', 'ASP', 'num adj norm', 'num numeral norm', 'density', 'Recuerdo_autop', 'ValeInt_autop', 'Valencia_autop', 'transitivity', 'tercera_persona_norm', 'L2', 'L3', 'k_mean', 'Comunidades_LSC', 'Detalles internos norm', 'Detalles externos norm', 'num_nodes_LSC','primera_persona_norm','num advs norm','Negativo pysent','diámetro'], axis = 1)
#el 70% se explica con 3

#comparando pres y filler queda para tirar: num advs norm selfloops diámetro transitivity
#df = df.drop(['num advs norm', 'selfloops', 'diámetro', 'transitivity'], axis=1)
df = df.dropna()

#%% PCA

X = df.to_numpy()

print('Dimensiones de la matriz de features: {}'.format(X.shape))

# Ajustamos el estandarizador
std_scale.fit(X)

# Aplicamos el estandarizador y obtenemos la matriz de features escaleados
X_scaled = std_scale.transform(X)

# Creación del modelo. Si el número de componentes no se específica, 
# se obtienen tantas componentes principales como features en nuestro dataset.
pca = PCA(n_components=None)

# Ajustamos el modelo a los datos escaleados
pca.fit(X_scaled)

# Obtenemos la descripción de los datos en el espacio de componentes principales
X_pca = pca.transform(X_scaled)

print('Dimensiones de la matriz en componentes principales: {}'.format(X_pca.shape))
print(X_pca)

# con .explained_variance_ratio_ vemos la fracción de información que aporta cada componente
evr = pca.explained_variance_ratio_

# Graficamos la fracción de varianza que aporta cada componente
# y la información acumulada
fig, ax = plt.subplots(1, 2, figsize = (12, 4))

ax[0].plot(range(1, len(evr) + 1), evr, '.-', markersize = 20)
ax[0].set_ylabel('Fracción de varianza explicada')
ax[0].set_xlabel('Número de componente principal')

# Calculamos el acumulado con la función cumsum de numpy 
varianza_acumulada = np.cumsum(evr)

ax[1].plot(range(1, len(evr) + 1), varianza_acumulada, '.-', markersize = 20)
ax[1].set_ylabel('Fracción acumulada de varianza explicada')
ax[1].set_xlabel('Cantidad de componentes principales')

#%%

fig, ax = plt.subplots(1, 1, figsize = (18, 10))

# Calculamos el acumulado con la función cumsum de numpy 
varianza_acumulada = np.cumsum(evr)

ax.plot(range(1, len(evr) + 1), varianza_acumulada, '.-', markersize = 20, color = color_celeste, zorder = 5)
ax.set_ylabel('Fracción acumulada de varianza explicada')
ax.set_xlabel('Cantidad de componentes principales')
ax.axhline(y=0.7, color=color_gris, linestyle='--', linewidth = 4, label='70%')
ax.axvline(x = 8, color=color_gris, linestyle='--', linewidth = 4)
ax.grid(True)
plt.legend()
#plt.savefig(path_imagenes + "/PCAfrac_varianza_explicada.png", dpi=300, bbox_inches='tight', transparent=True)

#LAS PRIMERAS 6 COMPONENTES NOS TIENEN EL 70% DE LA VARIANZA EXPLICADA solo con 30
#Las 8 primeras con los 65

#%%
'''
Una pregunta interesante que está bueno indagar es qué significan cada una de las componentes. Esta información esta 
contenida en la lista *.pca.components_* de nuestro modelo ya ajustado.
Las primeras dos cubren el 40% de nuestros datos
'''
print('Features = {}'.format(df.columns))
print('PCA1 = {}'.format(pca.components_[0]))
print('PCA2 = {}'.format(pca.components_[1]))
print('PCA3 = {}'.format(pca.components_[2]))
print('PCA4 = {}'.format(pca.components_[3]))
print('PCA5 = {}'.format(pca.components_[4]))
print('PCA6 = {}'.format(pca.components_[5]))
#%% las meto en un df
# Define tus listas de variables y componentes principales
import matplotlib as npl

npl.rcParams["axes.labelsize"] = 20
npl.rcParams['xtick.labelsize'] = 20
npl.rcParams['ytick.labelsize'] = 20


variables = list(df.columns)
componentes_principales = [pca.components_[0], pca.components_[1], pca.components_[2], pca.components_[3], pca.components_[4], pca.components_[5],  pca.components_[6],  pca.components_[7]]

# Crea un diccionario con las componentes principales y las variables
data = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}

# Crea el DataFrame
df = pd.DataFrame(data, index=variables)

df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/primeras_6_componentes.csv')

center_value= 0
plt.figure(figsize = (30, len(variables)))
sns.heatmap(df, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, linecolor="black", center = center_value) #cbar_kws={"shrink": 0.75}) #"YlGnBu,  annot=True


plt.yticks(rotation=0) #plt.yticks(variables)
plt.xticks(rotation=0)
#plt.xlabel("Componentes Principales")


# Muestra el gráfico
plt.show()


#plt.savefig(path_imagenes + "/PCA_primeras6_componentes.png", dpi=300, bbox_inches='tight', transparent=True)


#%% Visualización de los datos en el espacio reducido

fig, ax = plt.subplots(figsize = (18, 10))

# Hacemos un scatter de los datos en las dos primeras componentes
ax.scatter(X_pca[:,0], X_pca[:,1], alpha = 0.65)

ax.set_xlabel('Primer componente principal')
ax.set_ylabel('Segunda componente principal')
plt.show()


#%%
fig, ax = plt.subplots(figsize=(10, 6))

# Supongamos que X_pca es una matriz 2D con 150 filas y 2 columnas
# Divide X_pca en cinco grupos de 30 puntos cada uno
group_size = 65
groups = [X_pca[0:65], X_pca[65:130], X_pca[130:195], X_pca[195:260-2], X_pca[260-2:]]
#groups = [X_pca[i:i + group_size] for i in range(0, len(X_pca), group_size)]

# Colores para cada grupo
colors = [color_campeones, "#A64253", "#F4DBD8", color_arabia, color_gris]

# Nombres para los grupos
#temas = ["Campeones", "Presencial", "CFK", "Arabia", "Filler"]

# Scatter plot para cada grupo con colores y nombres diferentes
for i, group in enumerate(groups):
    ax.scatter(group[:, 0], group[:, 1], s=150, label=temas[i], c=colors[i])

ax.set_xlabel('Primer componente principal')#, fontsize= 15)
ax.set_ylabel('Segunda componente principal')#, fontsize = 15)
ax.legend(fontsize = 17)  # Muestra la leyenda con etiquetas de nombres
plt.grid()
plt.show()


#plt.savefig(path_imagenes + "/PCA_pc1vspc2.png", dpi=300, bbox_inches='tight', transparent=True)

#%%

pca_no_importantes = []
pca_importantes = []
nro_pca_imp = 8
i = 0
pca_norm = pca.components_[i] / np.linalg.norm(pca.components_[i])
pca_no_importantes.append(np.where(abs(pca_norm) < 0.3)[0])
for i in range(1, nro_pca_imp):
    pca_norm = pca.components_[i] / np.linalg.norm(pca.components_[i])
    pca_no_importantes.append(np.where(abs(pca_norm) < 0.25)[0])
    #pca_importantes.append(np.where(abs(pca_norm) > 0.315)[0])
    
elementos_comunes = set(pca_no_importantes[0]).intersection(*pca_no_importantes[1:])

#print(elementos_comunes)
#print(len(elementos_comunes))

a = set(np.where(abs(pca.components_[8]) > 0.32)[0])#2
b = set(np.where(abs(pca.components_[7]) > 0.33)[0])#2
c = set(np.where(abs(pca.components_[6]) > 0.37)[0]) #2
d = set(np.where(abs(pca.components_[5]) > 0.4)[0]) #2
e = set(np.where(abs(pca.components_[4]) > 0.35)[0]) #2
f = set(np.where(abs(pca.components_[3]) > 0.3)[0]) #3 importantes
g = set(np.where(abs(pca.components_[2]) > 0.227)[0])# 4 importantes
h = set(np.where(abs(pca.components_[1]) > 0.28)[0]) #5 importantes
i = set(np.where(abs(pca.components_[0]) > 0.28)[0]) #6 importantes


elem_com =  list(a | b | c | d | e | f | g | h | i)

complemento_elem_com = set(range(0,32)) - set(elem_com)
print(len(complemento_elem_com))
vars_no_imp = [df.index[indice] for indice in complemento_elem_com]
for j, indice in enumerate(complemento_elem_com):
    print(df.index[indice])

