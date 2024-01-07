# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:31:48 2023

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
#para ANOVA
import pingouin as pg
from scipy.stats import f
#%%
path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/11-7'

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

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

condicion = temas[0]

path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv'

path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_sinautopercepcion_todas)

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}



#%% ANOVA


df = pd.read_csv(path_sinautopercepcion_todas)

condiciones_a_eliminar = ['campeones_del_mundo', 'antesdevenir']

# Filtramos las condiciones que creo que dan las diferencias de las medias
df_sin_camp_ni_filler = df[~df['Condición'].isin(condiciones_a_eliminar)]

variables_dependientes = ['Intensidad pysent', 'Positivo pysennt', 'Negativo pysent', 'Nro palabras únicas', 
                          'num noun norm', 'num verb norm', 'num adj norm', 'num advs norm', 'num numeral norm',
                          'num propn norm', 'primera_persona_norm', 'tercera_persona_norm', 'cohe_norm_d=1', 
                          'cohe_norm_d=2', 'cohe_norm_d=3', 'num_nodes_norm', 'Comunidades_LSC', 'diámetro', 'k_mean', 
                          'transitivity', 'ASP', 'average_CC', 'selfloops', 'L2', 'L3', 'density',
                          'Detalles internos norm', 'Detalles externos norm'] # = np.array(df.columns[2:])

#aov = pg.rm_anova(dv = 'Nro palabras únicas', within = 'Condición', subject='Sujetos', data=df, detailed=True, effsize="np2")

aov = df.rm_anova(dv = 'Nro palabras únicas', within='Condición', subject='Sujetos',  detailed=False)

# Definir los grados de libertad del numerador y del denominador
df_between = aov['ddof1'][0]  # Grados de libertad del numerador
df_within = aov['ddof2'][0]   # Grados de libertad del denominador

# Definir el nivel de significancia (alfa)
alfa = 0.05

# Calcular el valor crítico de F
f_critical = f.ppf(1 - alfa, df_between, df_within)

print(f"Valor crítico de F: {f_critical}")

print(f"Valor de F: {aov['F'][0]}")

print(f"Valor de p: {aov['p-unc'][0]}")

print(f"Valor de epsilon: {aov['eps'][0]}")

print(f"Valor de p corregido: {aov['p-GG-corr'][0]}")

#%% veo si alguna variable da que no hay diferencias

# Definir el nivel de significancia (alfa)
alfa = 0.05

Fsignificativas = []
F = []
F_critico = []
psignificativas = []
p = []
epsilon = []
p_corr = []

for i, variable in enumerate(variables_dependientes):
    '''
    si quiero sin campeones ni filler tengo que poner aca df_sin_camp_ni_filler en vez de df
    '''
    aov = df.rm_anova(dv = variable, within='Condición', subject='Sujetos',  detailed=False)

    print(variable)
    
    # Definir los grados de libertad del numerador y del denominador
    df_between = aov['ddof1'][0]  # Grados de libertad del numerador
    df_within = aov['ddof2'][0]   # Grados de libertad del denominador

    # Calcular el valor crítico de F
    f_critical = f.ppf(1 - alfa, df_between, df_within)
    
    F.append(aov['F'][0])
    F_critico.append(f_critical)
    epsilon.append(aov['eps'][0])
    p.append(aov['p-unc'][0])
    
    if f_critical > aov['F'][0]:
        print('La variable ' + variable + f" tiene un F ({aov['F'][0]}) que no supera el crítico ({f_critical}).")
        Fsignificativas.append(False)
    else:
        Fsignificativas.append(True)
    if 'p-GG-corr' not in np.array(aov.columns):
        pval = aov['p-unc'][0]
        p_corr.append(False)
    else:
        pval = aov['p-GG-corr'][0]
        p_corr.append(aov['p-GG-corr'][0])
    if pval > alfa:
        print("La variable " + variable + f" tiene un pval corregido ({pval}) que no supera el nivel de significancia ({alfa}).")
        psignificativas.append(False)
    else:
        psignificativas.append(True)
        
#Acomodo todo en un csv

df_resultados = pd.DataFrame({
    'Variable': variables_dependientes,
    'F': F,
    'F_critico': F_critico,
    'F es significativo?': Fsignificativas,
    'P': p,
    'Epsilon': epsilon,
    'P_corr': p_corr,
    'P significativo?': psignificativas,
})

# Guardar el DataFrame en un archivo CSV
df_resultados.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/ANOVAS.csv', index=False)
# para guardar sin camp ni filler busca si quiero sin campeones ni filler tengo que poner aca df_sin_camp_ni_filler en vez de df
#%% para lo que sigue no queremos estas columnas

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df.drop(['Sujetos', 'Condición'], axis=1)

#%% correlacion

variables = ["Recuerdo", "Valencia", "Intensidad+valencia", "Intensidad", "Intensidad", "Positivo", "Negativo", "Nro. palabras", "Nro. sust", "Nro. verb", "Nro. adj", "Nro. advs", "Nro. num", "Nro. prop", "Coherencia 1", "Coherencia 2", "Coherencia 3", "Nodos", "Enlaces", "Comunidades", "Diámetro", "Grado medio", "Transitividad", "ASP", "CC", "L1", "L2", "L3", "Densidad", "Internos", "Externos"]
variables_x = [var[:6] for var in variables]

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear un mapa de calor (heatmap) de la matriz de correlación
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5, xticklabels = variables_x, yticklabels = variables)
plt.rc('font', size=25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
# Mostrar el mapa de calor
plt.show()

#plt.savefig(path_imagenes + "/matriz_corr.png", bbox_inches='tight', pad_inches=0, transparent=True)

# Crear listas para almacenar las variables y sus p-values asociados
variable_pairs = []
p_values = []

from scipy.stats import pearsonr

# Iterar a través de las filas y columnas de la matriz de correlación
for row in correlation_matrix.index:
    for col in correlation_matrix.columns:
        # Evitar duplicados y la diagonal principal
        if row != col and (col, row) not in variable_pairs:
            # Calcular la correlación de Pearson y el p-value
            corr, p_value = pearsonr(df[row], df[col])
            # Almacenar el par de variables y su p-value
            variable_pairs.append((row, col))
            p_values.append(p_value)

# Darle forma a las listas para que coincidan con la longitud de la matriz de correlación
variable_pairs = np.array(variable_pairs)
p_values = np.array(p_values)


#defino la significancia como 0.05 / numero de comparaciones = len(nro_variables**2 /2)
#(len(df.columns)*(len(df.columns)-1)) = len(p_values)
significancia = 0.05/len(p_values)

significativo = np.where(p_values < significancia, True, False)

# Crear un DataFrame para mostrar las variables y sus p-values
result_df = pd.DataFrame({"Variable 1": variable_pairs[:, 0], "Variable 2": variable_pairs[:, 1], "P-Value": p_values, "Dio significativo corrigiendo p?": significativo})

# Mostrar el DataFrame
print(result_df)

# Guardar el DataFrame en un archivo CSV
result_df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/p_val_matriz_corr.csv', index=False)


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
ax.axvline(x = 6, color=color_gris, linestyle='--', linewidth = 4)
ax.grid(True)
plt.legend()
plt.savefig(path_imagenes + "/PCAfrac_varianza_explicada.png", dpi=300, bbox_inches='tight', transparent=True)

#LAS PRIMERAS 6 COMPONENTES NOS TIENEN EL 70% DE LA VARIANZA EXPLICADA

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

npl.rcParams["axes.labelsize"] = 26
npl.rcParams['xtick.labelsize'] = 50
npl.rcParams['ytick.labelsize'] = 46


variables = ["Intensidad", "Positivo", "Negativo", "Nro. palabras", "Nro. sust", "Nro. verb", "Nro. adj", "Nro. advs", "Nro. num", "Nro. prop", "Coherencia 1", "Coherencia 2", "Coherencia 3", "Nodos", "Enlaces", "Comunidades", "Diámetro", "Grado medio", "Transitividad", "ASP", "CC", "L1", "L2", "L3", "Densidad", "Internos", "Externos"]

componentes_principales = [pca.components_[0], pca.components_[1], pca.components_[2], pca.components_[3], pca.components_[4], pca.components_[5]]

# Crea un diccionario con las componentes principales y las variables
data = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}

# Crea el DataFrame
df = pd.DataFrame(data, index=variables)

df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/primeras_6_componentes.csv')


plt.figure(figsize = (30, len(variables)))
sns.heatmap(df, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, linecolor="black"), #cbar_kws={"shrink": 0.75}) #"YlGnBu,  annot=True


plt.yticks(rotation=0) #plt.yticks(variables)
plt.xticks(rotation=0)
#plt.xlabel("Componentes Principales")


# Muestra el gráfico
plt.show()


plt.savefig(path_imagenes + "/PCA_primeras6_componentes.png", dpi=300, bbox_inches='tight', transparent=True)


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
group_size = 30
groups = [X_pca[i:i + group_size] for i in range(0, len(X_pca), group_size)]

# Colores para cada grupo
colors = [color_campeones, "#A64253", "#F4DBD8", color_arabia, color_gris]

# Nombres para los grupos
temas = ["Campeones", "Presencial", "CFK", "Arabia", "Filler"]

# Scatter plot para cada grupo con colores y nombres diferentes
for i, group in enumerate(groups):
    ax.scatter(group[:, 0], group[:, 1], s=150, label=temas[i], c=colors[i])

ax.set_xlabel('Primer componente principal')#, fontsize= 15)
ax.set_ylabel('Segunda componente principal')#, fontsize = 15)
ax.legend(fontsize = 17)  # Muestra la leyenda con etiquetas de nombres
plt.grid()
plt.show()


plt.savefig(path_imagenes + "/PCA_pc1vspc2.png", dpi=300, bbox_inches='tight', transparent=True)


