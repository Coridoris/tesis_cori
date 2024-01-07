# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:09:15 2023

@author: corir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

#%% grafico por tema

Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)
corrimiento = np.ones(30)*0.4

palabras = ['totales', 'unicas']

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    palabras = ['totales', 'unicas']
    
    df_cuentopalabras = pd.read_csv(path)
    
    percepcion = df_cuentopalabras["cuanto_recordaste"]
    
    nro_palabras = df_cuentopalabras["num_tot_palabras"]
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    nro = [nro_palabras, nro_palabras_unicas]

    for indice, palabra in enumerate(palabras):
        fig, ax1 = plt.subplots()
        
        # Configurar la primera serie (eje y izquierdo)
        color = 'tab:red'
        ax1.set_xlabel('Nro. sujeto')
        ax1.set_ylabel('Número de palabras', color=color)
        ax1.bar(nro_sujeto, nro[indice], color=color, width=0.4, label = palabra)
        #ax1.bar(nro_sujeto, nro_palabras_unicas, '.', color='lightcoral', label = 'unicas')
        ax1.legend()
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Configurar la segunda serie (eje y derecho)
        ax2 = ax1.twinx()  # Crea un segundo eje y
        color = 'tab:blue'
        ax2.set_ylabel('Autopercepción de cuánto recordaron', color=color)
        ax2.bar(nro_sujeto + corrimiento, percepcion, color=color, width=0.4)
        ax2.tick_params(axis='y', labelcolor=color)
        
        #titulo
        plt.title(tema)
        
        # Ajustar el tamaño del gráfico
        fig.tight_layout()
        
    
        # Mostrar el gráfico
        plt.show()
        
        
        #plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_tema/{tema}_{palabra}.png')

#%% grafico por sujeto

#nro_sujeto = 6
sujetos = range(1,14)

tema = ["campeones_del_mundo", "arabia", "presencial", "cfk", "antesdevenir"]

temas = ["a", "campeones", "arabia", "presencial", "cfk", "antesdevenir"]

palabras = ['totales', 'unicas']


for nro_sujeto in sujetos:

    percepcion_sujeto = []
    nro_palabras_sujeto = []
    nro_palabras_unicas_sujeto = []
    
    for i, c in enumerate(tema):
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{c}_cuentocadapalabra.csv'
        
        df_cuentopalabras_sujeto = pd.read_csv(path)
    
        percepcion_tema = df_cuentopalabras_sujeto["cuanto_recordaste"][nro_sujeto]
        percepcion_sujeto.append(percepcion_tema)
    
        nro_palabras_tema = df_cuentopalabras_sujeto["num_tot_palabras"][nro_sujeto]
        nro_palabras_sujeto.append(nro_palabras_tema)
    
        nro_palabras_unicas_tema = df_cuentopalabras_sujeto["num_tot_palabras_unicas"][nro_sujeto]
        nro_palabras_unicas_sujeto.append(nro_palabras_unicas_tema)
    
    nro = [nro_palabras_sujeto, nro_palabras_unicas_sujeto]
    
    for k, palabra in enumerate(palabras):
    
        nro_tema = np.linspace(1, 5, 5)
        corrimiento = np.ones(5)*0.4
        
        #plt.figure(1), plt.clf()
        #plt.plot(nro_sujeto, percepcion, 'o')
        #plt.plot(nro_sujeto, nro_palabras, 'o')
        
        
        fig, ax1 = plt.subplots()
        
        # Configurar la primera serie (eje y izquierdo)
        color = 'tab:red'
        ax1.set_xlabel('Nro. tema')
        ax1.set_xticklabels(temas)
        ax1.set_ylabel('Número de palabras', color=color)
        ax1.bar(nro_tema, nro[k], color=color, width=0.4, label = palabra)
        #ax1.bar(nro_tema, nro_palabras_unicas_sujeto, '.', color='lightcoral', label = 'unicas')
        ax1.legend()
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Configurar la segunda serie (eje y derecho)
        ax2 = ax1.twinx()  # Crea un segundo eje y
        color = 'tab:blue'
        ax2.set_ylabel('Autopercepción de cuánto recordaron', color=color)
        ax2.bar(nro_tema + corrimiento, percepcion_sujeto, color=color, width=0.4)
        ax2.tick_params(axis='y', labelcolor=color)
        
        #titulo
        plt.title(f"Sujeto {nro_sujeto}")
        
        # Ajustar el tamaño del gráfico
        fig.tight_layout()
        
        
        # Mostrar el gráfico
        plt.show()
        
        #plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_sujeto/sujeto_{nro_sujeto}_{palabra}.png')

#%% correlacion nro palabras tot y unicas

Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)
corrimiento = np.ones(30)*0.4

palabras = ['totales', 'unicas']

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]


nro_palabras_por_tema = []

nro_palabras_unicas_por_tema = []

correlation_por_tema = []


for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path)
    
    percepcion = df_cuentopalabras["cuanto_recordaste"]
    
    nro_palabras = df_cuentopalabras["num_tot_palabras"]
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    nro_palabras_por_tema.append(nro_palabras)
    
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)

plt.figure(1), plt.clf()
for i in range(len(temas)):
    plt.plot(nro_palabras_por_tema[i], nro_palabras_unicas_por_tema[i], 'o', label = temas[i])
    
    # Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(nro_palabras_por_tema[i], nro_palabras_unicas_por_tema[i])
    correlation_por_tema.append(correlation_matrix[0, 1])
    
plt.legend(fontsize = 12)
plt.grid(True)
plt.xlabel('Palabras totales', fontsize = 15)
plt.ylabel('Palabras únicas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
        


columnas = temas
valores = correlation_por_tema

data = [valores]
df_tema = pd.DataFrame(data, columns=columnas)

print(df_tema)

#plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_tema/{tema}_{palabra}.png')
#%% total no x tema

nro_palabras = []

nro_palabras_unicas = []

for lista in nro_palabras_por_tema:
    nro_palabras.extend(lista)
    
for lista in nro_palabras_unicas_por_tema:
    nro_palabras_unicas.extend(lista)


# Calcular el coeficiente de correlación
correlation_matrix = np.corrcoef(nro_palabras, nro_palabras_unicas)
correlation = correlation_matrix[0, 1]

# Imprimir el resultado
print("Coeficiente de correlación (R):", correlation)

#%% correlacion nro palabras tot y unicas x sujeto

sujetos = range(1,14)

tema = ["campeones_del_mundo", "arabia", "presencial", "cfk", "antesdevenir"]

palabras = ['totales', 'unicas']

nro_palabras = []

nro_palabras_unicas = []

correlation_por_sujeto = []

for nro_sujeto in sujetos:

    percepcion_sujeto = []
    nro_palabras_sujeto = []
    nro_palabras_unicas_sujeto = []
    
    for i, c in enumerate(tema):
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{c}_cuentocadapalabra.csv'
        
        df_cuentopalabras_sujeto = pd.read_csv(path)
    
        percepcion_tema = df_cuentopalabras_sujeto["cuanto_recordaste"][nro_sujeto]
        percepcion_sujeto.append(percepcion_tema)
    
        nro_palabras_tema = df_cuentopalabras_sujeto["num_tot_palabras"][nro_sujeto]
        nro_palabras_sujeto.append(nro_palabras_tema)
    
        nro_palabras_unicas_tema = df_cuentopalabras_sujeto["num_tot_palabras_unicas"][nro_sujeto]
        nro_palabras_unicas_sujeto.append(nro_palabras_unicas_tema)
    
    nro_palabras.append(nro_palabras_sujeto)
    nro_palabras_unicas.append(nro_palabras_unicas_sujeto)


markers = ['o', 's', 'x', 'v', '+']#misma cantidad que en nro_palabras[0]
plt.figure(2), plt.clf()
for j in range(len(nro_palabras[0])):
    for i in range(len(nro_palabras)):
        color = cm.tab20(i)
        plt.plot(nro_palabras[i][j], nro_palabras_unicas[i][j],  marker = markers[j], color =  color)
    
    # Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(nro_palabras[i], nro_palabras_unicas[i])
    correlation_por_sujeto.append(correlation_matrix[0, 1])
    
# Crear líneas ficticias para la leyenda
lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]

# Agregar las líneas ficticias y las etiquetas a la leyenda
plt.legend(lines, tema)


plt.grid(True)
plt.xlabel('Palabras totales', fontsize = 15)
plt.ylabel('Palabras únicas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#%%
columnas = sujetos
valores = correlation_por_sujeto

data = [valores]
df_sujetos = pd.DataFrame(data, columns=columnas)

print(df_sujetos)

        
#plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_sujeto/sujeto_{nro_sujeto}_{palabra}.png')



#%%

from scipy.stats import spearmanr

Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo"]#, "antesdevenir", "presencial", "cfk", "arabia"]

percepcion_por_tema = []
nro_palabras_unicas_por_tema = []
correlation_por_tema = []
pvals_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path).dropna()
    
    percepcion = df_cuentopalabras["cuanto_recordaste"]
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    percepcion_por_tema.append(percepcion)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que len(percepcion_por_tema)
plt.figure(3), plt.clf()
for i in range(len(temas)):
    for j in range(len(percepcion_por_tema[0])):
        if j in percepcion_por_tema[i].index:
            color = cm.tab20(j)
            plt.plot(percepcion_por_tema[i][j], nro_palabras_unicas_por_tema[i][j], marker = markers[i], color = color)
    
    # Calcular el coeficiente de correlación de Spearman
    correlation, p_value = spearmanr(percepcion_por_tema[i], nro_palabras_unicas_por_tema[i])
    correlation_por_tema.append(correlation)
    pvals_por_tema.append(p_value)
    
# Crear líneas ficticias para la leyenda
lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]

# Agregar las líneas ficticias y las etiquetas a la leyenda
plt.legend(lines, temas)

plt.grid(True)
plt.xlabel('Percepción', fontsize = 15)
plt.ylabel('Palabras únicas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
        


columnas = temas
valores = correlation_por_tema

data = [valores, pvals_por_tema]
df_sperman_tema = pd.DataFrame(data, columns=columnas)

print(df_sperman_tema)

   
#plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_tema/{tema}_{palabra}.png')

#%% total no x tema

percepcion = []

nro_palabras_unicas = []

for lista in percepcion_por_tema:
    percepcion.extend(lista)
    
for lista in nro_palabras_unicas_por_tema:
    nro_palabras_unicas.extend(lista)


# Calcular el coeficiente de correlación de Spearman
correlation, p_value = spearmanr(percepcion, nro_palabras_unicas)

# Imprimir el resultado
print("Coeficiente de correlación de Spearman:", correlation)
print("Valor p:", p_value)
