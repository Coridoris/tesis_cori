# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:09:15 2023

@author: corir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from scipy.stats import spearmanr

#%% correlacion nro palabras tot y unicas x sujeto y tema

sujetos = range(0,30)

tema = ["campeones_del_mundo", "arabia", "presencial", "cfk", "antesdevenir"]

palabras = ['totales', 'unicas']

nro_palabras_x_sujeto = []

nro_palabras_unicas_x_sujeto = []

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
    
    nro_palabras_x_sujeto.append(nro_palabras_sujeto)
    nro_palabras_unicas_x_sujeto.append(nro_palabras_unicas_sujeto)



mean_palabras_unicas_x_sujeto = np.mean(nro_palabras_unicas_x_sujeto, axis = 1)
mean_palabras_x_sujeto = np.mean(nro_palabras_x_sujeto, axis = 1)

nro_palabras_unicas_norm = []

for sublista, valor in zip(nro_palabras_unicas_x_sujeto, mean_palabras_unicas_x_sujeto):
    nro_palabras_unicas_norm.append([x / valor for x in sublista])
    
nro_palabras_norm = []

for sublista, valor in zip(nro_palabras_x_sujeto, mean_palabras_x_sujeto):
    nro_palabras_norm.append([x / valor for x in sublista])


#si quiero normalizar por el promedio de palabras x sujeto agrego _norm

palabras = nro_palabras_norm

palabras_unicas = nro_palabras_unicas_norm

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que en nro_palabras[0]
plt.figure(1), plt.clf()
for i in range(len(nro_palabras_x_sujeto)):
    for j in range(len(nro_palabras_x_sujeto[0])):
        color = cm.tab20(i)
        plt.plot(palabras[i][j], palabras_unicas[i][j],  marker = markers[j], color =  color)
    
    #Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(palabras[i], palabras_unicas[i])
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


columnas = sujetos
valores = correlation_por_sujeto

data = [valores]
df_sujetos = pd.DataFrame(data, columns=columnas)

print(df_sujetos)

        
#plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_sujeto/sujeto_{nro_sujeto}_{palabra}.png')



#%% correlacion nro palabras unicas y percepcion


Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]

percepcion_por_tema = []
nro_palabras_unicas_por_tema = []
correlation_por_tema = []
pvals_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path)
    
    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df_cuentopalabras["num_tot_palabras_unicas"] = df_cuentopalabras["num_tot_palabras_unicas"]/mean_palabras_unicas_x_sujeto
    
    df_cuentopalabras = df_cuentopalabras.dropna()
    
    percepcion = df_cuentopalabras["cuanto_recordaste"]
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    percepcion_por_tema.append(percepcion)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que len(percepcion_por_tema)
plt.figure(2), plt.clf()
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


#%% Histograma de número de palabras por tema (de todos los sujetos)

Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]


nro_palabras_unicas_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path)
    
    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df_cuentopalabras["num_tot_palabras_unicas"] = df_cuentopalabras["num_tot_palabras_unicas"]/mean_palabras_unicas_x_sujeto
    
    df_cuentopalabras = df_cuentopalabras.dropna()
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    
plt.figure(3), plt.clf()
nro_tema = 3
plt.hist(nro_palabras_unicas_por_tema[nro_tema].values, label = temas[nro_tema])
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Cuentas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#%%
'''
vamos a hacer lo mismo pero normalizando con el zscore = (datos - su media) / su desviación estandar
'''


#%% correlacion nro palabras tot y unicas x sujeto y tema

sujetos = range(0,30)

tema = ["campeones_del_mundo", "arabia", "presencial", "cfk", "antesdevenir"]

palabras = ['totales', 'unicas']

nro_palabras_x_sujeto = []

nro_palabras_unicas_x_sujeto = []

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
    
    nro_palabras_x_sujeto.append(nro_palabras_sujeto)
    nro_palabras_unicas_x_sujeto.append(nro_palabras_unicas_sujeto)



mean_palabras_unicas_x_sujeto = np.mean(nro_palabras_unicas_x_sujeto, axis = 1)
std_palabras_unicas_x_sujeto = np.std(nro_palabras_unicas_x_sujeto, axis = 1)
mean_palabras_x_sujeto = np.mean(nro_palabras_x_sujeto, axis = 1)
std_palabras_x_sujeto = np.std(nro_palabras_x_sujeto, axis = 1)

nro_palabras_unicas_norm = []

for palab_unicas, media, std in zip(nro_palabras_unicas_x_sujeto, mean_palabras_unicas_x_sujeto, std_palabras_unicas_x_sujeto):
    nro_palabras_unicas_norm.append([(x-media) / std for x in palab_unicas])
    
nro_palabras_norm = []

for palab, media, std in zip(nro_palabras_x_sujeto, mean_palabras_x_sujeto, std_palabras_x_sujeto):
    nro_palabras_norm.append([(x-media) / std for x in palab])


#si quiero normalizar por el promedio de palabras x sujeto agrego _norm

palabras = nro_palabras_norm

palabras_unicas = nro_palabras_unicas_norm

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que en nro_palabras[0]


plt.figure(4), plt.clf()
for i in range(len(nro_palabras_x_sujeto)):
    for j in range(len(nro_palabras_x_sujeto[0])):
        color = cm.tab20(i)
        plt.plot(palabras[i][j], palabras_unicas[i][j],  marker = markers[j], color =  color)
    
    #Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(palabras[i], palabras_unicas[i])
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


columnas = sujetos
valores = correlation_por_sujeto

data = [valores]
df_sujetos = pd.DataFrame(data, columns=columnas)

print(df_sujetos)

        
#plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/Imagenes/por_sujeto/sujeto_{nro_sujeto}_{palabra}.png')



#%% correlacion nro palabras unicas y percepcion


Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]

percepcion_por_tema = []
nro_palabras_unicas_por_tema = []
correlation_por_tema = []
pvals_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path)
    
    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df_cuentopalabras["num_tot_palabras_unicas"] = (df_cuentopalabras["num_tot_palabras_unicas"]-mean_palabras_unicas_x_sujeto)/std_palabras_unicas_x_sujeto
    
    df_cuentopalabras = df_cuentopalabras.dropna()
    
    percepcion = df_cuentopalabras["cuanto_recordaste"]
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    percepcion_por_tema.append(percepcion)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que len(percepcion_por_tema)
plt.figure(5), plt.clf()
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


#%% Histograma de número de palabras por tema (de todos los sujetos)

Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]


nro_palabras_unicas_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm_con_spacy_hasta_sujeto_13/{tema}_cuentocadapalabra.csv'
    
    
    df_cuentopalabras = pd.read_csv(path)
    
    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df_cuentopalabras["num_tot_palabras_unicas"] = (df_cuentopalabras["num_tot_palabras_unicas"]-mean_palabras_unicas_x_sujeto) / std_palabras_unicas_x_sujeto
    
    df_cuentopalabras = df_cuentopalabras.dropna()
    
    nro_palabras_unicas = df_cuentopalabras["num_tot_palabras_unicas"]
    
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    
plt.figure(6), plt.clf()
nro_tema = 4
histograma = plt.hist(nro_palabras_unicas_por_tema[nro_tema].values, label = temas[nro_tema])
alturas = histograma[0]
bordes = histograma[1]
plt.plot((bordes[1:]+bordes[:-1])/2, alturas, 'o')
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Cuentas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#%%
markers = ['o', 's', 'P', 'v', 'H']

plt.figure(6), plt.clf()
for nro_tema in range(len(temas)):
    histograma = np.histogram(nro_palabras_unicas_por_tema[nro_tema].values) # alpha = 1 - nro_tema/8, edgecolor= color, color = color, linewidth = 2, label = temas[nro_tema]
    alturas = histograma[0]
    bordes = histograma[1]
    color = cm.tab20(2*nro_tema)
    plt.plot((bordes[1:]+bordes[:-1])/2, alturas, marker = markers[nro_tema], color = color, label = temas[nro_tema])
    
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.figure(7), plt.clf()
for nro_tema in range(len(temas)):
    plt.hist(nro_palabras_unicas_por_tema[nro_tema].values, alpha = 1 - nro_tema/8, edgecolor= color, color = color, linewidth = 2, label = temas[nro_tema])
    color = cm.tab20(2*nro_tema)
    
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#%% 
'''

analisis sentimiento

'''
#%% correlacion nro palabras unicas y sentimiento


Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"]

percepcion_por_tema = []
sentimiento_por_tema = []
intensidad_sentimiento_por_tema = [] #es la autopercibida
nro_palabras_unicas_por_tema = []
correlation_por_tema = []
pvals_por_tema = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/{tema}_sentimiento_contando.csv'
    
    
    df_total = pd.read_csv(path) #sentimiento y contando palabras
    
    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df_total["num_tot_palabras_unicas"] = (df_total["num_tot_palabras_unicas"]-mean_palabras_unicas_x_sujeto)/std_palabras_unicas_x_sujeto
    
    df_total = df_total.dropna()
    
    percepcion = df_total["cuanto_recordaste"]
    
    nro_palabras_unicas = df_total["num_tot_palabras_unicas"]
    
    sentimiento = df_total["SentiLeak"]
    
    intensidad_sentimiento = df_total["intensidad_emocion"]
    
    percepcion_por_tema.append(percepcion)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    sentimiento_por_tema.append(sentimiento)
    intensidad_sentimiento_por_tema.append(intensidad_sentimiento)

    

markers = ['o', 's', 'x', 'v', '+']#misma cantidad que len(percepcion_por_tema)
plt.figure(7), plt.clf()
for i in range(len(temas)):
    for j in range(len(intensidad_sentimiento_por_tema[0])):
        if j in intensidad_sentimiento_por_tema[i].index:
            color = cm.tab20(j)
            plt.plot(intensidad_sentimiento_por_tema[i][j], sentimiento_por_tema[i][j], marker = markers[i], color = color)
    
    # Calcular el coeficiente de correlación de Spearman
    correlation, p_value = spearmanr(intensidad_sentimiento_por_tema[i], sentimiento_por_tema[i])
    correlation_por_tema.append(correlation)
    pvals_por_tema.append(p_value)
    
# Crear líneas ficticias para la leyenda
lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]

# Agregar las líneas ficticias y las etiquetas a la leyenda
plt.legend(lines, temas)

plt.grid(True)
plt.xlabel('Percepción sentimiento', fontsize = 15)
plt.ylabel('Sentimiento con SentiLeak', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#%% total no x tema

percepcion_sentimiento = []

sentimiento_sentileak = []

for lista in intensidad_sentimiento_por_tema:
    percepcion_sentimiento.extend(lista)
    
for lista in sentimiento_por_tema:
    sentimiento_sentileak.extend(lista)


# Calcular el coeficiente de correlación de Spearman
correlation, p_value = spearmanr(percepcion_sentimiento, np.abs(sentimiento_sentileak))

# Imprimir el resultado
print("Coeficiente de correlación de Spearman:", correlation)
print("Valor p:", p_value)

#%% hago un gráfico de barras a ver si se aprecia mejor, grafico por sujeto

#nro_sujeto = 6
sujetos = range(1,14)

tema = ["campeones_del_mundo", "arabia", "presencial", "cfk", "antesdevenir"]

temas = ["a", "campeones", "arabia", "presencial", "cfk", "antesdevenir"]


for nro_sujeto in sujetos:

    percepcion_sentimiento_sujeto = []
    sentimiento_sentileak_sujeto = []
    
    for i, tem in enumerate(tema):
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/{tem}_sentimiento_contando.csv'
        
        
        df_total = pd.read_csv(path) #sentimiento y contando palabras
    
        percepcion_sentimiento_tema = df_total["intensidad_emocion"][nro_sujeto]
        percepcion_sentimiento_sujeto.append(percepcion_sentimiento_tema)
    
    
        sentimiento_sentileak_tema = df_total["SentiLeak"][nro_sujeto]
        sentimiento_sentileak_sujeto.append(sentimiento_sentileak_tema)
    
    
    
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
    ax1.set_ylabel('SetiLeak', color=color)
    ax1.bar(nro_tema, np.abs(sentimiento_sentileak_sujeto), color=color, width=0.4, label = 'SentiLeak')
    #ax1.bar(nro_tema, nro_palabras_unicas_sujeto, '.', color='lightcoral', label = 'unicas')
    ax1.legend()
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Configurar la segunda serie (eje y derecho)
    ax2 = ax1.twinx()  # Crea un segundo eje y
    color = 'tab:blue'
    ax2.set_ylabel('Autopercepción de cuánto sintieron', color=color)
    ax2.bar(nro_tema + corrimiento, percepcion_sentimiento_sujeto, color=color, width=0.4)
    ax2.tick_params(axis='y', labelcolor=color)
    
    #titulo
    plt.title(f"Sujeto {nro_sujeto}")
    
    # Ajustar el tamaño del gráfico
    fig.tight_layout()
    
    
    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/Imagenes/por_sujeto/sujeto_{nro_sujeto}_sentiLeak.png')
