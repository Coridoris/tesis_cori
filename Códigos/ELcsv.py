# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:18:52 2023

@author: corir
"""

'''
vamos a mergear todo en un único dataset
'''

import pandas as pd
import numpy as np
    
#%% el santo trial

entrevista = 'Segunda'

autopercepcion = 'no'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

eliminando_outliers = True

#%%

dfs_autopercepcion = []

dfs_contenido = []

dfs_sent = []

dfs_mem = []

dfs_est = []

for ind, tema in enumerate(temas):
    
    path_autopercepcion = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables autopercepcion/variables_autopercepcion_{tema}.csv'

    path_contenido = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables contenido/variables_contenido_{tema}.csv'
    
    if eliminando_outliers == True:
        
        path_contenido = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables contenido/variables_contenido_sinoutliers_{tema}.csv'
        
    path_sentimiento = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables sentimiento nuevas/variables_sentimiento_{tema}.csv'

    path_estructura =  f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables estructurales/variables_estructurales_{tema}.csv'
    
    path_memoria =  f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables memoria/variables_memoria_{tema}.csv'
    
    dfs_autopercepcion.append(pd.read_csv(path_autopercepcion))
    
    dfs_contenido.append(pd.read_csv(path_contenido))
    
    dfs_sent.append(pd.read_csv(path_sentimiento))
    
    dfs_est.append(pd.read_csv(path_estructura))
    
    dfs_mem.append(pd.read_csv(path_memoria))
#%%

df_autop = pd.concat(dfs_autopercepcion, ignore_index = True)

df_cont = pd.concat(dfs_contenido, ignore_index=True)

df_sent = pd.concat(dfs_sent, ignore_index = True)

df_est = pd.concat(dfs_est, ignore_index = True)

df_mem = pd.concat(dfs_mem, ignore_index = True)

merged1_df = df_autop.merge(df_cont, on=["Sujetos", "Condición"])

merged2_df = df_sent.merge(df_est, on=["Sujetos", "Condición"], how='left')

merged3_df = merged2_df.merge(df_mem, on=["Sujetos", "Condición"], how='left')

merged_df = merged1_df.merge(merged3_df, on=["Sujetos", "Condición"])

if eliminando_outliers == True:
    nan_mask = np.isnan(merged_df["num_palabras_unicas_norm"])

    # Establece NaN en toda la fila, excepto en las columnas "Sujetos" y "Condición"
    merged_df.loc[nan_mask, merged_df.columns.difference(["Sujetos", "Condición"])] = np.nan



if autopercepcion == 'si':
    
    if eliminando_outliers == True:
       merged_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv', index=False)
    else:
       # Guardar el DataFrame en un archivo CSV
       merged_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv', index=False)
        
else:
    merged_df = merged_df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis=1)
     
    if eliminando_outliers == True:
        merged_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv', index=False)
    else:
        # Guardar el DataFrame en un archivo CSV
        merged_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv', index=False)


#%% Hago un csv de cada tema

condiciones = merged_df["Condición"].unique()

# Guardar un archivo CSV para cada condición
for condicion in condiciones:
    # Filtrar el DataFrame por la condición actual
    df_condicion = merged_df[merged_df["Condición"] == condicion]
    
    if autopercepcion == 'si':
        # Crear el nombre del archivo CSV
        nombre_archivo = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_{condicion}.csv'
    
    else:
        # Crear el nombre del archivo CSV
        nombre_archivo = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_{condicion}.csv'
    
        
    # Guardar el DataFrame filtrado en el archivo CSV
    df_condicion.to_csv(nombre_archivo, index=False)
    
#%% quiero un csv con ambos tiempos donde agrego la columna Tiempo que dice 1 o 2

dfs_tiempos = []
for i, entrevista in enumerate(['Primera', 'Segunda']):
    if autopercepcion == 'si':
        if eliminando_outliers == True:
            path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv'
        else:
            #abro_los_archivos
            path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'
        df = pd.read_csv(path)
        
        # Insertar la nueva columna en la posición 3 (index 2)
        df.insert(2, 'Tiempo',i+1)

        dfs_tiempos.append(df)
    else:
        if eliminando_outliers == True:
            path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv'
        else:
            
            path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'
        df = pd.read_csv(path)
        
        # Insertar la nueva columna en la posición 3 (index 2)
        df.insert(2, 'Tiempo',i+1)

        dfs_tiempos.append(df)


df_tiempos = pd.concat(dfs_tiempos, ignore_index = True)

#lo guardo
if autopercepcion == 'si':
    if eliminando_outliers == True:
        df_tiempos.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_dostiempos.csv', index=False)
    else:
        # Guardar el DataFrame en un archivo CSV
        df_tiempos.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_dostiempos.csv', index=False)
    
else:
    if eliminando_outliers == True:
        df_tiempos.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_dostiempos.csv', index=False)
    else:     
        # Guardar el DataFrame en un archivo CSV
        df_tiempos.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_dostiempos.csv', index=False)

