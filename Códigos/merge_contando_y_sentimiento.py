# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:03:17 2023

@author: corir
"""
'''
vamos a mergear los dataframe de sentimiento y contando palabras
'''
import pandas as pd


temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

tema = temas[3]


for i in range(len(temas)):
    path_sentimiento = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/pysentimiento/{temas[i]}_pysentimiento.csv' #sentimiento_todos_metodos_{temas[i]}
    path_contando =  f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/Lemm con stanza-texto limpio-primeros30/{temas[i]}_cuentocadapalabra.csv'
        
    
    df_sentimiento = pd.read_csv(path_sentimiento)
    
    df_contando = pd.read_csv(path_contando)
    
    # Crear un diccionario de mapeo
    
    # Crear un diccionario de mapeo
        
    mapeo_1 = {'Negativas': 'Negativa', 'Neutras': 'Neutra', 'Positivas': 'Positiva'}
        
    df_contando["tipo_emocion_"] = df_contando["tipo_emocion"].map(mapeo_1)
        
    mapeo_2 = {'Negativa': -1, 'Negativas': -1, 'Neutra':0, 'Neutras':0, 'Positiva': 1, 'Positivas': 1}
        
    df_contando["tipo_emocion_num"] = df_contando["tipo_emocion"].map(mapeo_2)
    
    # Definir una función para cambiar los valores
    def cambiar_valor(row):
        if row["tipo_emocion"] == 'Negativa':
            return -row["intensidad_emocion"]
        elif row["tipo_emocion"] == 'Neutra':
            return 0
        else:
            return row["intensidad_emocion"]
    
    # Aplicar la función a la columna y crear una nueva columna con los valores cambiados
    df_contando['intensidad_y_tipo'] = df_contando.apply(cambiar_valor, axis=1)
    
    # Obtener el nombre de la columna que deseas mover
    columna_objetivo_1 = 'intensidad_y_tipo'
    columna_objetivo_2 = 'tipo_emocion_num'
    
    # Obtener el índice del lugar donde deseas mover la columna
    nuevo_indice = 6
    nuevo_indice_ = 4
    
    # Extraer el nombre de todas las columnas
    columnas = df_contando.columns.tolist()
    
    # Remover la columna objetivo de la lista
    columnas.remove(columna_objetivo_1)
    columnas.remove(columna_objetivo_2)
    
    # Insertar la columna objetivo en el nuevo índice
    columnas.insert(nuevo_indice, columna_objetivo_1)
    columnas.insert(nuevo_indice_, columna_objetivo_2)
    
    # Reordenar las columnas del DataFrame
    df_contando = df_contando[columnas]
    
    df_sentimiento.insert(4, "num_tot_palabras", df_contando['num_tot_palabras'], True)
    df_sentimiento.insert(4, "num_tot_palabras_unicas", df_contando['num_tot_palabras_unicas'], True)
    df_sentimiento.insert(4, "cuanto_recordaste", df_contando['cuanto_recordaste'], True)
    df_sentimiento.insert(4, "tipo_emocion", df_contando['tipo_emocion'], True)
    df_sentimiento.insert(4, "intensidad_emocion", df_contando['intensidad_emocion'], True)
    df_sentimiento.insert(4, "intensidad_y_tipo", df_contando['intensidad_y_tipo'], True)
    df_sentimiento.insert(4, "tipo_emocion_num", df_contando['tipo_emocion_num'], True)
    
    #guardamos la data
    
    df_sentimiento.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/pysentimiento_contando_{temas[i]}.csv')
    
    

