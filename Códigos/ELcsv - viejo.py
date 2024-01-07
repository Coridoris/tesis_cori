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
import ast

def get_max_sentiment(diccionario):
    max_key = max(diccionario, key=diccionario.get)
    
    if max_key == 'NEU':
        return 0
    elif max_key == 'POS':
        return 1
    elif max_key == 'NEG':
        return -1

#%%
'''
Normalizamos número de palabras con el zscore = (datos - su media) / su desviación estandar
Usamos el número de palabras únicas (es equivalente a palabras totales)
'''
Sujetos = [f'Sujeto {i}' for i in range(1, 31)]

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

temas_tabla = ["campeones", "presencial", "cfk", "arabia", "filler"]


nro_palabras_unicas_por_tema_para_normalizar = []
nro_palabras_por_tema_para_normalizar = []

palabras = ['totales', 'unicas']


for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/pysentimiento_contando_{tema}.csv'
        
    df = pd.read_csv(path)
    
    nro_palabras_unicas_para_normalizar = df["num_tot_palabras_unicas"]

    nro_palabras_unicas_por_tema_para_normalizar.append(nro_palabras_unicas_para_normalizar)

    nro_palabras_para_normalizar = df["num_tot_palabras"]

    nro_palabras_por_tema_para_normalizar.append(nro_palabras_para_normalizar)
    

nro_palabras_unicas_x_sujeto = list(zip(*nro_palabras_unicas_por_tema_para_normalizar))
mean_palabras_unicas_x_sujeto = np.mean(nro_palabras_unicas_x_sujeto, axis = 1)
std_palabras_unicas_x_sujeto = np.std(nro_palabras_unicas_x_sujeto, axis = 1)

nro_palabras_x_sujeto = list(zip(*nro_palabras_por_tema_para_normalizar))
mean_palabras_x_sujeto = np.mean(nro_palabras_x_sujeto, axis = 1)
std_palabras_x_sujeto = np.std(nro_palabras_x_sujeto, axis = 1)

nro_palabras_unicas_norm = []

for palab_unicas, media, std in zip(nro_palabras_unicas_x_sujeto, mean_palabras_unicas_x_sujeto, std_palabras_unicas_x_sujeto):
    nro_palabras_unicas_norm.append([(x-media) / std for x in palab_unicas])
    
nro_palabras_norm = []

for palab, media, std in zip(nro_palabras_x_sujeto, mean_palabras_x_sujeto, std_palabras_x_sujeto):
    nro_palabras_norm.append([(x-media) / std for x in palab])
    
data_dict_list = []
for i, (sujeto_data1, sujeto_data2) in enumerate(zip(nro_palabras_unicas_norm, nro_palabras_norm), start=1):
    for j, (valor1, valor2) in enumerate(zip(sujeto_data1, sujeto_data2)):
        data_dict_list.append({
            "Sujeto": i,
            "Condición": temas[j],
            "Nro de palabras únicas": valor1,
            "Nro de palabras totales": valor2
        })

# Crear un DataFrame de pandas
df = pd.DataFrame(data_dict_list)

# Guardar el DataFrame en un archivo CSV
#df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/contando_palabras_todos_temas.csv', index=False)

percepcion_recuerdo_por_tema = []
pysent_pos_por_tema = []
pysent_neg_por_tema = []
#pysent_neu_por_tema = []
pysent_valencia_por_tema = []
intensidad_sentimiento_por_tema = [] #es la autopercibida
tipo_sentimiento_por_tema = [] #es la autopercibida de -1 a 1
nro_palabras_unicas_por_tema = []
nro_palabras_por_tema = []
intensidad_menos5_5_por_tema = [] #es la autopercibida de -5 a 5
palabra_unicas = []

intensidad_por_tema = []

num_sust = []
num_verb = []
num_adj = []
num_numeral = []
num_propn = []
num_advs = []
#coherencia_evolucion_normalizada = []
#coherencia_normalizada = []

internos = []
externos = []
totalwordcount = []


#dfs_red_cat = []

#dfs_red_atr = []

dfs_red = []

dfs_cohe = []

dfs_clasificaciones = []

dfs_primter = []

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/pysentimiento_contando_{tema}.csv'
    
    path_clasificaciones = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{tema}_sinMeacuerdo.csv'

    path_coherencia = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion_norm_ELcsv.csv'
    
    path_ep_sem = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Clasificacion_episodico_semantico/automated_autobio_scored_{tema}_ingles.csv'
    
    #path_redes_categorias = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Categorias seis/df_red_sustverbadjetc_{tema}_sin_meacuerdo.csv'
    
    #path_redes_atributos = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Atributos/df_red_atributos_{tema}.csv'
    
    path_red = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/Atributos/df_red_norm_{tema}.csv'
    
    path_primera_tercera = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Primera-tercera persona/{tema}_prim_ter_persona.csv'
    
    df = pd.read_csv(path) #sentimiento y contando palabras
    
    df_clasificaciones = pd.read_csv(path_clasificaciones)
    
    df_coherencia = pd.read_csv(path_coherencia)
    
    df_ep_sem = pd.read_csv(path_ep_sem)
    
    df_primera_tercera = pd.read_csv(path_primera_tercera)
    
    # df_red_atributos = pd.read_csv(path_redes_atributos)
    
    # df_red_categorias = pd.read_csv(path_redes_categorias)
    
    # df_red_categorias.rename(columns={"sujeto": "Sujetos"}, inplace=True)
    
    # columns_cat = ["Sujetos", "num noun", "num verb", "num adj", "num advs", "num numeral", "num propn", "('ADJ', 'NOUN')", "('NOUN', 'ADJ')",	"('ADJ', 'PROPN')",	"('PROPN', 'PROPN')", "('PROPN', 'VERB')", "('VERB', 'NOUN')", "('ADV', 'VERB')", "('NOUN', 'ADV')", "('ADJ', 'ADV')", "('VERB', 'VERB')", "('VERB', 'ADV')", "('NUM', 'NOUN')", "('ADV', 'ADJ')", "('ADJ', 'ADJ')", "('NOUN', 'VERB')", "('VERB', 'PROPN')", "('PROPN', 'ADV')", "('NOUN', 'PROPN')", "('PROPN', 'NOUN')", "('ADJ', 'VERB')", "('ADV', 'ADV')", "('ADV', 'NOUN')", "('ADV', 'PROPN')", "('VERB', 'NUM')", "('VERB', 'ADJ')", "('NUM', 'ADV')", "('ADJ', 'NUM')", "('PROPN', 'ADJ')", "('NOUN', 'NOUN')", "('PROPN', 'NUM')", "('NUM', 'PROPN')", "('NOUN', 'NUM')", "('NUM', 'VERB')", "('NUM', 'NUM')", "('NUM', 'ADJ')", "('ADV', 'NUM')"]

    # # Encuentra las columnas que no existen en el DataFrame original
    # columnas_faltantes = [col for col in columns_cat if col not in df_red_categorias.columns]
    
    # # Crea columnas faltantes con ceros en el DataFrame original
    # for col in columnas_faltantes:
    #     df_red_categorias[col] = 0

    # df_red_categ = df_red_categorias[columns_cat]
    
    # df_red_categ["Condición"] = tema
    
    # nuevo_orden = ['Sujetos', 'Condición', 'num noun', 'num verb', 'num adj', 'num advs', 'num numeral',
    #        'num propn', "('ADJ', 'NOUN')", "('NOUN', 'ADJ')",	"('ADJ', 'PROPN')",	"('PROPN', 'PROPN')", 
    #        "('PROPN', 'VERB')", "('VERB', 'NOUN')", "('ADV', 'VERB')", "('NOUN', 'ADV')", "('ADJ', 'ADV')", 
    #        "('VERB', 'VERB')", "('VERB', 'ADV')", "('NUM', 'NOUN')", "('ADV', 'ADJ')", "('ADJ', 'ADJ')", "('NOUN', 'VERB')", 
    #        "('VERB', 'PROPN')", "('PROPN', 'ADV')", "('NOUN', 'PROPN')", "('PROPN', 'NOUN')", "('ADJ', 'VERB')", 
    #        "('ADV', 'ADV')", "('ADV', 'NOUN')", "('ADV', 'PROPN')", "('VERB', 'NUM')", "('VERB', 'ADJ')", "('NUM', 'ADV')", 
    #        "('ADJ', 'NUM')", "('PROPN', 'ADJ')", "('NOUN', 'NOUN')", "('PROPN', 'NUM')", "('NUM', 'PROPN')", 
    #        "('NOUN', 'NUM')", "('NUM', 'VERB')", "('NUM', 'NUM')", "('NUM', 'ADJ')", "('ADV', 'NUM')"]

    # # Reordena las columnas del DataFrame
    # df_red_categ = df_red_categ[nuevo_orden]
    
    #dfs_red_cat.append(df_red_categ)
    
    #dfs_red_atr.append(df_red_atributos)
    
    dfs_red.append(pd.read_csv(path_red))
    
    dfs_cohe.append(df_coherencia)
    
    dfs_clasificaciones.append(df_clasificaciones)
    
    dfs_primter.append(df_primera_tercera)

    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df["num_tot_palabras_unicas"] = (df["num_tot_palabras_unicas"]-mean_palabras_unicas_x_sujeto)/std_palabras_unicas_x_sujeto
    
    #df = df.dropna()
    
    percepcion_recuerdo = df["cuanto_recordaste"]
    
    nro_palabras_unicas = df["num_tot_palabras_unicas"]
    

    intensidad_sentimiento = df["intensidad_emocion"]
    
    tipo_sentimiento = df["tipo_emocion_num"]
    
    intensidad_menos5_5 = df["intensidad_y_tipo"]

    df["pysent_curdo"] = df["pysent_curdo"].apply(eval)
    
    df['valencia_pysent'] = df['pysent_curdo'].apply(get_max_sentiment)
    
    py_pos = df["pysent_curdo"].apply(lambda x: x['POS'])
    
    py_neg = df["pysent_curdo"].apply(lambda x: x['NEG'])
    
    py_neu = df["pysent_curdo"].apply(lambda x: x['NEU'])
    
    py_int = df["pysent_curdo"].apply(lambda x: x['POS']) + df["pysent_curdo"].apply(lambda x: x['NEG'])
            

    percepcion_recuerdo_por_tema.append(percepcion_recuerdo)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    intensidad_sentimiento_por_tema.append(intensidad_sentimiento)
    tipo_sentimiento_por_tema.append(tipo_sentimiento)
    intensidad_menos5_5_por_tema.append(intensidad_menos5_5) 

    intensidad_por_tema.append(py_int)
    
    pysent_pos_por_tema.append(py_pos)
    pysent_neg_por_tema.append(py_neg)
    #pysent_neu_por_tema.append(py_neu)
    pysent_valencia_por_tema.append(df['valencia_pysent'])
    
    palabra_unicas.append(df['num_tot_palabras_unicas'])
    
    # num_sust.append(df_clasificaciones['num noun'])
    # num_verb.append(df_clasificaciones['num verb'])
    # num_adj.append(df_clasificaciones['num adj'])
    # num_numeral.append(df_clasificaciones['num numeral'])
    # num_propn.append(df_clasificaciones['num propn'])
    # num_advs.append(df_clasificaciones['num advs'])
    
    #coherencia_evolucion_normalizada.append(df_coherencia['coherencia_evolucion_normalizada'])
    
    #df_coherencia['coherencia_evolucion_normalizada'] = df_coherencia['coherencia_evolucion_normalizada'].apply(ast.literal_eval)
    
    #df_coherencia['cada oracion'] = df_coherencia['coherencia_evolucion_normalizada'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

    #coherencia_normalizada.append(df_coherencia['cada oracion'])
    
    internos.append(df_ep_sem['numInt_preds'])
    externos.append(df_ep_sem['numExt_preds'])
    totalwordcount.append(df_ep_sem['totalWordCount'])
    
data_dict_list = []
for i, tema in enumerate(temas):
    for sujeto, percepcion in enumerate(percepcion_recuerdo_por_tema[i]):
        data_dict_list.append({
            "Sujetos": sujeto + 1,  # Sumamos 1 para que los sujetos vayan de 1 a 30
            "Condición": tema,
            "Percepción cuánto recuerdo": percepcion,
            "Valencia percibida recuerdo": tipo_sentimiento_por_tema[i][sujeto],
            "Valencia percibida -5 a +5": intensidad_menos5_5_por_tema[i][sujeto],
            "Intensidad percibida recuerdo": intensidad_sentimiento_por_tema[i][sujeto],
            "Intensidad pysent": intensidad_por_tema[i][sujeto],
            "Positivo pysennt": pysent_pos_por_tema[i][sujeto],
            "Negativo pysent": pysent_neg_por_tema[i][sujeto],
            "Nro palabras únicas": nro_palabras_unicas_por_tema[i][sujeto],
            #"Neutro pysennt": pysent_neu_por_tema[i][sujeto],
            # "Nro sust": num_sust[i][sujeto],
            # "Nro verb": num_verb[i][sujeto],
            # "Nro adj": num_adj[i][sujeto],
            # "Nro numeral": num_numeral[i][sujeto],
            # "Nro propn": num_propn[i][sujeto],
            # "Nro advs": num_advs[i][sujeto],
            #"Coherencia norm": coherencia_normalizada[i][sujeto],
            #"Evolucion coherencia norm": coherencia_evolucion_normalizada[i][sujeto],
            "Detalles internos": internos[i][sujeto],
            "Detalles externos": externos[i][sujeto],
            "Total word count ruben": totalwordcount[i][sujeto]
        })
#%%
# Crear un DataFrame de pandas
df2 = pd.DataFrame(data_dict_list)

#df_red_categoria_temas = pd.concat(dfs_red_cat, ignore_index=True)

#df_red_atributos_temas = pd.concat(dfs_red_atr, ignore_index=True)

#df_redes_temas = df_red_categoria_temas.merge(df_red_atributos_temas, on=["Sujetos", "Condición"])

df_red_temas = pd.concat(dfs_red, ignore_index=True)

df_coher_temas = pd.concat(dfs_cohe, ignore_index = True)

df_clasificaciones_temas = pd.concat(dfs_clasificaciones, ignore_index = True)

df_primter_temas = pd.concat(dfs_primter, ignore_index = True)

merged1_df = df_coher_temas.merge(df_red_temas, on=["Sujetos", "Condición"])

merged2_df = df2.merge(df_clasificaciones_temas, on=["Sujetos", "Condición"])

merged3_df = merged2_df.merge(df_primter_temas, on=["Sujetos", "Condición"])

merged_df = merged3_df.merge(merged1_df, on=["Sujetos", "Condición"])


merged_df['Detalles internos norm'] = merged_df['Detalles internos']/merged_df['Total word count ruben']

merged_df['Detalles externos norm'] = merged_df['Detalles externos']/merged_df['Total word count ruben']

merged_df = merged_df.drop(['Detalles internos', 'Detalles externos', 'Total word count ruben'], axis=1)

autopercepcion = 'si'

if autopercepcion == 'si':
    
    # Guardar el DataFrame en un archivo CSV
    merged_df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv', index=False)
    
else:
    merged_df = merged_df.drop(['Percepción cuánto recuerdo', 'Valencia percibida recuerdo', 'Valencia percibida -5 a +5', 'Intensidad percibida recuerdo'], axis=1)
 
    # Guardar el DataFrame en un archivo CSV
    merged_df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv', index=False)


#%% Hago un csv de cada tema

condiciones = merged_df["Condición"].unique()

# Guardar un archivo CSV para cada condición
for condicion in condiciones:
    # Filtrar el DataFrame por la condición actual
    df_condicion = merged_df[merged_df["Condición"] == condicion]
    
    if autopercepcion == 'si':
        # Crear el nombre del archivo CSV
        nombre_archivo = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'
    
    else:
        # Crear el nombre del archivo CSV
        nombre_archivo = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'
    
        
    # Guardar el DataFrame filtrado en el archivo CSV
    df_condicion.to_csv(nombre_archivo, index=False)
