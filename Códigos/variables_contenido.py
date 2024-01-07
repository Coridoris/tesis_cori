# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:42:44 2023

@author: corir
"""
#LIBRERIAS

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import es_core_news_sm
from tqdm import tqdm
#pip install docx

nlp = es_core_news_sm.load()
#%% funciones

    
def clasificacion_texto(text):
    if type(text) != str:
        rta = np.nan
    else:
        doc = nlp(text)
        rta = [(w.text, w.pos_) for w in doc]
    return rta

def eliminar_frase(texto, frase_a_eliminar):
    if type(texto) != str:
        texto_sin_frase = np.nan
    else:
        # Reemplaza la frase a eliminar con una cadena vacía
        texto_sin_frase = texto.replace(frase_a_eliminar, "")
    return texto_sin_frase

def contar_palabras(texto):
    if type(texto) != str:
        rta = np.nan
    else:
        # Divide el texto en palabras utilizando el espacio en blanco como separador
        palabras = texto.split()
        # Cuenta el número de palabras y devuelve el resultado
        rta = len(palabras)
    return rta

def extract_nouns(lista, etiquet):
    if type(lista) != list:
        return np.nan
    else:
        return [palabra for palabra, etiqueta in lista if etiqueta == etiquet] if isinstance(lista, list) else []

def largo(x):
    if type(x) != list:
        return np.nan
    else:
        return len(x)


def clasificacion_tema(tema, frase_a_eliminar = None):
    '''
    si queres eliminar frases ponerlas en una lista 
    si queres el núm único de estas clasificaciones o con el texto lemm redirigite a conteo_sust_verb_adj_spyCy.py
    ahi vas a encontrar cómo hacerlo
    '''
        
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
    
    df_textos = pd.read_csv(path)
    
    if frase_a_eliminar != None:
        for i in range(len(frase_a_eliminar)):
            df_textos["texto_crudo"] = df_textos["texto_crudo"].apply(eliminar_frase, args=(frase_a_eliminar[i],))
            
    df_textos["num_total_palabras"] = df_textos["texto_crudo"].apply(contar_palabras)
    
    df_textos["clasificacion_total"] = df_textos["texto_crudo"].apply(clasificacion_texto)
    
    df_textos['nouns'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("NOUN",))
    
    #df_textos['nouns'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'NOUN'] if pd.notnull(lista).any() else lista)
    
    #si quiero mas de una etiqueta, por ej si quiero sumar PROPN a sustantivos correria algo asi:
    #df_textos['nouns'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta in ['NOUN', 'PRON']])

    df_textos['verbs'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("VERB",))
    
    df_textos['adjs'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("ADJ",))
    
    df_textos['advs'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("ADV",))
    
    df_textos['numeral'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("NUM",))
    
    df_textos['propn'] = df_textos["clasificacion_total"].apply(extract_nouns, args=("PROPN",))
    
    
    df_textos['num noun'] = df_textos['nouns'].apply(largo)
    df_textos['num verb'] = df_textos['verbs'].apply(largo)
    df_textos['num adj'] = df_textos['adjs'].apply(largo)
    
    df_textos['num advs'] = df_textos['advs'].apply(largo)
    df_textos['num numeral'] = df_textos['numeral'].apply(largo)
    df_textos['num propn'] = df_textos['propn'].apply(largo)
    
    df_textos['num noun norm'] = df_textos['num noun']/df_textos["num_total_palabras"]
    df_textos['num verb norm'] = df_textos['num verb']/df_textos["num_total_palabras"]
    df_textos['num adj norm'] = df_textos['num adj']/df_textos["num_total_palabras"]
    
    df_textos['num advs norm'] = df_textos['num advs']/df_textos["num_total_palabras"]
    df_textos['num numeral norm'] = df_textos['num numeral']/df_textos["num_total_palabras"]
    df_textos['num propn norm'] = df_textos['num propn']/df_textos["num_total_palabras"]
       
    #si queremos que nos de todo el txt hay que poner cualquier otra cosa en la variable todas, 
    #sino nos va a dar un csv solo con lo importante
        
    # Selecciona las columnas que queremos
    columnas_seleccionadas = ['Sujetos', 'num noun norm', 'num verb norm', 'num adj norm', 'num advs norm', 'num numeral norm', 'num propn norm']
    
    #reescribe el dataframe
    df_textos = df_textos[columnas_seleccionadas]
    
    df_textos.insert(1, 'Condición', tema)
        
    return df_textos

def contar_palabras_en_texto(texto, lista_palabras):
    if type(texto) != str:
        rta = np.nan
    else:
        # Convertir el texto a minúsculas para comparaciones sin distinción entre mayúsculas y minúsculas
        texto = texto.lower()
    
        # Dividir el texto en palabras
        palabras = texto.split()
    
        # Inicializar el contador de palabras
        contador = 0
    
        # Contar las palabras que están en la lista proporcionada
        for palabra in palabras:
            if palabra in lista_palabras:
                contador += 1
        #si lo quiero normalizado por el num total de palabras cambiar a contador/len(palabras)
        rta = contador/len(palabras)
    return rta


def primera_tercera_persona(tema, pronombres_primera, pronombres_tercera ,frase_a_eliminar = None):
    '''
    a esta función le metes un csv de textos y te devuelve un csv con nro sujeto, condicion, nro_palabras_prim_persona, nro_palabras_tercera_persona
    o sea, algo listo para meter en ELcsv
    '''
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
    
    if frase_a_eliminar != None:
        for i in range(len(frase_a_eliminar)):
            df_textos["texto_crudo"] = df_textos["texto_crudo"].apply(eliminar_frase, args=(frase_a_eliminar[i],))
    
    df_textos["tot"] = df_textos["texto_crudo"].apply(contar_palabras)
    
    df_textos["primera_persona_norm"] = df_textos["texto_crudo"].apply(contar_palabras_en_texto, args=(pronombres_primera,))
    
    df_textos["tercera_persona_norm"] = df_textos["texto_crudo"].apply(contar_palabras_en_texto, args=(pronombres_tercera,))
    
    #si lo quiero normalizado por el num total de palabras cambiar la función contar_palabras_en_texto el return por lo comentado
    #y ya va a estar directamente normalizado
    #si lo quiero normalizado por el num total de palabras en primera+tercera usar lo que sigue
    #df_textos["tot_primera+tercera"] = df_textos["primera_persona"] + df_textos["tercera_persona"]
    
    #df_textos["primera_persona_norm"] = df_textos["primera_persona"]/df_textos["tot_primera+tercera"]
    
    #df_textos["tercera_persona_norm"] = df_textos["tercera_persona"]/df_textos["tot_primera+tercera"]
    
    # Selecciona las columnas que queremos
    columnas_seleccionadas = ['Sujetos', 'primera_persona_norm', 'tercera_persona_norm']#, "tot_primera+tercera", "tot"]
    
    #reescribe el dataframe
    df_textos = df_textos[columnas_seleccionadas]
    
    df_textos.insert(1, 'Condición', tema)

    return df_textos

#%% santo trial

temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

nro_sujetos = 65

 # Lista de los sujetos   
Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
entrevista = "Primera"

eliminando_outliers = True
    
#%% datos
    
dfs = []
for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
    
    dfs.append(pd.read_csv(path))
    
#%% número de palabras únicas

nro_palabras_unicas_por_tema = []
nro_palabras_por_tema = []
nro_palabras_unicas_por_tema_para_normalizar = []
nro_palabras_por_tema_para_normalizar = []
palabra_unicas = []
palabras_unicas_norm = []

for i in range(len(temas)):
    # Inicializo el modelo 
    cv = CountVectorizer()
    #relleno con nans porque sino después no me deja
    
    dfs[i]['lematizado'] = dfs[i]['lematizado'].fillna('')

    # Ajustamos el modelo y lo aplicamos al texto de nuestro dataframe generando una matriz esparsa
    data_cv = cv.fit_transform(dfs[i].lematizado)
    # Nos creamos un dataframe transformando a densa la matriz generada recien que tiene como columnas las palabras (terminos) y como filas los documentos
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    # Le asignamos los indices del dataframe anterior
    data_dtm.index = dfs[i].index

    df_contandopalabras = data_dtm.copy(deep=True)
    
    df_contandopalabras_paracalculo = data_dtm.copy(deep=True)

    # cantidad de palabras total que dijo cada sujeto
    
    palabras_tot = data_dtm.sum(axis='columns')
    
    df_contandopalabras.insert(0, "num_tot_palabras", palabras_tot, True)
    

    #cantidad de palabras unicas total que dijo cada sujeto
    
    df_contandopalabras_paracalculo[df_contandopalabras_paracalculo > 1] = 1
    
    palabras_unicas_tot = df_contandopalabras_paracalculo.sum(axis='columns')
    
    df_contandopalabras.insert(1, "num_tot_palabras_unicas", palabras_unicas_tot, True)
    
    nro_palabras_unicas_por_tema_para_normalizar.append(df_contandopalabras["num_tot_palabras_unicas"])

    nro_palabras_por_tema_para_normalizar.append(df_contandopalabras["num_tot_palabras"])

if eliminando_outliers == True:
    if entrevista == "Primera": 
        indices_a_reemplazar = [[3,25,30,39,52,54,58], [3,21,30,35,39,44,52], [3,6,21,30,39,58], [8,18,21,30,31,39,49,57,60], [3,9,25,35,39,64]]
    
    if entrevista == "Segunda":
        indices_a_reemplazar = [[3,8,39,41,43,48,52,54,61], [3,8,21,39,40,41,43,52,61], [3,4,6,8,34,39,41,43,52,61], [3,9,39,41,43,47,49,52,60,61], [3,8,9,21,39,41,43,52,54,61,62]]


for i, list_tema in enumerate(nro_palabras_unicas_por_tema_para_normalizar):
    for indice in indices_a_reemplazar[i]:
        nro_palabras_unicas_por_tema_para_normalizar[i][indice] = np.nan

nro_palabras_unicas_x_sujeto = list(zip(*nro_palabras_unicas_por_tema_para_normalizar))
mean_palabras_unicas_x_sujeto = np.nanmean(nro_palabras_unicas_x_sujeto, axis = 1)
std_palabras_unicas_x_sujeto = np.nanstd(nro_palabras_unicas_x_sujeto, axis = 1)

nro_palabras_x_sujeto = list(zip(*nro_palabras_por_tema_para_normalizar))
mean_palabras_x_sujeto = np.nanmean(nro_palabras_x_sujeto, axis = 1)
std_palabras_x_sujeto = np.nanstd(nro_palabras_x_sujeto, axis = 1)

nro_palabras_unicas_norm = []


for palab_unicas, media, std in zip(nro_palabras_unicas_x_sujeto, mean_palabras_unicas_x_sujeto, std_palabras_unicas_x_sujeto):
    nro_palabras_unicas_norm.append([(x-media) / std if std != 0  else 0 for x in palab_unicas])
    
nro_palabras_norm = []

for palab, media, std in zip(nro_palabras_x_sujeto, mean_palabras_x_sujeto, std_palabras_x_sujeto):
    nro_palabras_norm.append([(x-media) / std if std != 0 else 0 for x in palab])

for ind, tema in enumerate(temas):
    numerador = nro_palabras_unicas_por_tema_para_normalizar[ind]- mean_palabras_unicas_x_sujeto
    
    posiciones = np.where(numerador == -mean_palabras_unicas_x_sujeto)

    # Haz que en esas posiciones pre_numerador sea cero
    numerador[posiciones[0]] = np.nan
    
    palabras_unicas_norm.append(numerador/std_palabras_unicas_x_sujeto)
    
    
#%%

pronombres_primera = ['yo', 'me', 'mi', 'mí', 'conmigo', 'nosotros', 'nos', 'nosotras', 'nuestro', 'nuestra']

pronombres_tercera = ['él', 'lo', 'le', 'se', 'sí', 'consigo', 'ella', 'la', 'ello', 'lo', 'ellos', 'ellas', 'las', 'los', 'les', 'se', 'usted', 'ustedes']

for i in tqdm(range(len(temas))):

    df_clasificaciones = clasificacion_tema(temas[i], ["Me acuerdo", "Recuerdo", "me acuerdo", "recuerdo"])
    
    df_primera_tercera = primera_tercera_persona(temas[i], pronombres_primera, pronombres_tercera ,frase_a_eliminar = ["Me acuerdo", "Recuerdo", "me acuerdo", "recuerdo"])
    
    df_primera_tercera.insert(2, 'num_palabras_unicas_norm', palabras_unicas_norm[i])
    
    df_contenido = df_primera_tercera.merge(df_clasificaciones, on=["Sujetos", "Condición"])
    
    if eliminando_outliers == True:
        df_contenido.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables contenido/variables_contenido_sinoutliers_{temas[i]}.csv', index=False)    
    #else:
       # df_contenido.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables contenido/variables_contenido_{temas[i]}.csv', index=False)    


    