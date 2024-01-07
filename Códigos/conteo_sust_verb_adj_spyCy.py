# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:25:35 2023

@author: corir
"""

import numpy as np
import pandas as pd
import os
import es_core_news_sm
from tqdm import tqdm

nlp = es_core_news_sm.load()
#%%
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
def clasificacion_texto(text):
    doc = nlp(text)
    return [(w.text, w.pos_) for w in doc]

def eliminar_frase(texto, frase_a_eliminar):
    # Reemplaza la frase a eliminar con una cadena vacía
    texto_sin_frase = texto.replace(frase_a_eliminar, "")
    return texto_sin_frase

def contar_palabras(texto):
    # Divide el texto en palabras utilizando el espacio en blanco como separador
    palabras = texto.split()
    # Cuenta el número de palabras y devuelve el resultado
    return len(palabras)

def clasificacion_tema(tema, frase_a_eliminar = None, todas = False):
    '''
    si queres eliminar frases ponerlas en una lista 
    '''
    #si quiero ver lemmatizando
    #path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
        
    #df_textos = pd.read_csv(path)
    
    #y despues de hacer textos crudos:
    #df_textos["texto_crudo"] = textos_crudos
    #la función clasificacion texto se la tengo que aplicar a la columna transcript, o sea
    #df_textos["clasificacion_total_lemm"] = df_textos["transcript"].apply(clasificacion_texto)

    #y al final de todo lo que le hago al pandas (columna sust, verb, adj, conteo de los mismos) se pone esto antes del return
    
    #df_textos = df_textos.drop(['Unnamed: 0'], axis=1)
    #df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})
    #df_textos.index += 1
    #df_textos.index.name = 'Sujeto'

    textos_crudos = []
    for i in range(1, len(Sujetos)+1):
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/Sujeto {i}/'
        if os.path.isfile(path + f'sujeto{i}_{tema}.txt') is True: #si existe hacemos esto
            pre_text = open(path+f'sujeto{i}_{tema}.txt', 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
            text = pre_text.read() #esto es un string
            textos_crudos.append(text)
        elif os.path.isfile(path + f'sujeto{i}_{tema}_2021.txt'):
            pre_text = open(path + f'sujeto{i}_{tema}_2021.txt', 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
            text = pre_text.read() #esto es un string
            textos_crudos.append(text) 
        else:
            print(f"El archivo Sujeto{i}_{tema} no existe")
            
    datos = {
    'Sujetos': np.linspace(1,30,30),
    'texto': textos_crudos}
    
    
    df_textos = pd.DataFrame(datos)
    
    if frase_a_eliminar != None:
        for i in range(len(frase_a_eliminar)):
            df_textos["texto"] = df_textos["texto"].apply(eliminar_frase, args=(frase_a_eliminar[i],))
            
    df_textos["num_total_palabras"] = df_textos["texto"].apply(contar_palabras)
    
    df_textos["clasificacion_total"] = df_textos["texto"].apply(clasificacion_texto)
    
    df_textos['nouns'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'NOUN'])
    
    #si quiero mas de una etiqueta, por ej si quiero sumar PROPN a sustantivos correria algo asi:
    #df_textos['nouns'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta in ['NOUN', 'PRON']])

    df_textos['verbs'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'VERB'])
    
    df_textos['adjs'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'ADJ'])
    
    df_textos['advs'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'ADV'])
    
    df_textos['numeral'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'NUM'])
    
    df_textos['propn'] = df_textos["clasificacion_total"].apply(lambda lista: [palabra for palabra, etiqueta in lista if etiqueta == 'PROPN'])
    
    
    df_textos['num noun'] = df_textos['nouns'].apply(len)
    df_textos['num verb'] = df_textos['verbs'].apply(len)
    df_textos['num adj'] = df_textos['adjs'].apply(len)
    
    df_textos['num advs'] = df_textos['advs'].apply(len)
    df_textos['num numeral'] = df_textos['numeral'].apply(len)
    df_textos['num propn'] = df_textos['propn'].apply(len)
    
    #unicos
    df_textos['num unique noun'] = df_textos['nouns'].apply(lambda lista: len(set(lista)))
    df_textos['num unique verb'] = df_textos['verbs'].apply(lambda lista: len(set(lista)))
    df_textos['num unique adj'] = df_textos['adjs'].apply(lambda lista: len(set(lista)))
    
    df_textos['num unique advs'] = df_textos['advs'].apply(lambda lista: len(set(lista)))
    df_textos['num unique numeral'] = df_textos['numeral'].apply(lambda lista: len(set(lista)))
    df_textos['num unique propn'] = df_textos['propn'].apply(lambda lista: len(set(lista)))
    
    df_textos['num noun norm'] = df_textos['num noun']/df_textos["num_total_palabras"]
    df_textos['num verb norm'] = df_textos['num verb']/df_textos["num_total_palabras"]
    df_textos['num adj norm'] = df_textos['num adj']/df_textos["num_total_palabras"]
    
    df_textos['num advs norm'] = df_textos['num advs']/df_textos["num_total_palabras"]
    df_textos['num numeral norm'] = df_textos['num numeral']/df_textos["num_total_palabras"]
    df_textos['num propn norm'] = df_textos['num propn']/df_textos["num_total_palabras"]
    
    #si queremos que nos de todo el txt hay que poner cualquier otra cosa en la variable todas, 
    #sino nos va a dar un csv solo con lo importante
    if todas == False:
        
        # Selecciona las columnas que queremos
        columnas_seleccionadas = ['Sujetos', 'num noun norm', 'num verb norm', 'num adj norm', 'num advs norm', 'num numeral norm', 'num propn norm']
        
        #reescribe el dataframe
        df_textos = df_textos[columnas_seleccionadas]
        
        df_textos.insert(1, 'Condición', tema)
        
    return df_textos

#%%
temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

for i in tqdm(range(len(temas))):

    df_contando = clasificacion_tema(temas[i])
    
    df_contando.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{temas[i]}.csv')    

#%% eliminando el "me acuerdo"

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

for i in tqdm(range(len(temas))):

    df_contando = clasificacion_tema(temas[i], ["Me acuerdo", "Recuerdo", "me acuerdo", "recuerdo"])
    
    df_contando.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{temas[i]}_sinMeacuerdo.csv', index=False)    

#%% veo si hay correlación aca

correlation_matrix_sust = []
correlation_matrix_verb = []
correlation_matrix_adj = []
correlation_matrix_advs = []
correlation_matrix_numeral = []
correlation_matrix_propn = []

import matplotlib.pyplot as plt 
path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/9-9'

for i in tqdm(range(len(temas))):
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{temas[i]}_sinMeacuerdo.csv'
    
    df = pd.read_csv(path)
    
    sust = df['num noun']
    sust_unique =df['num unique noun']
    
    verb = df['num verb']
    verb_unique =df['num unique verb']
    
    adj = df['num adj']
    adj_unique =df['num unique adj']
    
    advs = df['num advs']
    advs_unique =df['num unique advs']

    numeral = df['num numeral']
    numeral_unique =df['num unique numeral']
    
    propn = df['num propn']
    propn_unique =df['num unique propn']
    
    plt.figure(i), plt.clf()
    plt.title(f'{temas[i]}')
    plt.plot(sust, sust_unique, 'o', label = "sust")
    plt.plot(verb, verb_unique, 'o', label = "verb")
    plt.plot(adj, adj_unique, 'o', label = "adj")
    plt.plot(advs, advs_unique, 'o', label = "advs")
    plt.plot(numeral, numeral_unique, 'o', label = "numeral")
    plt.plot(propn, propn_unique, 'o', "propn")
    plt.xlabel("total")
    plt.ylabel("unicos")
    plt.legend()
    plt.savefig(path_imagenes + f'/sust_verb_adj_unicas_vs_totales_{temas[i]}.png')
    plt.show()
    
    
    
    #Calcular los coeficiente de correlación
    correlation_matrix_sust.append(np.corrcoef(sust, sust_unique)[0, 1])
    correlation_matrix_verb.append(np.corrcoef(verb, verb_unique)[0, 1])
    correlation_matrix_adj.append(np.corrcoef(adj, adj_unique)[0, 1])
    correlation_matrix_advs.append(np.corrcoef(advs, advs_unique)[0, 1])
    correlation_matrix_numeral.append(np.corrcoef(numeral, numeral_unique)[0, 1])
    correlation_matrix_propn.append(np.corrcoef(propn, propn_unique)[0, 1])

data = {"tema": ["arab", "camp", "filler", "pres", "cfk"],
        "Corr sust": correlation_matrix_sust, "Corr verb": correlation_matrix_verb, "Corr adj": correlation_matrix_adj,
        "Corr advs": correlation_matrix_advs, "Corr numeral": correlation_matrix_numeral, "Corr propn": correlation_matrix_propn}

df_corr = pd.DataFrame(data)
