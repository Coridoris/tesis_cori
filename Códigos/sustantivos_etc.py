# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 20:04:09 2023

@author: Usuario
"""
import nltk as nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

#%%
def contar_partes_discurso(texto):
    # Tokenizar el texto en palabras
    palabras = word_tokenize(texto)
    
    # Etiquetar las partes del discurso de las palabras
    etiquetas = pos_tag(palabras)
    
    # Crear un diccionario para contar las partes del discurso
    conteo = {
        'Sustantivos': 0,
        'Verbos': 0,
        'Adjetivos': 0,
        # Añade más partes del discurso según tus necesidades
    }
    
    # Contar las partes del discurso
    for palabra, etiqueta in etiquetas:
        if etiqueta.startswith('N'):  # Sustantivo
            conteo['Sustantivos'] += 1
        elif etiqueta.startswith('V'):  # Verbo
            conteo['Verbos'] += 1
        elif etiqueta.startswith('J'):  # Adjetivo
            conteo['Adjetivos'] += 1
        # Añade más condiciones para contar otras partes del discurso
        
    return conteo, etiquetas
#%%

tema = "cfk"
nro_sujeto = 0

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/sentimiento_contando_{tema}.csv'

df_prueba = pd.read_csv(path)

resultado_lemm = contar_partes_discurso(df_prueba['transcript'][nro_sujeto])
resultado_crudo = contar_partes_discurso(df_prueba['texto_crudo'][nro_sujeto])

print(df_prueba['transcript'][nro_sujeto])
print(resultado_lemm)


print(df_prueba['texto_crudo'][nro_sujeto])
print(resultado_crudo)

#%% viendo como funciona
temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

for i in tqdm(range(len(temas))):
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/sentimiento_contando_{temas[i]}.csv'

    df = pd.read_csv(path)
    
    nouns = []
    verbs = []
    adjs = []
    
    nouns_1 = []
    verbs_1 = []
    adjs_1 = []
    for i in range(len(df['transcript'])):
    
        noun, verb, adj = list(contar_partes_discurso(df['transcript'][i]))
        noun_1, verb_1, adj_1 = list(contar_partes_discurso(df['texto_crudo'][i]))
        
        nouns.append(noun)
        verbs.append(verb)
        adjs.append(adj)
        
        nouns_1.append(noun_1)
        verbs_1.append(verb_1)
        adjs_1.append(adj_1)
        
    df["noun"] = nouns
    df["verb"] = verbs
    df["adj"] = adjs
    
    df["noun crudo"] = nouns_1
    df["verb crudo"] = verbs_1
    df["adj crudo"] = adjs_1
    
    df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/clasificacion_palabras_sentimiento_frecuencia_{temas[i]}.csv')
