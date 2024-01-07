# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 23:57:02 2023

@author: corir
"""

import pandas as pd
import os

Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
tema = "arabia" #"campeones_del_mundo", "antesdevenir", "presencial", "arabia"

#busco nuevas maneras de contar sust verb adj
#%%
import es_core_news_sm
nlp = es_core_news_sm.load()

#%%
doc = nlp("El copal se usa principalmente para sahumar en distintas ocasiones como lo son las fiestas religiosas.")
print([(w.text, w.pos_) for w in doc])
#%%
path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
    
df_textos = pd.read_csv(path)

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
        
df_textos["texto_crudo"] = textos_crudos

#%%

doc = nlp(df_textos["texto_crudo"][2])
print([(w.text, w.pos_) for w in doc])

#%%
df_textos["clasificacion"] = df_textos["texto_crudo"].apply(nlp)


#%%

import pattern.es as lemEsp
print(lemEsp.parse('estan', lemmata=True))