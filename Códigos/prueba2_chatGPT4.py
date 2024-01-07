# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:34:52 2023

@author: Usuario
"""
import openai
import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
tema = "arabia" 

key = "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"

def generate_list_sust_verbs_adj_directo(text, key):
    '''
    le das el texto del cual queres que encuentre los sustantivos verbos y adjetivos, te los devuelve como listas
    '''
    openai.api_key = key
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + openai.api_key
    }
    
    
    prompt = "Del siguiente texto me hagas tres listas de python, las listas va a ser una con los sustantivos, otra con los verbos y la última con los adjetivos. El texto es el siguiente: " + text + ". Ponelo como tres listas como listas de python. O sea, completa sustantivos = [], verbos = [], adjetivos = []"
    
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    }

    # Realizar la solicitud a la API
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Obtener la respuesta del modelo
    response_text = result['choices'][0]['message']['content']
    
    inicio_sustantivos = response_text.find("ustantivos = [")
    inicio_verbos = response_text.find("erbos = [")
    inicio_adjetivos = response_text.find("djetivos = [")
    fin_sustantivos = response_text.find("]", inicio_sustantivos) + 1
    fin_verbos = response_text.find("]", inicio_verbos) + 1
    fin_adjetivos = response_text.find("]", inicio_adjetivos) + 1

    sustantivos = eval(response_text[inicio_sustantivos+len('ustantivos = '):fin_sustantivos])
    verbos = eval(response_text[inicio_verbos+len('erbos = '):fin_verbos])
    adjetivos = eval(response_text[inicio_adjetivos+len('djetivos = '):fin_adjetivos])

    return sustantivos, verbos, adjetivos


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

conteo_sust = []
conteo_verb = []
conteo_adj = []

#conteo_crudo = []
for j in tqdm(range(9,10)):
    while True:
        try:
            sust, verb, adj = generate_list_sust_verbs_adj_directo(df_textos["texto_crudo"][j], key)
            conteo_sust.append(sust)
            conteo_verb.append(verb)
            conteo_adj.append(adj)
            print(f"Paso el sujeto {j+1}")
            break
        except KeyError:
            print("Error: KeyError - 'choices'. Esperando 7 minutos y 12 segundos antes de volver a intentarlo")
            time.sleep (7*60 + 12)
            continue
        except Exception:
            print("Error: Excepción no manejada. Volviendo a iterar sobre el mismo texto...")
            time.sleep(1)  # Retraso de 1 segundo
            continue
    

    #conteo_crudo.append(generate_list_sust_verbs_adj(df_textos['texto_crudo'][j], key))
#%%
df_textos['lista_sust'] = conteo_sust
df_textos['nro. sust'] = df_textos['listas_sust'].apply(len)

df_textos['verb'] = conteo_verb
df_textos['nro. verb'] =df_textos['verb'].apply(len)

df_textos['adj'] = conteo_adj
df_textos['nro. adj'] = df_textos['adj'].apply(len)

#df_textos['_curdo'] = sent_crudo

df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

df_textos.index += 1

df_textos.index.name = 'Sujeto'

#%%
path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj_lemm.csv'
    
df_textos = pd.read_csv(path)

#paso a listas de python los strings
df_textos['listas_sust_verb_adj'] = df_textos['listas_sust_verb_adj'].apply(literal_eval)
df_textos['verb'] = df_textos['verb'].apply(literal_eval)
df_textos['adj'] = df_textos['adj'].apply(literal_eval)

df_textos['nro. sust'] = df_textos['listas_sust_verb_adj'].apply(len)

df_textos['nro. verb'] =df_textos['verb'].apply(len)

df_textos['nro. adj'] = df_textos['adj'].apply(len)

df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj_lemm.csv')


#%%

openai.api_key = key

text = df_textos["texto_crudo"][9]

url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + openai.api_key
}


prompt = "Del siguiente texto me hagas tres listas de python, las listas va a ser una con los sustantivos, otra con los verbos y la última con los adjetivos. Hacelo con TODO EL TEXTO, no dejes puntos suspensivos. El texto es el siguiente: " + text + ". Ponelo como tres listas como listas de python. O sea, completa sustantivos = [], verbos = [], adjetivos = []"

data = {
    'model': 'gpt-3.5-turbo',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
}

# Realizar la solicitud a la API
response = requests.post(url, headers=headers, json=data)
result = response.json()

# Obtener la respuesta del modelo
response_text = result['choices'][0]['message']['content']