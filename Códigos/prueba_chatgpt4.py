# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:07:57 2023

@author: corir
"""

import openai
import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

def generate_list_sust_verbs_adj_GPT4(text, key):
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
        'model': 'gpt-4',
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
#%%
tema = "arabia"

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
#%%
text = textos_crudos[0]

key = "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"

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

