# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:23:09 2023

@author: corir
"""

import openai
import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from ast import literal_eval

def generate_list_sust_verbs_adj(text, key):
    '''
    le das el texto del cual queres que encuentre los sustantivos verbos y adjetivos, te los devuelve como le pinte
    a chat GPT pero algo parecido a una lista
    '''
    openai.api_key = key
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + openai.api_key
    }
    
    prompt = "Del siguiente texto me hagas tres listas de python, las listas va a ser una con los sustantivos, otra con los verbos y la última con los adjetivos. El texto es el siguiente: " + text
    
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
    
    
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'me pones estas tres listas como listas de python? O sea, completa sustantivos = [], verbos = [], adjetivos = [] de aca' + response_text}
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


# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"

def conteo_sust_verb_adj_lemm(tema, key):
    
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
            
    #df_textos["texto_crudo"] = textos_crudos
    
    conteo_sust_lemm = []
    conteo_verb_lemm = []
    conteo_adj_lemm = []
    
    #conteo_crudo = []
    for j in tqdm(range(len(Sujetos))):
        while True:
            try:
                sust, verb, adj = generate_list_sust_verbs_adj_directo(df_textos['transcript'][j], key)
                conteo_sust_lemm.append(sust)
                conteo_verb_lemm.append(verb)
                conteo_adj_lemm.append(adj)
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
    
    
    #defino la columna con la lista de sustantivos
    df_textos['sust'] = conteo_sust_lemm
    #la hago una lista, hasta ahora es un string
    df_textos['sust'] = df_textos['sust'].apply(literal_eval)
    df_textos['nro. sust'] = df_textos['sust'].apply(len)
    
    df_textos['verb'] = conteo_verb_lemm
    df_textos['verb'] = df_textos['verb'].apply(literal_eval)
    df_textos['nro. verb'] =df_textos['verb'].apply(len)
    
    df_textos['adj'] = conteo_adj_lemm
    df_textos['adj'] = df_textos['adj'].apply(literal_eval)
    df_textos['nro. adj'] = df_textos['adj'].apply(len)

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos

def conteo_sust_verb_adj(tema, key):
    
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
    for j in tqdm(range(len(Sujetos))):
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
    
    #defino la columna con la lista de sustantivos
    df_textos['sust'] = conteo_sust
    #la hago una lista, hasta ahora es un string
    df_textos['sust'] = df_textos['sust'].apply(literal_eval)
    df_textos['nro. sust'] = df_textos['sust'].apply(len)
    
    df_textos['verb'] = conteo_verb
    df_textos['verb'] = df_textos['verb'].apply(literal_eval)
    df_textos['nro. verb'] =df_textos['verb'].apply(len)
    
    df_textos['adj'] = conteo_adj
    df_textos['adj'] = df_textos['adj'].apply(literal_eval)
    df_textos['nro. adj'] = df_textos['adj'].apply(len)
    

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos




#%%

temas = ["arabia"] #"campeones_del_mundo", "antesdevenir", "presencial", "arabia"

key = "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"

for i in tqdm(range(len(temas))):

    df_contando = conteo_sust_verb_adj(temas[i], key)
    
    df_contando.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{temas[i]}_contando_sust_verb_adj.csv')
#%% a mano (o sea sin usar la funcion)

tema = "arabia"

key = "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"

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
for j in tqdm(range(len(Sujetos))):
    while True:
        try:
            sust, verb, adj = generate_list_sust_verbs_adj(df_textos["texto_crudo"][j], key)
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

#defino la columna con la lista de sustantivos
df_textos['sust'] = conteo_sust
df_textos['verb'] = conteo_verb
df_textos['adj'] = conteo_adj

df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv')
#%%
#la hago una lista, hasta ahora es un string
#df_textos['sust'] = df_textos['sust'].apply(literal_eval)
df_textos['nro. sust'] = df_textos['sust'].apply(len)

#df_textos['verb'] = df_textos['verb'].apply(literal_eval)
df_textos['nro. verb'] =df_textos['verb'].apply(len)

#df_textos['adj'] = df_textos['adj'].apply(literal_eval)
df_textos['nro. adj'] = df_textos['adj'].apply(len)


df_text = df_text.drop(['Unnamed: 0'], axis=1)

df_text = df_text.rename(columns={'Unnamed: 0': 'Sujeto'})

df_text.index += 1

df_text.index.name = 'Sujeto'


df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv')

    
#%%

tema = "campeones_del_mundo"

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
        
#df_textos["texto_crudo"] = textos_crudos

conteo_sust_lemm = []
conteo_verb_lemm = []
conteo_adj_lemm = []

#conteo_crudo = []
#%%

#print(len(conteo_sust_lemm), len(conteo_verb_lemm), len(conteo_adj_lemm))

for j in range(23, len(Sujetos)):
    print(j)
    while True:
        try:
            sust, verb, adj = generate_list_sust_verbs_adj(df_textos['transcript'][j], key)
            print(f"Paso el sujeto {j+1}")
            break
        except KeyError:
            print("Error: KeyError - 'choices'. Esperando 15 minutos y 30 segundos antes de volver a intentarlo")
            time.sleep (15*60 + 30)
            continue
        except Exception:
            print("Error: Excepción no manejada. Volviendo a iterar sobre el mismo texto...")
            time.sleep(1)  # Retraso de 1 segundo
            continue
        
    conteo_sust_lemm.append(sust)
    conteo_verb_lemm.append(verb)
    conteo_adj_lemm.append(adj)
    

    #conteo_crudo.append(generate_list_sust_verbs_adj(df_textos['texto_crudo'][j], key))
    
df_textos['sust'] = conteo_sust_lemm
df_textos['nro. sust'] = df_textos['sust'].apply(len)

df_textos['verb'] = conteo_verb_lemm
df_textos['nro. verb'] = df_textos['verb'].apply(len)

df_textos['adj'] = conteo_adj_lemm
df_textos['nro. adj'] = df_textos['adj'].apply(len)

#df_textos['_curdo'] = sent_crudo

df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

df_textos.index += 1

df_textos.index.name = 'Sujeto'

#%%

df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv')
