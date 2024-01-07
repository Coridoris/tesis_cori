# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:50:33 2023

@author: corir
"""

import openai
import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

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
            {'role': 'user', 'content': 'me pones estas tres listas como listas de python? Solo responde eso!!' + response_text}
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
            
    #df_textos["texto_crudo"] = textos_crudos
    
    conteo_sust_lemm = []
    conteo_verb_lemm = []
    conteo_adj_lemm = []
    
    #conteo_crudo = []
    for j in range(len(Sujetos)):
        
        sust, verb, adj = generate_list_sust_verbs_adj(df_textos['transcript'][j], key)
    
        conteo_sust_lemm.append(sust)
        conteo_verb_lemm.append(verb)
        conteo_adj_lemm.append(adj)

        #conteo_crudo.append(generate_list_sust_verbs_adj(df_textos['texto_crudo'][j], key))
    
    df_textos['listas_sust_verb_adj'] = conteo_sust_lemm
    df_textos['nro. sust'] = len(conteo_sust_lemm)
    
    df_textos['verb'] = conteo_verb_lemm
    df_textos['nro. verb'] = len(conteo_verb_lemm)
    
    df_textos['adj'] = conteo_adj_lemm
    df_textos['nro. adj'] = len(conteo_adj_lemm)
    
    #df_textos['_curdo'] = sent_crudo

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos





#%%

temas = ["cfk"] #"campeones_del_mundo", "antesdevenir", "presencial", "arabia"

key = "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"

for i in tqdm(range(len(temas))):

    df_contando = conteo_sust_verb_adj(temas[i], key)
    
    df_contando.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{temas[i]}_contando_sust_verb_adj.csv')

#%%

tema = "presencial"

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

for j in range(29, len(Sujetos)):
    print(j)
    sust, verb, adj = generate_list_sust_verbs_adj(df_textos['transcript'][j], key)
    conteo_sust_lemm.append(sust)
    conteo_verb_lemm.append(verb)
    conteo_adj_lemm.append(adj)

    #conteo_crudo.append(generate_list_sust_verbs_adj(df_textos['texto_crudo'][j], key))
#%%    
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


#%% Dos manera que vi para comunicarse con la API, la manera mas "facil" y la q peor modelo tiene creo

key = "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" #noentiendojeje
#openai.api_key = os.getenv(key) #esto es cuando estas de un enviroment creo

openai.api_key = key

prompt = "Del siguiente texto me hagas tres listas de python, las listas va a ser una con los sustantivos, otra con los verbos y la última con los adjetivos. El texto es el siguiente: buen partido acuerdo ser mañana creer jugado 7 día sueño así levanté decir primero partido ir pasar interesante después ver celu salir calle acuerdo segundo gol anulado anulado mano dedo mano según var poter varios idioma incluido árabe después terminar partido eitan tomar yo auto venir facultad acuerdo cursar acuerdo ánimo facultad bastante bajo mundo emocionalmente destruido derrota"
response1 = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt,
  max_tokens=500
)

print(response1.choices[0].text.strip())

#%% aca podes elegir el modelo

#response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)

url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + openai.api_key
}

prompt = "Del siguiente texto me hagas tres listas de python, las listas va a ser una con los sustantivos, otra con los verbos y la última con los adjetivos. El texto es el siguiente: buen partido acuerdo ser mañana creer jugado 7 día sueño así levanté decir primero partido ir pasar interesante después ver celu salir calle acuerdo segundo gol anulado anulado mano dedo mano según var poter varios idioma incluido árabe después terminar partido eitan tomar yo auto venir facultad acuerdo cursar acuerdo ánimo facultad bastante bajo mundo emocionalmente destruido derrota"

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
print(response_text)


#%%
import numpy as np
import pandas as pd

path = "d:/Facultad/Tesis/"
base = pd.read_csv(path+'Corpus_medios_nac.csv', nrows = 10)
notas = base['nota']
notas = list(notas)

def generate_chat_response(prompt, messages):
    openai.api_key = ''  # Replace with your actual API key

    # Format the messages as per OpenAI's chat format
    formatted_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    for idx, msg in enumerate(messages):
        formatted_messages.append({'role': 'user', 'content': msg})
        if idx < len(messages) - 1:
            formatted_messages.append({'role': 'assistant', 'content': ''})

    # Generate a response from the ChatGPT model
    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the ChatGPT engine
        prompt=formatted_messages,
        temperature=0.7,
        max_tokens=50,
        n=1,
        stop=None,
    )

    # Extract the reply from the response and return it
    reply = response.choices[0].text.strip()
    return reply


def generate_2(message):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': messege}
        ]
    }

    # Realizar la solicitud a la API
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Obtener la respuesta del modelo
    response_text = result['choices'][0]['message']['content']
    return response_text

prompt = "Chat with the assistant:"
text = notas[0]
messege = f"Extract the qouted phreasis in this text and give me the answer in a list of python with the qouted phrases: {text}"
response = generate_2(messege)
print(response)

