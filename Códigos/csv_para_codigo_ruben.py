# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:31:21 2023

@author: corir
"""

'''
Este código es para armar un csv para usar el código de Ruben para clasificar el texto como semántico o espisódico.

Your .csv file includes three columns called 'participantID', 'prompt', and 'text'. These columns contain your participant 
IDs, the names or numbers of the prompts the participants saw, and the responses.

Hay que leer el README antes para ver que no cambie nada.

Como no indica como hay que separar los diferentes textos del participante voy a hacer un csv por tema.
Primero lo hago en español, después hago un csv con la columna text en inglés usando chatGPT
'''
#%% librerias
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import openai
import requests
import time
#%% funciones
def traductor_español_a_ingles(text, key):
    '''
    le das el texto del cual queres que te traduzca y te lo devuelve en inglés. Ver de actualizar la key
    '''
    openai.api_key = key
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + openai.api_key
    }
    
    
    prompt = "Traducime a inglés lo mas textual que se pueda este texto " + text 
    
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
    while True:
        try:
            # Obtener la respuesta del modelo
            response_text = result['choices'][0]['message']['content']
            break
        except KeyError:
            print(result)
            time.sleep(60)
            continue
        except Exception:
            print(result)
            continue

    return response_text

#%% el santo trial

entrevista = 'Segunda'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]


#%% Traduccion

temas = ["campeones_del_mundo"]

key = "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"

for i, tema in tqdm(enumerate(temas)):
    

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Texto/{tema}_crudo_y_lemm_usando_stanza_.csv'
        
    df = pd.read_csv(path)
           
    lenght = len(df)
    datos = {
    #'participantID': np.linspace(1,lenght,lenght),
    'prompt': np.ones(lenght),
    'text': list(df["texto_crudo"])}
    
    df_textos = pd.DataFrame(datos)
    
    #df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})
    df_textos.index += 1
    df_textos.index.name = 'participantID'
    
    #df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Clasificacion_episodico_semantico/textos_limpios_{tema}.csv')    

    #si estoy segura que va a correr de una aplico esto, pero puede que sature la cantidad de pedidos y tenga que ir
    #de nuevo lento
    #df_textos['text'] = df_textos['text'].apply(traductor_español_a_ingles)
    textos_en_ingles = []
#%%
for j in tqdm(range(0,len(df_textos['text']))):
    if type(df_textos['text'][j+1]) == str:
        textos_en_ingles.append(traductor_español_a_ingles(df_textos['text'][j+1], key))
    else:
        textos_en_ingles.append(np.nan)
#%%   

#nuevo_textos_en_ingles = [np.nan] * 30 + textos_en_ingles si ya tenia hecho los primeros 30

df_textos['text'] = textos_en_ingles

df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Clasificacion_episodico_semantico/clean_text_{tema}.csv')

#%%

ttt = "Bueno, de ese día del intento de asesinato a Cristina. Me acuerdo que era de noche. Y me enteré viendo Twitter. Creo que primero me enteré de que la habían intentado, y después me enteré más tarde que habían decretado feriado, al día siguiente. Y me acuerdo que estaba como, no tenía ganas de ir a la facultad y me pusieron a ese feriado y era como un viernes, creo, y entonces fue como un fin de semana largo. Y de cómo enterarme, me acuerdo del intento, me enteré viendo Twitter y después del feriado, me había enterado y después nos mandó mi mamá por WhatsApp y nos enteramos todos. Me acuerdo que nadie en mi familia se había enterado, estábamos todos como haciendo cosas en la facultad y eso me acuerdo. Un detalle más. Me acuerdo de emoción, no sentí ni bronca, ni felicidad, pero obviamente no está bueno que pase eso. No tiene que pasar de ninguno de los dos lados, eso pensaba. Me acuerdo que era de noche, ya bastante de noche así que fue enterarme, ponerme feliz por el feriado, después como incertidumbre por lo que pasó y ya está, me fui a dormir, no hice mucho más. Y detalle del feriado. Me acuerdo que era viernes, así que yo me fui para mi ciudad, que es Mercedes, que voy a un colectivo de siempre, pero como era feriado ese colectivo no estaba para irse. Y me acuerdo, bueno, fui hasta Luján, que no sé por qué el de Mercedes no estaba, pero para Luján sí. Fui de Luján a Mercedes, que fue un viaje re largo, por culpa del feriado que no fue el colectivo que es rápido."

ttt_traducido = traductor_español_a_ingles(ttt, key)

#%%

tema = "cfk"

p = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Clasificacion_episodico_semantico/clean_text_{tema}.csv'
    
df_ = pd.read_csv(p)
#%%
df_.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Clasificacion_episodico_semantico/clean_text_{tema}.csv', index = False)

