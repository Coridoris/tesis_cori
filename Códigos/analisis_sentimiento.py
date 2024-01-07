# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:53:11 2023

@author: corir
"""

'''
Aca vamos a hacer el análisis de sentimiento
'''

#primero dejo cositas que se necesitan ni bien lo empezas a usar en otra compu

#pip install gensim
#pip install sentileak
#pip install sentiment_analysis_spanish
#pip install pysentimiento

#librerias

import pandas as pd # Data frames
from gensim.utils import tokenize # Tokenizacion, saca simbolos y números (para otras funcionalidades, hacerlo a mano o buscar en nltk)
from gensim.matutils import corpus2csc # Llevar las listas de palabras a una descripción frecuentista
from gensim.corpora import Dictionary # Armado de base de palabras
from gensim.models import TfidfModel # Implementación del modelo tf-idf
import nltk # En este caso, únicamente para stopwords

from sentileak import dataloader
from sentileak import SentiLeak # Lexicon Based Method

from sentiment_analysis_spanish import sentiment_analysis # Supervised Machine Learning Based Aproach


import seaborn as sbn
import matplotlib.pylab as plt
import numpy as np
from wordcloud import WordCloud
nltk.download('stopwords')

from tqdm import tqdm
import os

#%% Método basado del léxico con diccionarios ---> SentiLeak

sentimental = SentiLeak() # Inicializamos el objeto

prueba = sentimental.compute_sentiment('Qué triste verte partir. Qué alegría saber que volverás')


# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

tema = temas[4]


path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
    
    
df_textos = pd.read_csv(path)

sentimiento = []
for i in range(len(df_textos['transcript'])):
    sentimiento.append(sentimental.compute_sentiment(df_textos['transcript'][i])['global_sentiment'])

df_textos['sentileak'] = sentimiento

df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

#df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

df_textos.index += 1

df_textos.index.name = 'Sujeto'

#guardamos la data

#df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{tema}_sentimiento.csv')
#%%

'''
hago una función, chequear que da lo mismo antes de borrar lo de arriba
'''
def sentileak(tema):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
        
    df_textos = pd.read_csv(path)
    
    Sujetos = ['0']*30
    for j in range(30):
        Sujetos[j] = f"Sujeto {j+1}"
    
    sent_palabra_a_palabra = []
    for i in range(len(Sujetos)):
    
        palabra_a_palabra = df_textos['transcript'][i].split()
        
        sent_palabra_total = 0
        for i in range(len(palabra_a_palabra)):
            sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra[i])['global_sentiment']
            sent_palabra_total = sent_palabra_total + sentimiento_palabra
            
        sent_palabra_a_palabra.append(sent_palabra_total)
    
    sentimiento_lemm = []
    for i in range(len(df_textos['transcript'])):
        sentimiento_lemm.append(sentimental.compute_sentiment(df_textos['transcript'][i])['global_sentiment'])

    
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
        
    sentimiento_crudo = []
    for i in range(len(df_textos['texto_crudo'])):
        sentimiento_crudo.append(sentimental.compute_sentiment(df_textos['texto_crudo'][i])['global_sentiment'])
        

    df_textos['sentileak_lemm'] = sentimiento_lemm
    
    df_textos['sentileak_por_palabra'] = sent_palabra_a_palabra
    
    df_textos['sentileak_curdo'] = sentimiento_crudo

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos

#df_prueba = sentileak(tema)

#%%

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

for i in tqdm(range(len(temas))):

    df_sent_sentileak = sentileak(temas[i])
    
    df_sent_sentileak.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_sentileak_sentimiento.csv')



#%% Intentando entender qué hace sentileak

textos_crudos = []
for i in range(1, len(Sujetos)+1):
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/Sujeto {i}/Sujeto{i}_{tema}.txt'
    pre_text = open(path, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
    text = pre_text.read() #esto es un string
    textos_crudos.append(text)
    
df_textos["texto_crudo"] = textos_crudos

sent_lemm = sentimental.compute_sentiment(df_textos['transcript'][1])['global_sentiment']

sent_crudo = sentimental.compute_sentiment(df_textos['texto_crudo'][1])['global_sentiment']

palabra_a_palabra = df_textos['texto_crudo'][1].split()

sent_palabra_total = 0
sent_palabra = []
palabras_sent = []
for i in range(len(palabra_a_palabra)):
    sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra[i])['global_sentiment']
    sent_palabra_total = sent_palabra_total + sentimiento_palabra
    if sentimiento_palabra != 0:
        sent_palabra.append(sentimiento_palabra)
        palabras_sent.append(palabra_a_palabra[i])
    
print('sentimiento global lemmatizando:', sent_lemm,'sentimiento global texto crudo:', sent_crudo, 'suma sentimiento de cada palabra', sent_palabra_total)

data = {'palabra': palabras_sent,
        'sentileak': sent_palabra}

# Crear el DataFrame
df = pd.DataFrame(data)

df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{tema}_palabra_a_palabra_crudo_sentimiento.csv')

#%% Tasa positividad, negatividad, intensidad y neutralidad (de tesis de Carillo) 
#por ahora solo hecha con el diccionario de sentileak


def sentileak_tasas(tema):
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
        
    df_textos = pd.read_csv(path)
    
    Sujetos = ['0']*30
    for j in range(30):
        Sujetos[j] = f"Sujeto {j+1}"
    
    tasa_positividad = []
    tasa_negatividad = []
    intensidad = []
    neutralidad = []
    
    tasa_positividad_unicas = []
    tasa_negatividad_unicas = []
    intensidad_unicas = []
    neutralidad_unicas = []
    
    tasa_muy_positividad = []
    tasa_muy_negatividad = []
    muy_intensidad = []
    muy_neutralidad = []
    
    tasa_muy_positividad_unicas = []
    tasa_muy_negatividad_unicas = []
    muy_intensidad_unicas = []
    muy_neutralidad_unicas = []
    
    
    for i in range(len(Sujetos)):
    
        palabra_a_palabra = df_textos['transcript'][i].split()
        
        positivas = []
        negativas = []
        
        positivas_unicas = []
        negativas_unicas = []

        muy_positivas = []
        muy_negativas = []
        
        muy_positivas_unicas = []
        muy_negativas_unicas = []
        
        palabras_ya_sumadas = []
        
        for j in range(len(palabra_a_palabra)):
            sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra[j])['global_sentiment']
            if palabra_a_palabra[j] not in palabras_ya_sumadas:
                if sentimiento_palabra > 0:
                    positivas.append(1)
                    positivas_unicas.append(1)
                if sentimiento_palabra > 2:
                    muy_positivas.append(1)
                    muy_positivas_unicas.append(1)
                if sentimiento_palabra < 0:
                    negativas.append(1)
                    negativas_unicas.append(1)
                if sentimiento_palabra < -2:
                    muy_negativas.append(1)
                    muy_negativas_unicas.append(1)
            else:
                if sentimiento_palabra > 0:
                    positivas.append(1)
                if sentimiento_palabra > 2:
                    muy_positivas.append(1)
                if sentimiento_palabra < 0:
                    negativas.append(1)
                if sentimiento_palabra < -2:
                    muy_negativas.append(1)
                    
            palabras_ya_sumadas.append(palabra_a_palabra[j])
                
        tasa_positividad.append(sum(positivas)/len(palabra_a_palabra))
        tasa_negatividad.append(sum(negativas)/len(palabra_a_palabra))
        
        intensidad.append((sum(positivas)+sum(negativas))/len(palabra_a_palabra))
        neutralidad.append(1-intensidad[-1])
        
        palabra_a_palabra_unicas = list(set(palabra_a_palabra))
        
        tasa_positividad_unicas.append(sum(positivas_unicas)/len(palabra_a_palabra_unicas))
        tasa_negatividad_unicas.append(sum(negativas_unicas)/len(palabra_a_palabra_unicas))
        
        intensidad_unicas.append((sum(positivas_unicas)+sum(negativas_unicas))/len(palabra_a_palabra_unicas))
        neutralidad_unicas.append(1-intensidad_unicas[-1])
        
        tasa_muy_positividad.append(sum(muy_positivas)/len(palabra_a_palabra))
        tasa_muy_negatividad.append(sum(muy_negativas)/len(palabra_a_palabra))
        
        muy_intensidad.append((sum(muy_positivas)+sum(muy_negativas))/len(palabra_a_palabra))
        muy_neutralidad.append(1-muy_intensidad[-1])
        
        tasa_muy_positividad_unicas.append(sum(muy_positivas_unicas)/len(palabra_a_palabra_unicas))
        tasa_muy_negatividad_unicas.append(sum(muy_negativas_unicas)/len(palabra_a_palabra_unicas))
        
        muy_intensidad_unicas.append((sum(muy_positivas_unicas)+sum(muy_negativas_unicas))/len(palabra_a_palabra_unicas))
        muy_neutralidad_unicas.append(1-muy_intensidad_unicas[-1])
        
        
    # tasa_positividad_unicas_ = []
    # tasa_negatividad_unicas_ = []
    # intensidad_unicas_ = []
    # neutralidad_unicas_ = []

    # for i in range(1, len(Sujetos)):
    
    #     palabra_a_palabra = df_textos['transcript'][i].split()
        
    #     palabra_a_palabra_unicas = list(set(palabra_a_palabra))
        
    #     positivas_unicas_ = []
    #     negativas_unicas_ = []

        
    #     for j in range(len(palabra_a_palabra_unicas)):
            
    #         sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra_unicas[j])['global_sentiment']
            
    #         if sentimiento_palabra > 0:
    #             positivas_unicas_.append(1)
    #         if sentimiento_palabra < 0:
    #             negativas_unicas_.append(1)

        
    #     tasa_positividad_unicas_.append(sum(positivas_unicas_)/len(palabra_a_palabra_unicas))
    #     tasa_negatividad_unicas_.append(sum(negativas_unicas_)/len(palabra_a_palabra_unicas))
        
    #     intensidad_unicas_.append((sum(positivas_unicas_)+sum(negativas_unicas_))/len(palabra_a_palabra_unicas))
    #     neutralidad_unicas_.append(1-intensidad_unicas_[-1])            
 
    
    df_textos['tasa_positividad'] = tasa_positividad
    df_textos['tasa_negatividad'] = tasa_negatividad
    
    df_textos['intensidad'] = intensidad
    df_textos['neutralidad'] = neutralidad
    
    df_textos['tasa_positividad_unicas'] = tasa_positividad_unicas
    df_textos['tasa_negatividad_unicas'] = tasa_negatividad_unicas
    
    df_textos['intensidad_unicas'] = intensidad_unicas
    df_textos['neutralidad_unicas'] = neutralidad_unicas


    df_textos['tasa_muy_positividad'] = tasa_muy_positividad
    df_textos['tasa_muy_negatividad'] = tasa_muy_negatividad
    
    df_textos['muy_intensidad'] = muy_intensidad
    df_textos['muy_neutralidad'] = muy_neutralidad
    
    df_textos['tasa_muy_positividad_unicas'] = tasa_muy_positividad_unicas
    df_textos['tasa_muy_negatividad_unicas'] = tasa_muy_negatividad_unicas
    
    df_textos['muy_intensidad_unicas'] = muy_intensidad_unicas
    df_textos['muy_neutralidad_unicas'] = muy_neutralidad_unicas    

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos
#%%

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

tema = temas[1]

for i in tqdm(range(len(temas))):

    df_sent_tasas = sentileak_tasas(temas[i])
    
    df_sent_tasas.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_tasas_sentimiento.csv')



#%%
sentiment_words = pd.DataFrame({i : item for i, item in enumerate(dataloader.load_dict('es', 'sentiment_words.csv').items())}).T.rename({0 : 'palabra/lema',
                                                                                                                                         1 : 'valoración'},
                                                                                                                                         axis = 1)
#print(sentiment_words.sort_values(by = 'valoración', ascending = True)[:20])

palabras_negativas = sentiment_words.loc[sentiment_words['valoración'] < 0]

palabras_positivas = sentiment_words.loc[sentiment_words['valoración'] > 0]

print(len(sentiment_words), len(palabras_negativas), len(palabras_positivas))

palabras_muy_negativas = sentiment_words.loc[sentiment_words['valoración'] < -2] #el 48%, va hasta -4, si solo me quedo con esas hay 9% (149 palabras)

palabras_muy_positivas = sentiment_words.loc[sentiment_words['valoración'] > 2] #el 26%, va hasta 4, si solo me quedo con esas hay 2% (22 palabras)

print(len(sentiment_words), len(palabras_muy_negativas), len(palabras_muy_positivas))

#%% Ahora vamos a intentar entender un método de ML

sentimental = sentiment_analysis.SentimentAnalysisSpanish() # Inicializamos el objeto

#el método devuelve la probabilidad de que un determinado texto tenga una connotación positiva.

#%%
sentimental.sentiment('Qué triste verte partir. Pero qué alegría saber que volverás') # Aplicamos el análisis

# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

tema = temas[2]

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza.csv'
    
df_textos = pd.read_csv(path)

textos_crudos = []
for i in range(1, len(Sujetos)+1):
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/Sujeto {i}/Sujeto{i}_{tema}.txt'
    path2 = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/Sujeto {i}/Sujeto{i}_{tema}_2021.txt'
    if os.path.isfile(path) is True: #si existe hacemos esto
        pre_text = open(path, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
        text = pre_text.read() #esto es un string
        textos_crudos.append(text)
    elif os.path.isfile(path2):
        pre_text = open(path2, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
        text = pre_text.read() #esto es un string
        textos_crudos.append(text) 
    else:
        print(f"El archivo Sujeto{i}_{tema} no existe")
    
df_textos["texto_crudo"] = textos_crudos

print(df_textos['texto_crudo'][1])

print(df_textos['transcript'][1])

sent_lemm = sentimental.sentiment(df_textos['transcript'][1])

sent_crudo = sentimental.sentiment(df_textos['texto_crudo'][1])

    
print('sentimiento global lemmatizando:', sent_lemm,'sentimiento global texto crudo:', sent_crudo)

#%% por lo que veo funciona bastante mal, dice que campeones del mundo es lo menos positivo... Igual, la gente en su relato cuenta lo mal q la paso durante el partido...

def sentiment_ML(tema):
    
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
    
    sent_lemm = []
    
    sent_crudo = []
    for j in range(len(Sujetos)):
    
        sent_lemm.append(sentimental.sentiment(df_textos['transcript'][j]))

        sent_crudo.append(sentimental.sentiment(df_textos['texto_crudo'][j]))
    
        

    df_textos['sentML_lemm'] = sent_lemm
    
    df_textos['senTml_curdo'] = sent_crudo

    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    return df_textos

#%%

temas = ["campeones_del_mundo", "antesdevenir", "presencial", "cfk", "arabia"] 

for i in tqdm(range(len(temas))):

    df_sent_ML = sentiment_ML(temas[i])
    
    df_sent_ML.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_ML_sentimiento.csv')

#%%

for i in range(len(temas)):

    path_ML = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_ML_sentimiento.csv'
    
    path_tasas = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_tasas_sentimiento.csv'
    
    path_sentileak = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/{temas[i]}_sentileak_sentimiento.csv'
    
    df_ML = pd.read_csv(path_ML)

    df_tasas = pd.read_csv(path_tasas)
    
    df_sentileak = pd.read_csv(path_sentileak)
    
    df_tasas.insert(2, 'sentileak_curdo', df_sentileak['sentileak_curdo'], True)
    
    df_tasas.insert(2, 'sentileak_lemm', df_sentileak['sentileak_lemm'], True)
    
    df_tasas.insert(2, 'sentileak_por_palabra', df_sentileak['sentileak_por_palabra'], True)
    
    df_tasas.insert(2, 'sentML_lemm', df_ML['sentML_lemm'], True)
    
    df_tasas.insert(2, 'senTml_curdo', df_ML['senTml_curdo'], True)
    
    df_tasas.insert(2, 'texto_crudo', df_sentileak['texto_crudo'], True)
    
    
    df_tasas.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/sentimiento_todos_metodos_{temas[i]}.csv')

    