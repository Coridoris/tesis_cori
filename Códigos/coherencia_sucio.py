# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:36:09 2023

@author: corir
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import re
import random 

import nltk
nltk.download('punkt')  # Descargar el tokenizador de oraciones (solo se necesita ejecutar una vez)
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


#%%

def coherence(text):

  sentence = sent_tokenize(text)

  embeddings = []
  for i in range(len(sentence)):
      #Compute embedding 
      embeddings.append(model.encode(sentence[i], convert_to_tensor=True))

  cosine_scores1 = []
  #Compute cosine-similarities
  for i in range(len(sentence)-1):
      cosine_scores1.append(util.cos_sim(embeddings[i], embeddings[i+1]).item())
      
  coherence1 = np.mean(cosine_scores1)

  cosine_scores2 = []
  #Compute cosine-similarities
  for i in range(len(sentence)-2):
      cosine_scores2.append(util.cos_sim(embeddings[i], embeddings[i+2]).item())
      
  coherence2 = np.mean(cosine_scores2)

  return coherence1, coherence2

def coherence_evolution(text):

  sentence = sent_tokenize(text)
  
  numb_sentences = len(sentence)

  embeddings = []
  for i in range(len(sentence)):
      #Compute embedding 
      embeddings.append(model.encode(sentence[i], convert_to_tensor=True))

  coherence = []
  for j in range(1, numb_sentences):
      cosine_scores = []
      #Compute cosine-similarities
      for i in range(len(sentence)-j):
          cosine_scores.append(util.cos_sim(embeddings[i], embeddings[i+j]).item())
      
      coherence.append(np.mean(cosine_scores))
      
  return coherence


def shuffle_sentences(text):
    # Suponemos que las oraciones están separadas por puntos seguidos de un espacio o un salto de línea.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Mezclamos el orden de las oraciones.
    random.shuffle(sentences)

    # Unimos las oraciones en un texto mezclado.
    mixed_text = ' '.join(sentences)

    return mixed_text


#%%

# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]


def csv_coherence(tema):

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
    
    
    coherence1 = []
    
    coherence2 = []
    
    for j in range(len(Sujetos)):
        coher1, coher2 = coherence(df_textos['texto_crudo'][j])
        
        coherence1.append(coher1)
        coherence2.append(coher2)
        
    df_textos['coherencia_cada_oracion'] = coherence1
    df_textos['coherencia_cada_dos_oraciones'] = coherence2
    
    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia.csv')
    
    return 'ok'

#%%

# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]


def csv_coherence_evolution(tema):
    '''
    esta función le das le decis el tema del que queres que te busque la coherencia y devuelve una lista que en la 
    primer componente tiene la coherencia promedio del texto, midiendo la coherencia entre oraciones seguidas
    en la segunda tiene la coherencia promedio del texto, midiendo la coherencia entre dos oraciones separadas por una
    y asi
    '''
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
    
    coherence = []
    
    for j in range(len(Sujetos)):
        coher = coherence_evolution(df_textos['texto_crudo'][j])
        
        coherence.append(coher)
        
    df_textos['coherencia_evolucion'] = coherence
    
    df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

    df_textos.index += 1

    df_textos.index.name = 'Sujeto'
    
    df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv')
    
    return 'ok'
#%%
#modelo nulo
'''
para comparar con un modelo nulo la idea es ver si baja la coherencia si la calculamos mezclando oraciones del 
mismo sujeto pero de diferentes textos, por ahora uso también el filler, no se si dejarlo.
'''

#voy a acomodar los textos en un dataframe, pero ya los separo por oraciones
data = dict()
for j in tqdm(range(len(temas))):

    textos_crudos = []
    for i in range(1, len(Sujetos)+1):
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/Sujeto {i}/'
        if os.path.isfile(path + f'sujeto{i}_{temas[j]}.txt') is True: #si existe hacemos esto
            pre_text = open(path+f'sujeto{i}_{temas[j]}.txt', 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
            text = pre_text.read() #esto es un string
            textos_crudos.append(sent_tokenize(text)) #si no lo quiero separados en oraciones tendria que sacar el sent_tokenize
        elif os.path.isfile(path + f'sujeto{i}_{temas[j]}_2021.txt'):
            pre_text = open(path + f'sujeto{i}_{temas[j]}_2021.txt', 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
            text = pre_text.read() #esto es un string
            textos_crudos.append(sent_tokenize(text)) 
        else:
            print(f"El archivo Sujeto{i}_{temas[j]} no existe")
            
    data[f"texto_crudo_{temas[j]}"] = textos_crudos
        
df = pd.DataFrame(data)

def columna_con_mas_oraciones(row):
    '''
    tenes que cambiar las columnas en las que queres ver la máxima cantidad de oraciones
    '''
    #esto busco las oraciones que tiene cada  fila en los diferentes temas
    oraciones_tema1 = len(row['texto_crudo_cfk'])
    oraciones_tema2 = len(row['texto_crudo_campeones_del_mundo'])
    oraciones_tema3 = len(row['texto_crudo_antesdevenir'])
    oraciones_tema4 = len(row['texto_crudo_presencial'])
    oraciones_tema5 = len(row['texto_crudo_arabia'])
    #las acomodo
    oraciones_textos = [oraciones_tema1, oraciones_tema2, oraciones_tema3, oraciones_tema4, oraciones_tema5]
    #me quedo con el número mas grande
    return(max(oraciones_textos))

def columna_con_menos_oraciones(row):
    '''
    tenes que cambiar las columnas en las que queres ver la mínima cantidad de oraciones
    '''
    #esto busco las oraciones que tiene cada  fila en los diferentes temas
    oraciones_tema1 = len(row['texto_crudo_cfk'])
    oraciones_tema2 = len(row['texto_crudo_campeones_del_mundo'])
    oraciones_tema3 = len(row['texto_crudo_antesdevenir'])
    oraciones_tema4 = len(row['texto_crudo_presencial'])
    oraciones_tema5 = len(row['texto_crudo_arabia'])
    #las acomodo
    oraciones_textos = [oraciones_tema1, oraciones_tema2, oraciones_tema3, oraciones_tema4, oraciones_tema5]
    #me quedo con el número mas grande
    return(min(oraciones_textos))

# Aplicar la función a cada fila del DataFrame, nos armamos una columna donde el valor es el número máximo de oracion de los 5 temas
df['num_max_oraciones_del_sujeto'] = df.apply(columna_con_mas_oraciones, axis=1)
df['num_min_oraciones_del_sujeto'] = df.apply(columna_con_menos_oraciones, axis=1)

#ahora necesitamos un código que agarre oraciones aleatorias de cada uno de los temas. La cantidad de oraciones
#al principio iba a ser el máximo de oraciones del sujeto, pero a veces ese número se va a la mierda.
#después pensé en el mínimo, pero a veces no llega ni a 5. Lo que se me ocurrió es tomar 5, una de cada tema, si 
#después se cambia hay que modificar esto.

# Función para elegir una oración al azar de una lista de oraciones
def elegir_oracion_al_azar(oraciones):
    return random.choice(oraciones)

for i in tqdm(range(0,1000)): 
    df[f'texto_random_{i}'] = df.apply(lambda row: elegir_oracion_al_azar(row['texto_crudo_cfk']) + " " + 
                                         elegir_oracion_al_azar(row['texto_crudo_campeones_del_mundo']) + " " + 
                                         elegir_oracion_al_azar(row['texto_crudo_antesdevenir'])
                                         + " " + elegir_oracion_al_azar(row['texto_crudo_presencial'])
                                         + " " + elegir_oracion_al_azar(row['texto_crudo_arabia']), axis=1)
    
    #primero vamos a desordenar las oraciones y después vamos a buscar la coherencia, sino siempre van e estar en este
    #orden los temas
    
    df[f'texto_random_{i}'] = df[f'texto_random_{i}'].apply(shuffle_sentences)
    
    #la calculamos la coherencia
    
    df[f'coherencia_evolucion_de_random_{i}'] = df[f'texto_random_{i}'].apply(coherence_evolution)
    
    df.index += 1
    
    df.index.name = 'Sujeto'

df.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/coherencia_evolucion_random.csv')



#%%

data_prueba = {'Texto1': [['Oración 1.',' Esta es la segunda oración.',' Tercera oración aquí.'],
  ['uno nuevo, otro nuevo.', ' Jaja, god.',' Oa que potra.'],
  ['oye oye princesa, eres perfecta.',' DIUG.']],
 'Texto2': [['Primer texto. Más oraciones aquí.', ' Solo una oración en este texto.', ' a.'],
  ['marito.', ' mato.'],
  ['mañana.']]}

df_prueba = pd.DataFrame(data_prueba)

# Función para elegir una oración al azar de una lista de oraciones
def elegir_oracion_al_azar(oraciones):
    return random.choice(oraciones)

# Crear una nueva columna con oraciones al azar de Texto1 y Texto2 combinadas
df_prueba['Oracion_al_azar'] = df_prueba.apply(lambda row: elegir_oracion_al_azar(row['Texto1']) + " " + elegir_oracion_al_azar(row['Texto2']), axis=1)

print(df_prueba['Oracion_al_azar'])

#df_prueba.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/prueba3_eligiendorandom.csv')



#%%

for i in tqdm(range(len(temas))):
    ok = csv_coherence(temas[i])
    
#%%
for i in tqdm(range(len(temas))):
    ok = csv_coherence_evolution(temas[i])
