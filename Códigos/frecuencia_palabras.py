# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:50:13 2023

@author: corir
"""

'''
en el siguiente notebook vamos buscar la frecuencia de las palabras de cada sujeto
En la primera celda se bajas las liberias
En la segunda celda tenes funciones para limpiar el texto y lemmatizar
En la tercera celda bajamos el texto
En la cuarte celda acomodamos la data en un dataframe
En la quinta celda usamos count vectoraizer para contar la cantidad de palabras
En la sexta celda buscamos la frecuencia
En la septima celda vemos las mas usadas por sujeto
'''

#LIBRERIAS

import numpy as np
import pandas as pd
import re       # libreria de expresiones regulares
import string   # libreria de cadena de caracteres
import time
import nltk
#nltk.download('stopwords') # hay que descargar este modulo en particular
#ademas para stemmizar
from nltk.stem import SnowballStemmer
#nltk.download('punkt')
import os

#ademas para lemm con spacy
#!pip install spacy
#!python -m spacy download es_core_news_sm
import spacy

#para stanza
import stanza
#stanza.download('es') # descarga el modelo en español
nlp = stanza.Pipeline('es')
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from tqdm import tqdm


#%% FUNCIONES

def pre_analisis_texto(file_name): #le das la ubicacion del texto
    pre_text = open(file_name, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
    text = pre_text.read() #esto es un string
    #empezamos a arreglarlo
    text = text.lower()  # pasa las mayusculas del texto a minusculas                                                             
    # reemplaza singnos de puntuacion por espacio en blanco. %s es \S+ que quiere decir cualquier caracter que no sea un espacio en blanco
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    
    text = re.sub('\n', '', text)  #saca los enter
    
    text = re.sub('¿', '', text) #saca los ¿
    #re.sub sustituye una secuencia de caracteres por otra:
    #re.sub('lo que queres reemplazar', 'por lo que lo reemplazas', texto donde esta)

    stop_words = nltk.corpus.stopwords.words('spanish')          
    #dejo por si necesito agregar o sacar una stopword
    #stopwords.append('ejemplo')
    #stopwords.remove('ejemplo')

    stopwords_dict = {word: 1 for word in stop_words}

    text_sin_sw = " ".join([word for word in text.split() if word not in stopwords_dict])
    
    return(text_sin_sw)

def stemmatizar(text):
    #stemmatizar
    t0 = time.time()
    spanish_stemmer = SnowballStemmer('spanish')
    
    tokens = nltk.word_tokenize(text)
    list_stemm = [spanish_stemmer.stem(token) for token in tokens]
    text_stemm = ' '.join(list_stemm)
    
    t1 = time.time()
    t_stemm = t1-t0
    
    return list_stemm, text_stemm, t_stemm


def lemm_spacy(text): #le das la ubicacion del texto

    t0 = time.time()
    nlp = spacy.load('es_core_news_sm')

    doc = nlp(text)
    list_lemmas_spacy = [token.lemma_ for token in doc] #da una lista

    lemmas_spacy = ' '.join(list_lemmas_spacy) #lo hacemos un string

    t1 = time.time()

    t_lemm_spacy = t1-t0

    return list_lemmas_spacy, lemmas_spacy, t_lemm_spacy


def lemm_stanza(text):
    t0 = time.time()

    doc = nlp(text)
    list_lemmas_stanza = [word.lemma for sent in doc.sentences for word in sent.words]

    lemmas_stanza = ' '.join(list_lemmas_stanza)

    t1 = time.time()

    t_lemm_stanza = t1-t0
    
    return list_lemmas_stanza, lemmas_stanza, t_lemm_stanza


#%%

path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Transcripciones_limpias/'

# Lista de los sujetos
Sujetos = ['Sujeto 1', 'Sujeto 2', 'Sujeto 3', 'Sujeto 4', 'Sujeto 5','Sujeto 6','Sujeto 7','Sujeto 8','Sujeto 9','Sujeto 10','Sujeto 11','Sujeto 12','Sujeto 13','Sujeto 14','Sujeto 15', 'Sujeto 16', 'Sujeto 17', 'Sujeto 18', 'Sujeto 19' ,'Sujeto 20','Sujeto 21','Sujeto 22','Sujeto 23','Sujeto 24','Sujeto 25', 'Sujeto 26', 'Sujeto 27', 'Sujeto 28', 'Sujeto 29', 'Sujeto 30']

tema = "presencial"
    
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"

data = {} # Creamos un diccionario vacio
for i, c in tqdm(enumerate(Sujetos)):
    #cargamos el texto
    filename = path + c  + f"/Sujeto{i+1}_{tema}.txt"
    if os.path.isfile(filename) is True: #si existe hacemos esto
        texto_limpio = pre_analisis_texto(filename)
        #lo lemmatizamos
        list_lemmas, lemmas, t_lemm = lemm_stanza(texto_limpio)
        data[i] = lemmas # asignamos los sujetos como key al diccionario y el valor es el texto
    elif os.path.isfile(path + c  + f"/Sujeto{i+1}_{tema}_2021.txt"):
        texto_limpio = pre_analisis_texto(path + c  + f"/Sujeto{i+1}_{tema}_2021.txt")
        #lo lemmatizamos
        list_lemmas, lemmas, t_lemm = lemm_stanza(texto_limpio)
        data[i] = lemmas # asignamos los sujetos como key al diccionario y el valor es el texto
    else:
        print(f"El archivo Sujeto{i+1}_{tema}.txt")

# la acomodamos en un dataframe 

data_combined = {key: [value] for (key, value) in data.items()}
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']    

data_df.index += 1
#guardamos la data

#data_df.to_pickle("/content/drive/My Drive/Ari/FCEN/Docencia/ldd1c2021/Clase: procesamiento del lenguaje natural/corpus.pkl")

data_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza_.csv')
#%% le agrego la columna del texto crudo

def csv_con_texto_crudo(tema):
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
    
    return df_textos

temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

#%%
for tema in temas:
    
    df_textos = csv_con_texto_crudo(tema)
    
    df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujetos'})
    
    df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza_.csv', index=False)
    
    

#%% matriz de numero de cada palabra --> esto tira los numeros 1,2, etc. Si nos interesa conservarlos hay que escribir uno, dos, etc.

# Vamos a crear la matriz de documentos-terminos usando usando CountVectorizer, y excluiremos las stop words del espaniol

# Inicializo el modelo 
cv = CountVectorizer()
# Ajustamos el modelo y lo aplicamos al texto de nuestro dataframe generando una matriz esparsa
data_cv = cv.fit_transform(data_df.transcript)
# Nos creamos un dataframe transformando a densa la matriz generada recien que tiene como columnas las palabras (terminos) y como filas los documentos
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
# Le asignamos los indices del dataframe anterior
data_dtm.index = data_df.index

df_contandopalabras = data_dtm.copy(deep=True)

df_contandopalabras_paracalculo = data_dtm.copy(deep=True)

# cantidad de palabras total que dijo cada sujeto

palabras_tot = data_dtm.sum(axis='columns')

df_contandopalabras.insert(0, "num_tot_palabras", palabras_tot, True)

#cantidad de palabras unicas total que dijo cada sujeto

df_contandopalabras_paracalculo[df_contandopalabras_paracalculo > 1] = 1

palabras_unicas_tot = df_contandopalabras_paracalculo.sum(axis='columns')

df_contandopalabras.insert(1, "num_tot_palabras_unicas", palabras_unicas_tot, True)

# le agregamos la data de la encuentra post entrevista

path2 = f"C:/Users/Usuario/Desktop/Cori/Tesis/Encuestas/Postentrevista/Primera_entrevista/Post_entrevista_{tema}.csv"

df_encuesta = pd.read_csv(path2)

df_encuesta.index += 1

df_contandopalabras.insert(2, "cuanto_recordaste", df_encuesta["Cuanto_recordaste"], True)
df_contandopalabras.insert(3, "tipo_emocion", df_encuesta["Tipo_emocion"], True)
df_contandopalabras.insert(4, "intensidad_emocion", df_encuesta["Intensidad_emocion"], True)

df_contandopalabras.index.name = 'Sujeto'

#%% hagamos el análisis de sentimiento 


#%%
#guardemos la data

df_contandopalabras.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_palabras/{tema}_cuentocadapalabra.csv')

# Guardamos en formato pickle el dataframes
#data_dtm.to_pickle("/content/drive/My Drive/Ari/FCEN/Docencia/ldd1c2021/Clase: procesamiento del lenguaje natural/dtm.pkl")
# Guardamos tambien el objeto CountVectorize
#pickle.dump(cv, open("/content/drive/My Drive/Ari/FCEN/Docencia/ldd1c2021/Clase: procesamiento del lenguaje natural/cv.pkl", "wb"))


#para cargarlo:
#pd.read_pickle('/content/drive/My Drive/Ari/FCEN/Docencia/ldd1c2021/Clase: procesamiento del lenguaje natural/dtm.pkl')

#%% pasemos a frecuencia por sujeto de cada palabra

data_frec = data_dtm.apply(lambda x: x/x.sum(), axis=1)

#%% palabras mas usadas por sujeto

#trasponemos el df
data = data_dtm.transpose()

data.head()

# Creo un diccionario
top_dict = {}

# Por cada sujeto
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30) # Ordeno las filas en forma decreciente y me quedo con las 30 palabras mas usadas
    top_dict[c]= list(zip(top.index, top.values))       # le asigno el sujeto a la key del diccionario y como valor una tupla con la palabra y su frecuencia

print(top_dict)
print("\n")

# Imprimo las 15 palabras mas frecuentes por año
for sujeto, top_words in top_dict.items():
    print(sujeto) # imprimo la key
    print(', '.join([word for word, count in top_words[0:14]])) # imprimo las palabras en orden decreciente segun frecuencia y separadas con espacio y coma


