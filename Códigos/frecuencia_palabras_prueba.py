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

#ademas para lemm con spacy
#!pip install spacy
#!python -m spacy download es_core_news_sm
import spacy

#para stanza
import stanza
#stanza.download('es') # descarga el modelo en español
nlp = stanza.Pipeline('es')


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

filename1 = 'C:/Users/Usuario/Desktop/Cori/Tesis/Audios/audio_mio/partido_arabia_concorreccionpuntuacion.txt'
filename2 = 'C:/Users/Usuario/Desktop/Cori/Tesis/Audios/audio_mio/partido_arabia_sincorreccionpuntuacion.txt'
#con = np.loadtxt(filename1,  dtype=str) #, skiprows=1 posible atributo de la funcion, delimiter=',',
#sin = np.loadtxt(filename2, dtype=str)

pre_texto_con = open(filename1, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
pre_texto_sin = open(filename2, 'r', encoding='utf-8')

#texto = pre_texto.read() #esto es un string

texto_con = pre_analisis_texto(filename1)
texto_sin = pre_analisis_texto(filename2)

#list_stemm, texto_stemm, t_stemm = stemmatizar(texto_con)

#list_lemmas_spacy, lemmas_spacy, t_lemm_spacy = lemm_spacy(texto_con)

list_lemmas_stanza_con, lemmas_stanza_con, t_lemm_stanza_con = lemm_stanza(texto_con)

list_lemmas_stanza_sin, lemmas_stanza_sin, t_lemm_stanza_sin = lemm_stanza(texto_sin)

#%% acomodemos la data en un dataframe que tenga en cada columna a un sujeto


# Lista con los años de las publicaciones
#sujetos = ['2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

textos = [lemmas_stanza_con, lemmas_stanza_sin]
texto = ['con', 'sin']

#data = {} # Creamos un diccionario vacio
#for i, c in enumerate(sujetos):
#    with open(path + c + ".txt", "rb") as file: # abrimos cada archivo .txt
#        data[c] = pickle.load(file) # asignamos los años como key al diccionario y el valor es el texto despickleado

data = {} # Creamos un diccionario vacio
for i, c in enumerate(textos):
    data[texto[i]] = c # asignamos los años como key al diccionario y el valor es el texto despickleado


data_combined = {key: [value] for (key, value) in data.items()}
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']      

#guardamos la data

#data_df.to_pickle("/content/drive/My Drive/Ari/FCEN/Docencia/ldd1c2021/Clase: procesamiento del lenguaje natural/corpus.pkl")

#%% matriz de numero de cada palabra --> esto tira los numeros 1,2, etc. Si nos interesa conservarlos hay que escribir uno, dos, etc.

# Vamos a crear la matriz de documentos-terminos usando usando CountVectorizer, y excluiremos las stop words del espaniol
from sklearn.feature_extraction.text import CountVectorizer

# Inicializo el modelo 
cv = CountVectorizer()
# Ajustamos el modelo y lo aplicamos al texto de nuestro dataframe generando una matriz esparsa
data_cv = cv.fit_transform(data_df.transcript)
# Nos creamos un dataframe transformando a densa la matriz generada recien que tiene como columnas las palabras (terminos) y como filas los documentos
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
# Le asignamos los indices del dataframe anterior
data_dtm.index = data_df.index

#guardemos la data

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


