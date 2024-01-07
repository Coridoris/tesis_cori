# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:33:41 2023

@author: cori
"""
'''
en el siguiente notebook vamos limpiar texto
En la primera celda se baja el texto como un unico string
En la segunda celda le limpiamos todas las puntuaciones (signos de pregunta, comas, puntos, espacios)
En la tercera celda sacamos las stop words
En la cuarte selda stemmeamos usando nktl (esta libreria no deja lemmatizar en español)
En la quinta celda lemmatizamos usando spacy
En la sexta celda lemmatizamos usando stanza
'''

import numpy as np
import re       # libreria de expresiones regulares
import string   # libreria de cadena de caracteres



filename1 = 'partido_arabia_concorreccionpuntuacion.txt'
filename2 = 'partido_arabia_sincorreccionpuntuacion.txt'
#con = np.loadtxt(filename1,  dtype=str) #, skiprows=1 posible atributo de la funcion, delimiter=',',
#sin = np.loadtxt(filename2, dtype=str)


con = open(filename1, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>
sin = open(filename2, 'r', encoding='utf-8')
#f = io.open('textfr', 'r', encoding='utf-8')

print(type(con))

content_con = con.read() #esto es un string
content_sin = sin.read()

print(type(content_con))

print(content_con)
#%%
print(string.punctuation)
print(re.escape(string.punctuation)) #con esto ves la expresion regular de un caracter especial, o sea el # para verlo en un texto tenes que poner
# \#, sino hace otra cosa, entonces re.escape te muestra la expresion regular
#%% primer limpieza de caracteres

# Defino una funcion que recibe un texto y devuelve el mismo texto sin singnos,
def clean_text_round1(text):
    # pasa las mayusculas del texto a minusculas
    text = text.lower()                                                               
    # reemplaza singnos de puntuacion por espacio en blanco. %s es \S+ que quiere decir cualquier caracter que no sea un espacio en blanco
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    #saca los enter
    text = re.sub('\n', '', text)  #si lo hago arriba deja el enter, no se bien pq
    #saca los ¿
    text = re.sub('¿', '', text)
    #re.sub sustituye una secuencia de caracteres por otra:
    #re.sub('lo que queres reemplazar', 'por lo que lo reemplazas', texto donde esta) 
    #por ahora saco la ultima parte, no se dsi dejarlo
    # remueve palabras que contienen numeros    
    #text = re.sub('\w*\d\w*', '', text)                              
    return text

# Defino una funcion anonima que al pasarle un argumento devuelve el resultado de aplicarle la funcion anterior a este mismo argumento
round1 = lambda x: clean_text_round1(x)
#esto sirve si meto todo en un dataframe

# Dataframe que resulta de aplicarle a las columnas la funcion de limpieza
content_con_clean = clean_text_round1(content_con)
content_sin_clean = clean_text_round1(content_sin)

print(content_con_clean==content_sin_clean)
print(content_con_clean)
print(content_sin_clean)
#bien, quedan igual

#%% saco stopwords

# Cargamos del paquete nltk las stopwords del español a la lista "lines"
import nltk
nltk.download('stopwords') # hay que descargar este modulo en particular

stop_words = nltk.corpus.stopwords.words('spanish')
#dejo por si necesito agregar o sacar una stopword
#stopwords.append('ejemplo')
#stopwords.remove('ejemplo')

stopwords_dict = {word: 1 for word in stop_words}
content_sin_sw = " ".join([word for word in content_con_clean.split() if word not in stopwords_dict])

print(content_sin_sw)
#content_sin_sw = [w for w in content_con_clean if not w in stop_words]


#%% intento de lematizar con ntkl en español pero no se puede

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

nltk.download('omw')
nltk.download('stopwords')
nltk.download('punkt')


lemmatizer = WordNetLemmatizer()
text = 'Los gatos están cazando ratones'
tokens = nltk.word_tokenize(text)
lemmas = [lemmatizer.lemmatize(token, 'spa') for token in tokens]
print(lemmas)

#%% stemming

from nltk.stem import SnowballStemmer
spanish_stemmer = SnowballStemmer('spanish')

tokens = nltk.word_tokenize(content_sin_sw)
stemm = [spanish_stemmer.stem(token) for token in tokens]
stemm = ' '.join(stemm)


#%%lemmatizar con spacy
#!pip install spacy
#!python -m spacy download es_core_news_sm

import spacy

nlp = spacy.load('es_core_news_sm')
text = content_sin_sw
doc = nlp(text)
lemmas = [token.lemma_ for token in doc]
print(lemmas)

lemas = ' '.join(lemmas)
print(lemas)

#%% lemmatizar con stanza

import stanza

stanza.download('es') # descarga el modelo en español
nlp = stanza.Pipeline('es')
#%%
text = content_sin_sw
doc = nlp(text)
lemmas_c = [word.lemma for sent in doc.sentences for word in sent.words]
print(lemmas_c)

lemas_c = ' '.join(lemmas_c)
print(lemas_c)