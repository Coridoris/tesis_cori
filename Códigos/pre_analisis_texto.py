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
En la septima celda hay una comparacion entre los distintos modelos para lematizar
'''

import numpy as np
import re       # libreria de expresiones regulares
import string   # libreria de cadena de caracteres
import time



filename1 = 'C:/Users/Usuario/Desktop/Cori/Tesis/Audios/audio_mio/partido_arabia_concorreccionpuntuacion.txt'
#con = np.loadtxt(filename1,  dtype=str) #, skiprows=1 posible atributo de la funcion, delimiter=',',
#sin = np.loadtxt(filename2, dtype=str)

pre_texto = open(filename1, 'r', encoding='utf-8') #esto es un <class '_io.TextIOWrapper'>

texto = pre_texto.read() #esto es un string

print(texto)

#%% primer limpieza de caracteres

#print(string.punctuation) #son caracteres de puntuacion
#print(re.escape(string.punctuation)) #con esto ves la expresion regular de un caracter especial, o sea el # para verlo en un texto tenes que poner
# \#, sino hace otra cosa, entonces re.escape te muestra la expresion regular

# Defino una funcion que recibe un texto y devuelve el mismo texto sin singnos,
def limpia_caracteres_puntuacion(text):
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
    return text

# Dataframe que resulta de aplicarle a las columnas la funcion de limpieza
solo_texto = limpia_caracteres_puntuacion(texto)


print(solo_texto)


#%% saco stopwords

# Cargamos del paquete nltk las stopwords del español a la lista "lines"
import nltk
#nltk.download('stopwords') # hay que descargar este modulo en particular

stop_words = nltk.corpus.stopwords.words('spanish')
#dejo por si necesito agregar o sacar una stopword
#stopwords.append('ejemplo')
#stopwords.remove('ejemplo')

stopwords_dict = {word: 1 for word in stop_words}

texto_sin_sw = " ".join([word for word in solo_texto.split() if word not in stopwords_dict])

print(texto_sin_sw)

#si lo quiero como lista
#lista_texto_sin_sw = [w for w in solo_texto if not w in stop_words]



#%% stemming con ntkl

from nltk.stem import SnowballStemmer
#nltk.download('punkt')

t0 = time.time()
spanish_stemmer = SnowballStemmer('spanish')

tokens = nltk.word_tokenize(texto_sin_sw)
stemm = [spanish_stemmer.stem(token) for token in tokens]
texto_stemm = ' '.join(stemm)

t1 = time.time()
t_stemm = t1-t0
print('Tardo', t_stemm, 'seg')

#print(texto_stemm)

#%%lemmatizar con spacy
#!pip install spacy
#!python -m spacy download es_core_news_sm

import spacy

t0 = time.time()
nlp = spacy.load('es_core_news_sm')

text = texto_sin_sw
doc = nlp(text)
list_lemmas_spacy = [token.lemma_ for token in doc] #da una lista

lemmas_spacy = ' '.join(list_lemmas_spacy) #lo hacemos un string

t1 = time.time()

t_lemm_spacy = t1-t0
print('Tardo', t_lemm_spacy, 'seg')

#print(lemmas_spacy)

#%% lemmatizar con stanza

import stanza

#t0 = time.time()
#stanza.download('es') # descarga el modelo en español
nlp = stanza.Pipeline('es')
#%%
#text = texto_sin_sw
text = "Dijeron que encontraron unas lindas sillas para la mesa"
doc = nlp(text)
list_lemmas_stanza = [word.lemma for sent in doc.sentences for word in sent.words]

lemmas_stanza = ' '.join(list_lemmas_stanza)

#t1 = time.time()

t_lemm_stanza = t1-t0
print('Tardo', t_lemm_stanza, 'seg')

#print(lemmas_stanza)

#%%

print('Texto stemmatizado')
print(texto_stemm)
print('Texto lematizado con spacy')
print(lemmas_spacy)
print('Texto lematizado con stanza')
print(lemmas_stanza)


print('stemmatizar con ntkl tarda ', t_stemm, 's')
print('lemmatizar con spacy tarda ', t_lemm_spacy, 's')
print('lemmatizar con stanza tarda ', t_lemm_stanza, 's')

print('Las diferencias entre las lematizaciones son')

for i in range(len(list_lemmas_stanza)):
    if list_lemmas_spacy[i] != list_lemmas_stanza[i]:
        print('spacy lemmatizo', list_lemmas_spacy[i])
        print('stanza lemmatizo', list_lemmas_stanza[i])
        
#%% Dilema: me gusta mas la lemmatizacion de stanza, pero tarda 19 veces mas, no se cuanto 
#podria llegar a tardar en todo el analisis, por ahora me defino una func con stanza y otra con spacy

#LIBRERIAS


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

texto = pre_analisis_texto(filename1)

list_stemm, texto_stemm, t_stemm = stemmatizar(texto)

list_lemmas_spacy, lemmas_spacy, t_lemm_spacy = lemm_spacy(texto)

list_lemmas_stanza, lemmas_stanza, t_lemm_stanza = lemm_stanza(texto)


#%%

texto = ['Me acuerdo, lo vi en mi casa. Estábamos mi papá mi mamá y yo nada mas. Me acuerdo que mi perro sufría mucho, de hecho temblaba. Porque estaba el volumen de la tele muy alto y como que entendía que estaba pasando algo él. En mi casa hay un sillón grande y un sillón chico, en el que estaba sentada mi mamá y se subía al sillón y escondía la cabeza. Me acuerdo, no acuerdo ni siquiera del orden de los goles. Pero me acuerdo de una sensación de, dale, no puede ser. Me acuerdo que me dejó una sensación de intranquilidad muy fuerte a nivel corporal. Incluso que me duró hasta el siguiente partido, estuve incómodo, mal. Me acuerdo que había tomado la resolución antes del partido. De que este mundial iba a estar completamente comprometido con la idea de que íbamos a ser campeones. Para mí éramos campeones. Si perdía, ya está, esta era la última. Era lo peor que podía pasar. Y fa, la primera ya así, me acuerdo, todo sensaciones lo que te estoy diciendo. Y respecto al día en concreto, no me acuerdo mucho. Me acuerdo que por esa semana, no sé si fue, o fue cerca de otro partido, tal vez. Tal vez me lo confundo. Estaba nublado. Como que acompañaba el sentimiento, digamos. Me acuerdo de colgar mis dos camisetas. La de Boca y la de Argentina suplente, porque no me quedan, no las podía poner. De colgar una bandera de Argentina. Y no más.']

list_lemmas_stanza, lemmas_stanza, t_lemm_stanza = lemm_stanza(texto)