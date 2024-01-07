# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:36:11 2023

@author: corir
"""
'''
NUEVO MODELO NULO 
'''
#%% hacer un sujeto mezclarle un texto y hacer coherencia en el t

#librerias y funciones
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np
import re
import random
import pandas as pd
import ast
import matplotlib.pyplot as plt 

path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Progreso/9-25'

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

def shuffle_words(texto):
    # Dividir el texto en palabras
    palabras = texto.split()
    
    # Mezclar las palabras
    random.shuffle(palabras)
    
    # Unir las palabras mezcladas en un nuevo texto
    texto_mezclado = ' '.join(palabras)
    
    return texto_mezclado

#%%
#algunas variables que necesito definir en general
temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
#%%

tema = temas[2]

sujeto = 4

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'

df_coherence_evolution = pd.read_csv(path)

texto = df_coherence_evolution['texto_crudo'][sujeto]

coherencia_sujeto = coherence_evolution(texto)[0]
#%% NO CORRER
coherence_evolution_oraciones_mezcladas = []
coherence_evolution_palabras_mezcladas = []

textos_con_oraciones_mezcladas = []

textos_con_palabras_mezcladas = []

for i in tqdm(range(1000)):

    texto_con_oraciones_mezcladas = shuffle_sentences(texto)
    
    texto_con_palabras_mezcladas = shuffle_words(texto)
    
    textos_con_palabras_mezcladas.append(texto_con_palabras_mezcladas)
    textos_con_oraciones_mezcladas.append(texto_con_oraciones_mezcladas)
    
    coherence_evolution_oraciones_mezcladas.append(coherence_evolution(texto_con_oraciones_mezcladas))
    coherence_evolution_palabras_mezcladas.append(coherence_evolution(texto_con_palabras_mezcladas))

#%%   
pruebas_modelo_nulo = {'texto oraciones mezcladas':  textos_con_oraciones_mezcladas, 
                       'coherence evolution oraciones mezcladas': coherence_evolution_oraciones_mezcladas,
                       'texto palabras mezcladas':  textos_con_palabras_mezcladas, 
                       'coherence evolution palabras mezcladas': coherence_evolution_palabras_mezcladas}

df_pruebas_modelo_nulo = pd.DataFrame(pruebas_modelo_nulo)

df_pruebas_modelo_nulo.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/prueba_modelonulo_porsujeto_coherencia_evolucion_{sujeto}_{tema}.csv')

#%%
path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/prueba_modelonulo_porsujeto_coherencia_evolucion_{sujeto}_{tema}.csv'
    
df_pruebas_modelo_nulo = pd.read_csv(path)

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'

df_coherence_evolution = pd.read_csv(path)

texto = df_coherence_evolution['texto_crudo'][sujeto]

coherencia_sujeto = coherence_evolution(texto)[0]


# Convierte las cadenas de texto en listas
df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas'] = df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas'].apply(ast.literal_eval)

df_pruebas_modelo_nulo['coherence evolution palabras mezcladas'] = df_pruebas_modelo_nulo['coherence evolution palabras mezcladas'].apply(ast.literal_eval)

coherencia_oraciones_mezcladas = [vector[0] for vector in df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas']]

coherencia_palabras_mezcladas = [vector[0] for vector in df_pruebas_modelo_nulo['coherence evolution palabras mezcladas']]

q_coherence_rand_oraciones = np.percentile(coherencia_oraciones_mezcladas, 95)
q_coherence_rand_palabras = np.percentile(coherencia_palabras_mezcladas, 95)

# Crear una figura con dos subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Crear los histogramas en cada subgráfico
ax1.hist(coherencia_oraciones_mezcladas, edgecolor='black')
ax2.hist(coherencia_palabras_mezcladas, edgecolor='black')
ax1.axvline(x = coherencia_sujeto, color='red', linestyle='--', label= tema)
ax2.axvline(x = coherencia_sujeto, color='red', linestyle='--', label=tema)
ax1.axvline(x = q_coherence_rand_oraciones, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')
ax2.axvline(x = q_coherence_rand_palabras, color='black', linestyle='-', linewidth = 3, label = 'cuantil 95%')

# Etiquetas y título para cada subgráfico
ax1.set_xlabel('Valores de coherencia')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Mezclando oraciones')
ax1.legend()

ax2.set_xlabel('Valores de coherencia')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Mezclando palabras')
ax2.legend()

# Título general para los dos histogramas
fig.suptitle('Modelo nulo', fontsize=16)

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

#plt.savefig(path_imagenes + '/ModeloNulo_hist_coherencia.png')

# Mostrar los histogramas
plt.show()

#%% hago coherencia mezclando oraciones en el tiempo haciendo el promedio de las 1000

coherencia_nulo_en_t = []
coherencia_nulo_en_t_err = []
for time in range(len(df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas'][0])):
 coherencia_nulo_en_t.append(np.mean([vector[time] for vector in df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas']]))
 coherencia_nulo_en_t_err.append(np.std([vector[time] for vector in df_pruebas_modelo_nulo['coherence evolution oraciones mezcladas']]))
 
t = np.linspace(1, len(coherencia_nulo_en_t), len(coherencia_nulo_en_t)) 
plt.figure()
plt.errorbar(t, coherencia_nulo_en_t, coherencia_nulo_en_t_err, fmt = 'o')
plt.xlabel("Time")
plt.ylabel("Coherencia mezclando oraciones")
plt.show()



