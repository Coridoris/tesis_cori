# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:26:23 2023

@author: corir
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

import nltk
nltk.download('punkt')  # Descargar el tokenizador de oraciones (solo se necesita ejecutar una vez)
from nltk.tokenize import sent_tokenize
#%%
#Sentences we want to encode. Example:
sentence1 = ['En la galaxia lejana, un astronauta de chocolate baila con los pájaros morados.']#['Mi perro canta de noche']

sentence2 =  ['Mañana me van a operar de las amigdalas.'] #['Mi gato canta de día']

sentence3 = ['Quiero irme a mi casa.']#['Mañana todos vamos a ir al parque']

sentence4 = ['El medicamento se toma tres veces por semana']


#Sentences are encoded by calling model.encode()
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)
embedding4 = model.encode(sentence4)

cosine_score1 = util.cos_sim(embedding1, embedding2)

cosine_score2 = util.cos_sim(embedding1, embedding3)

cosine_score3 = util.cos_sim(embedding1, embedding4)

print(cosine_score1, cosine_score2, cosine_score3)

#%%

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
#%%
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[2], sentences2[i], cosine_scores[2][i]))
    
#%%
# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

tema = temas[1]

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

#%%

text = df_textos['texto_crudo'][j]
text = "En la galaxia lejana, un astronauta de chocolate baila con los pájaros morados. Mañana me van a operar de las amigdalas. Quiero irme a mi casa. A Franco le gusta mucho el ramen. El color verde del techo representa la adversidad. Un abismo separa madre e hija."

sentence = sent_tokenize(text)


#print(sentence)

embeddings = []
for i in range(len(sentence)):
    #Compute embedding 
    embeddings.append(model.encode(sentence[i], convert_to_tensor=True))

cosine_scores1 = []
#Compute cosine-similarities
for i in range(len(sentence)-1):
    cosine_scores1.append(util.cos_sim(embeddings[i], embeddings[i+1]).item())
    
coherence1 = np.mean(cosine_scores1)

print(coherence1)

cosine_scores2 = []
#Compute cosine-similarities
for i in range(len(sentence)-2):
    cosine_scores2.append(util.cos_sim(embeddings[i], embeddings[i+2]).item())
    
coherence2 = np.mean(cosine_scores2)

print(coherence2)

#%%

for j in tqdm(range(len(Sujetos))):
    while True:
        try:
            sust, verb, adj = generate_list_sust_verbs_adj(df_textos['transcript'][j], key)