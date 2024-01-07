# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:42:15 2023

@author: Usuario
"""



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


for i in range(1,2):

    palabra_a_palabra = df_textos['transcript'][i].split()
    
    positivas = []
    negativas = []
    
    positivas_unicas = []
    negativas_unicas = []
    
    palabras_ya_sumadas = []
    
    for j in range(len(palabra_a_palabra)):
        sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra[j])['global_sentiment']
        if palabra_a_palabra[j] not in palabras_ya_sumadas:
            print("no estaba")
            if sentimiento_palabra > 0:
                positivas.append(palabra_a_palabra[j])
                positivas_unicas.append(palabra_a_palabra[j])
            if sentimiento_palabra < 0:
                negativas.append(palabra_a_palabra[i])
                negativas_unicas.append(palabra_a_palabra[j])
        else:
            print(j)
            print(palabra_a_palabra[j] not in palabras_ya_sumadas)
            print(palabra_a_palabra[j], palabras_ya_sumadas)
            if sentimiento_palabra > 0:
                positivas.append(palabra_a_palabra[j])

            if sentimiento_palabra < 0:
                negativas.append(palabra_a_palabra[j])
        palabras_ya_sumadas.append(palabra_a_palabra[j])

            
    tasa_positividad.append(len(positivas)/len(palabra_a_palabra))
    tasa_negatividad.append(len(negativas)/len(palabra_a_palabra))
    
    intensidad.append((len(positivas)+len(negativas))/len(palabra_a_palabra))
    neutralidad.append(1-intensidad[-1])
    
    palabra_a_palabra_unicas = list(set(palabra_a_palabra))
    
    tasa_positividad_unicas.append(len(positivas_unicas)/len(palabra_a_palabra_unicas))
    tasa_negatividad_unicas.append(len(negativas_unicas)/len(palabra_a_palabra_unicas))
    
    intensidad_unicas.append((len(positivas_unicas)+len(negativas_unicas))/len(palabra_a_palabra_unicas))
    neutralidad_unicas.append(1-intensidad_unicas[-1])
    
    
tasa_positividad_unicas_ = []
tasa_negatividad_unicas_ = []
intensidad_unicas_ = []
neutralidad_unicas_ = []

for i in range(1,2):

    palabra_a_palabra = df_textos['transcript'][i].split()
    
    palabra_a_palabra_unicas = list(set(palabra_a_palabra))
    
    positivas_unicas_ = []
    negativas_unicas_ = []

    
    for j in range(len(palabra_a_palabra_unicas)):
        
        sentimiento_palabra = sentimental.compute_sentiment(palabra_a_palabra_unicas[j])['global_sentiment']
        
        if sentimiento_palabra > 0:
            positivas_unicas_.append(palabra_a_palabra_unicas[j])
        if sentimiento_palabra < 0:
            negativas_unicas_.append(palabra_a_palabra_unicas[j])

    
    tasa_positividad_unicas_.append(len(positivas_unicas_)/len(palabra_a_palabra_unicas))
    tasa_negatividad_unicas_.append(len(negativas_unicas_)/len(palabra_a_palabra_unicas))
    
    intensidad_unicas_.append((len(positivas_unicas_)+len(negativas_unicas_))/len(palabra_a_palabra_unicas))
    neutralidad_unicas_.append(1-intensidad_unicas_[-1])            
 

# df_textos['tasa_positividad'] = tasa_positividad
# df_textos['tasa_negatividad'] = tasa_negatividad

# df_textos['intensidad'] = intensidad
# df_textos['neutralidad'] = neutralidad


# df_textos = df_textos.drop(['Unnamed: 0'], axis=1)

# df_textos = df_textos.rename(columns={'Unnamed: 0': 'Sujeto'})

# df_textos.index += 1

# df_textos.index.name = 'Sujeto'
    
