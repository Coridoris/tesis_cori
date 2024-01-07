# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:16:52 2023

@author: corir
"""
#%%librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#%%funciones

def contar_palabras_en_texto(texto, lista_palabras):
    # Convertir el texto a minúsculas para comparaciones sin distinción entre mayúsculas y minúsculas
    texto = texto.lower()

    # Dividir el texto en palabras
    palabras = texto.split()

    # Inicializar el contador de palabras
    contador = 0

    # Contar las palabras que están en la lista proporcionada
    for palabra in palabras:
        if palabra in lista_palabras:
            contador += 1
    #si lo quiero normalizado por el num total de palabras cambiar a contador/len(palabras)
    return contador/len(palabras) 

def eliminar_frase(texto, frase_a_eliminar):
    # Reemplaza la frase a eliminar con una cadena vacía
    texto_sin_frase = texto.replace(frase_a_eliminar, "")
    return texto_sin_frase

def contar_palabras(texto):
    palabras = texto.split()  # Dividir el texto en palabras por los espacios
    return len(palabras)  # Devolver la cantidad de palabras


def primera_tercera_persona(tema, pronombres_primera, pronombres_tercera ,frase_a_eliminar = None):
    '''
    a esta función le metes un csv de textos y te devuelve un csv con nro sujeto, condicion, nro_palabras_prim_persona, nro_palabras_tercera_persona
    o sea, algo listo para meter en ELcsv
    '''
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Texto_lemmatizado/Stanza/{tema}_usando_stanza_.csv'
        
    df_textos = pd.read_csv(path)
    
    if frase_a_eliminar != None:
        for i in range(len(frase_a_eliminar)):
            df_textos["texto_crudo"] = df_textos["texto_crudo"].apply(eliminar_frase, args=(frase_a_eliminar[i],))
    
    df_textos["tot"] = df_textos["texto_crudo"].apply(contar_palabras)
    
    df_textos["primera_persona_norm"] = df_textos["texto_crudo"].apply(contar_palabras_en_texto, args=(pronombres_primera,))
    
    df_textos["tercera_persona_norm"] = df_textos["texto_crudo"].apply(contar_palabras_en_texto, args=(pronombres_tercera,))
    
    #si lo quiero normalizado por el num total de palabras cambiar la función contar_palabras_en_texto el return por lo comentado
    #y ya va a estar directamente normalizado
    #si lo quiero normalizado por el num total de palabras en primera+tercera usar lo que sigue
    #df_textos["tot_primera+tercera"] = df_textos["primera_persona"] + df_textos["tercera_persona"]
    
    #df_textos["primera_persona_norm"] = df_textos["primera_persona"]/df_textos["tot_primera+tercera"]
    
    #df_textos["tercera_persona_norm"] = df_textos["tercera_persona"]/df_textos["tot_primera+tercera"]
    
    # Selecciona las columnas que queremos
    columnas_seleccionadas = ['Sujetos', 'primera_persona_norm', 'tercera_persona_norm']#, "tot_primera+tercera", "tot"]
    
    #reescribe el dataframe
    df_textos = df_textos[columnas_seleccionadas]
    
    df_textos.insert(1, 'Condición', tema)

    return df_textos

#%% variables generales
# Lista de los sujetos
   
Sujetos = ['0']*30
for j in range(30):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

#%% armo el csv con el número de palabras en primera persona y el número en tercera persona

pronombres_primera = ['yo', 'me', 'mi', 'mí', 'conmigo', 'nosotros', 'nos', 'nosotras', 'nuestro', 'nuestra']

pronombres_tercera = ['él', 'lo', 'le', 'se', 'sí', 'consigo', 'ella', 'la', 'ello', 'lo', 'ellos', 'ellas', 'las', 'los', 'les', 'se', 'usted', 'ustedes']

for tema in temas:
    
    df_textos = primera_tercera_persona(tema, pronombres_primera, pronombres_tercera ,frase_a_eliminar = ["Me acuerdo", "Recuerdo", "me acuerdo", "recuerdo"])
        
    df_textos.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Primera-tercera persona/{tema}_prim_ter_persona.csv', index=False)
    
    
#%% guardo en df_textos el num total de palabras y la sum total de palabras en primera y tercera persona y veo si hay correlación

for tema in temas:
    
    df_textos = primera_tercera_persona(tema, pronombres_primera, pronombres_tercera ,frase_a_eliminar = ["Me acuerdo", "Recuerdo", "me acuerdo", "recuerdo"])
        
    prim_tercera = df_textos["tot_primera+tercera"]
    
    tot_palabras = df_textos["tot"]
    
    #Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(tot_palabras, prim_tercera)
    
    # Imprimir el resultado
    print(f"Coeficiente de correlación de Pearson de {tema}:", correlation_matrix[0, 1])
    
    plt.figure()
    plt.plot(tot_palabras, prim_tercera, 'o')
    plt.text(0.75, 0.05, f'R de {tema} = {correlation_matrix[0, 1]:,.2f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    plt.grid(True)
    plt.xlabel('Palabras totales', fontsize = 15)
    plt.ylabel('Palabras primera+tercera', fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()


'''
los respectivos R son
Coeficiente de correlación de Pearson de cfk: 0.9279725379366685
Coeficiente de correlación de Pearson de campeones_del_mundo: 0.9309220790465217
Coeficiente de correlación de Pearson de antesdevenir: 0.9367278949726312
Coeficiente de correlación de Pearson de presencial: 0.9732577616012764
Coeficiente de correlación de Pearson de arabia: 0.9610504854995646
si uso primera+tercera van a quedar totalmente correlacionadas... y la otra opción es menos código...
no se voy con la otra
'''