# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:46:34 2023

@author: corir
"""

import pandas as pd
import ast
from collections import Counter
from tqdm import tqdm

#da lo mismo q la de abajo, pero la de abajo es mas general pq puedo poner todas las etiquetas que quiera
def find_verb_noun_adj_pairs(tagged_words):
    target_tags = ['NOUN', 'VERB', 'ADJ']
    pairs = []
    
    # Eliminar tuplas con etiqueta 'PUNCT'
    filtered_words = [(word, tag) for word, tag in tagged_words if tag != 'PUNCT']

    for i in range(len(filtered_words) - 1):
        word, tag = filtered_words[i]
        next_word, next_tag = filtered_words[i + 1]

        if tag in target_tags and next_tag in target_tags:
            if tag == 'VERB' and (next_tag == 'NOUN' or next_tag == 'ADJ' or next_tag == 'VERB'):
                pairs.append((tag, next_tag))
            elif tag == 'ADJ' and (next_tag == 'NOUN' or next_tag == 'ADJ' or next_tag == 'VERB'):
                pairs.append((tag, next_tag))
            elif tag == 'NOUN' and (next_tag == 'NOUN' or next_tag == 'ADJ' or next_tag == 'VERB'):
                pairs.append((tag, next_tag))
                
    return pairs
    

def generate_valid_combinations(target_tags):
    valid_combinations = []

    for tag1 in target_tags:
        for tag2 in target_tags:
            valid_combinations.append((tag1, tag2))

    return valid_combinations

def find_verb_noun_adj_etc_pairs(tagged_words):
    target_tags = ['NOUN', 'VERB', 'ADJ', 'NUM', 'PROPN', 'ADV']
    valid_combinations = generate_valid_combinations(target_tags)
    
    pairs = []
    
    # Eliminar tuplas con etiqueta 'PUNCT'
    filtered_words = [(word, tag) for word, tag in tagged_words if tag != 'PUNCT']

    for i in range(len(filtered_words) - 1):
        word, tag = filtered_words[i]
        next_word, next_tag = filtered_words[i + 1]

        if (tag, next_tag) in valid_combinations:
            pairs.append((tag, next_tag))
    
    return pairs


def find_pairs(tagged_words):
    pairs = []
    
    # Eliminar tuplas con etiqueta 'PUNCT'
    filtered_words = [(word, tag) for word, tag in tagged_words if tag != 'PUNCT']

    for i in range(len(filtered_words) - 1):
        word, tag = filtered_words[i]
        next_word, next_tag = filtered_words[i + 1]
        
        pairs.append((tag, next_tag))
        
    return pairs

# Ejemplo de uso con la lista de etiquetas y palabras proporcionada
#tagged_words = [('Este', 'DET'), ('recuerdo', 'NOUN'), ('me', 'PRON'), ('levanté', 'VERB'), ('muy', 'ADV'), ('temprano', 'ADV'), ('.', 'PUNCT'), ('Estaba', 'AUX'), ('emocionado', 'ADJ'), (',', 'PUNCT'), ('ya', 'ADV'), ('lo', 'PRON'), ('venía', 'VERB'), ('pensando', 'VERB'), ('hace', 'VERB'), ('bastante', 'ADV'), ('.', 'PUNCT'), ('Aunque', 'SCONJ'), ('a', 'ADP'), ('mí', 'PRON'), ('esto', 'PRON'), ('no', 'ADV'), ('es', 'AUX'), ('algo', 'PRON'), ('que', 'PRON'), ('me', 'PRON'), ('emocione', 'VERB'), ('en', 'ADP'), ('general', 'NOUN'), (',', 'PUNCT'), ('pero', 'CCONJ'), ('bueno', 'ADJ'), (',', 'PUNCT'), ('al', 'ADP'), ('ser', 'AUX'), ('la', 'DET'), ('selección', 'NOUN'), (',', 'PUNCT'), ('sí', 'INTJ'), ('.', 'PUNCT'), ('Recuerdo', 'NOUN'), ('levantarme', 'ADJ'), ('...', 'PUNCT'), ('Creo', 'VERB'), ('haber', 'AUX'), ('comprado', 'VERB'), ('facturas', 'NOUN'), ('el', 'DET'), ('día', 'NOUN'), ('anterior', 'ADJ'), (',', 'PUNCT'), ('o', 'CCONJ'), ('ese', 'DET'), ('mismo', 'DET'), ('día', 'NOUN'), (',', 'PUNCT'), ('ir', 'AUX'), ('a', 'ADP'), ('comprar', 'VERB'), ('yo', 'PRON'), ('a', 'ADP'), ('mi', 'DET'), ('papá', 'NOUN'), (',', 'PUNCT'), ('ya', 'ADV'), ('no', 'ADV'), ('me', 'PRON'), ('acuerdo', 'VERB'), ('.', 'PUNCT'), ('Para', 'ADP'), ('desayunar', 'VERB'), ('rico', 'NOUN'), ('mientras', 'SCONJ'), ('lo', 'PRON'), ('veíamos', 'VERB'), (',', 'PUNCT'), ('porque', 'SCONJ'), ('era', 'AUX'), ('muy', 'ADV'), ('temprano', 'ADJ'), ('.', 'PUNCT'), ('Recuerdo', 'VERB'), ('que', 'SCONJ'), ('mi', 'DET'), ('hermano', 'NOUN'), ('no', 'ADV'), ('se', 'PRON'), ('levantó', 'VERB'), (',', 'PUNCT'), ('no', 'ADV'), ('le', 'PRON'), ('importaba', 'VERB'), ('tanto', 'ADV'), ('.', 'PUNCT'), ('Y', 'CCONJ'), ('lo', 'PRON'), ('vi', 'VERB'), ('solo', 'NOUN'), ('con', 'ADP'), ('mi', 'DET'), ('papá', 'NOUN'), ('.', 'PUNCT'), ('Mi', 'DET'), ('mamá', 'NOUN'), ('tampoco', 'ADV'), ('se', 'PRON'), ('levantó', 'VERB'), (',', 'PUNCT'), ('me', 'PRON'), ('parece', 'VERB'), ('.', 'PUNCT'), ('Fue', 'AUX'), ('insólito', 'ADJ'), ('lo', 'PRON'), ('que', 'PRON'), ('perdimos', 'VERB'), ('.', 'PUNCT'), ('Recuerdo', 'VERB'), ('putear', 'VERB'), ('a', 'ADP'), ('la', 'DET'), ('pantalla', 'NOUN'), ('por', 'ADP'), ('un', 'DET'), ('montón', 'NOUN'), ('de', 'ADP'), ('offsides', 'NOUN'), ('que', 'PRON'), ('eran', 'AUX'), ('dudosos', 'ADJ'), (',', 'PUNCT'), ('sobre', 'ADP'), ('todo', 'PRON'), ('.', 'PUNCT'), ('Recuerdo', 'VERB'), ('mucho', 'ADV'), ('el', 'DET'), ('post', 'PROPN'), (',', 'PUNCT'), ('de', 'ADP'), ('decepción', 'NOUN'), (',', 'PUNCT'), ('aunque', 'SCONJ'), ('también', 'ADV'), ('había', 'AUX'), ('un', 'DET'), ('ambiente', 'NOUN'), ('de', 'ADP'), ('quizás', 'ADV'), ('motivación', 'NOUN'), ('dentro', 'ADV'), ('de', 'ADP'), ('la', 'DET'), ('misma', 'DET'), ('selección', 'NOUN'), ('de', 'ADP'), ('no', 'ADV'), ('rendirse', 'VERB'), ('.', 'PUNCT'), ('Creo', 'VERB'), ('que', 'SCONJ'), ('después', 'ADV'), ('de', 'ADP'), ('terminar', 'VERB'), ('el', 'DET'), ('partido', 'NOUN'), ('me', 'PRON'), ('volví', 'VERB'), ('a', 'ADP'), ('la', 'DET'), ('cama', 'NOUN'), ('a', 'ADP'), ('dormir', 'VERB'), (',', 'PUNCT'), ('porque', 'SCONJ'), ('había', 'AUX'), ('dormido', 'VERB'), ('muy', 'ADV'), ('poco', 'PRON'), (',', 'PUNCT'), ('también', 'ADV'), ('por', 'ADP'), ('los', 'DET'), ('nervios', 'NOUN'), ('creo', 'VERB'), ('.', 'PUNCT'), ('Y', 'CCONJ'), ('creo', 'VERB'), ('que', 'SCONJ'), ('ese', 'DET'), ('mismo', 'DET'), ('día', 'NOUN'), (',', 'PUNCT'), ('a', 'ADP'), ('la', 'DET'), ('noche', 'NOUN'), (',', 'PUNCT'), ('curse', 'VERB'), ('.', 'PUNCT'), ('No', 'ADV'), ('estoy', 'AUX'), ('seguro', 'ADJ'), ('de', 'ADP'), ('que', 'SCONJ'), ('día', 'NOUN'), ('fue', 'AUX'), (',', 'PUNCT'), ('pero', 'CCONJ'), ('me', 'PRON'), ('parece', 'VERB'), ('que', 'SCONJ'), ('fue', 'AUX'), ('así', 'ADV'), ('.', 'PUNCT'), ('Y', 'CCONJ'), ('no', 'ADV'), ('sé', 'VERB'), ('si', 'SCONJ'), ('hay', 'AUX'), ('más', 'ADV'), ('.', 'PUNCT')]
#result = find_verb_noun_adj_pairs(tagged_words)
#print(result)

def dataframe_red(tema, funcion, elimina_frases =  True):
    if elimina_frases == False:
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{tema}.csv'
    else:
        path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/spyCy/contando_sust_verb_adj_spyCy_{tema}_sinMeacuerdo.csv'

    df_clasificacion = pd.read_csv(path)
    
    df_clasificacion["clasificacion_total"] = df_clasificacion["clasificacion_total"].apply(ast.literal_eval)
    
    df_clasificacion["verb_noun_adj_pairs"] = df_clasificacion["clasificacion_total"].apply(funcion)
    
    #data = df_clasificacion["verb_noun_adj_pairs"].tolist() #lista de strings
    
    #data = [ast.literal_eval(item) for item in data] #lo paso a lista de listas
    
    # Contar las tuplas para cada sujeto
    #counters = [Counter(tuplas) for tuplas in data]
    
    # Crear un DataFrame a partir de los contadores
    #df = pd.DataFrame(counters)
    
    # Si alguna tupla no aparece para un sujeto, rellenar con 0
    #df = df.fillna(0).astype(int)
    
    # Visualizar el DataFrame
    #print(df)
    
    # Contar las tuplas para cada sujeto
    df_clasificacion['Contadores'] = df_clasificacion["verb_noun_adj_pairs"].apply(Counter)
    
    # Expandir los contadores en columnas separadas en el DataFrame
    df_contadores = df_clasificacion['Contadores'].apply(pd.Series)
    
    # Rellenar NaN con 0 y convertir a tipo entero
    df_contadores = df_contadores.fillna(0).astype(int)
    
    # Concatenar los contadores al DataFrame original
    df_clasificacion = pd.concat([df_clasificacion, df_contadores], axis=1)
    
    # Eliminar la columna 'Contadores' si no la necesitas más
    df_clasificacion = df_clasificacion.drop(columns=['Contadores', 'Unnamed: 0', 'sujeto'])
    
    df_clasificacion.index += 1
    df_clasificacion.index.name = 'sujeto'
    
    return df_clasificacion

#%%

temas = ["arabia", "campeones_del_mundo", "antesdevenir", "presencial", "cfk"]

for i in tqdm(range(len(temas))):

    '''
    si quiero que tenga las frases me acuerdo o recuerdo tengo que agregar la variable elimina_frases =  False
    a la funcion dataframe_red (la que uso en las siguientes dos lineas)
    cambia el nombre del archivo que guardas
    '''
    df_red = dataframe_red(temas[i], find_verb_noun_adj_etc_pairs)
    df_red2 = dataframe_red(temas[i], find_pairs)
    
    df_red.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/df_red_sustverbadjetc_{temas[i]}_sin_meacuerdo.csv')  
    df_red2.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Red/df_red_partofspeech_{temas[i]}_sin_meacuerdo.csv')  
    
        
    