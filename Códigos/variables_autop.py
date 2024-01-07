# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:36:02 2023

@author: corir
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

#%% el santo trial

entrevista = 'Segunda'

ent = 'segunda'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

#%%

for i, tema in tqdm(enumerate(temas)):

    path_autopercepcion = f'C:/Users/Usuario/Desktop/Cori/Tesis/Encuestas/Postentrevista/{entrevista}_entrevista/Post_entrevista_{tema}_{ent}_TODOS.csv'
    
    df = pd.read_csv(path_autopercepcion)
    
    mapeo_sentimientos = {'Positiva': 1, 'Negativa': 0, 'Neutra': -1, 'Positivas': 1, 'Negativas': -1, 'Neutras': 0, np.nan: np.nan}
    
    df['Valencia_autop'] = df['Tipo_emocion'].map(mapeo_sentimientos)
    
    df['ValeInt_autop'] = df['Valencia_autop']*df['Intensidad_emocion']
    
    mapeo_columnas = {'Cuanto_recordaste': 'Recuerdo_autop', 'Intensidad_emocion': 'Intensidad_autop'}
    
    # Usa el método 'rename' para renombrar las columnas
    df.rename(columns=mapeo_columnas, inplace=True)
        
    df.insert(1, 'Condición', tema)
    
    df_autopercepcion = df[['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop']]

    df_autopercepcion.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Variables autopercepcion/variables_autopercepcion_{tema}.csv', index=False)    

