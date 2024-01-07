# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:25:12 2023

@author: corir
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#%%paleta de colores de la presentacion

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

#%%


path = 'C:/Users/Usuario/Desktop/Cori/Tesis/Validación de estimulos/Encuesta sobre eventos del 2022 (Respuestas) - Respuestas de formulario 1.csv'


df = pd.read_csv(path)


edades = df["Edad"]

genero = df["Género"]


arabia_te_acordas = df['¿Te acordas cuando sucedió o te enteraste del evento?']

arabia_cuanto_te_acordas = df['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.']

arabia_tipo_emocion = df['¿Qué tipo de emociones te genera el recuerdo?']

arabia_intensidad_emocion = df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.']

campeones_te_acordas = df['¿Te acordas cuando sucedió o te enteraste del evento?.1']

campeones_cuanto_te_acordas = df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.']

campeones_tipo_emocion = df['¿Qué tipo de emociones te genera el recuerdo?.1']

campeones_intensidad_emocion = df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1']

cfk_te_acordas = df['¿Te acordas cuando sucedió o te enteraste del evento?.2']

cfk_cuanto_te_acordas = df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1']

cfk_tipo_emocion = df['¿Qué tipo de emociones te genera el recuerdo?.2']

cfk_intensidad_emocion = df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2']


df_pres = df[df['Edad'] <= 32]

presencial_te_acordas = df_pres['¿Tuviste la vuelta a las clases presenciales en el 2022?']

presencial_cuanto_te_acordas = df_pres['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8']

presencial_tipo_emocion = df_pres['¿Qué tipo de emociones te genera el recuerdo?.10']

presencial_intensidad_emocion = df_pres['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10']


te_acordas = [arabia_te_acordas, campeones_te_acordas, cfk_te_acordas, presencial_te_acordas]

cuanto_te_acordas = [arabia_cuanto_te_acordas, campeones_cuanto_te_acordas, cfk_cuanto_te_acordas, presencial_cuanto_te_acordas]

tipo_emocion = [arabia_tipo_emocion, campeones_tipo_emocion, cfk_tipo_emocion, presencial_tipo_emocion]

intensidad_emocion = [arabia_intensidad_emocion, campeones_intensidad_emocion, cfk_intensidad_emocion, presencial_intensidad_emocion]

#%%

tema = ['arabia', 'campeones', 'cfk', 'presencial']

for i, data in enumerate(te_acordas):
    plt.figure(i), plt.clf()
    
    # Calcular la cantidad de cada género
    conteo_acuerdo = data.value_counts()

    # Personaliza los colores para los géneros
    colores = [color_celeste, color_celestito, color_palido, color_gris, 'orange'] 

    plt.pie(conteo_acuerdo, labels = conteo_acuerdo.index, colors=colores, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})


    # Muestra el gráfico
    plt.show()

    plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/validacion_{tema[i]}.png', transparent=True)


#%%

tema = ['arabia', 'campeones', 'cfk']



colores = [color_positivas if label == 'Positivas' else color_negativas if label == 'Negativas' else colores_restantes[i] for i, label in enumerate(conteo_acuerdo.index)]


for i, data in enumerate(tipo_emocion):
    
    df['Respuestas'] = data.fillna('')

    # Divide las respuestas múltiples en categorías individuales
    df['Respuestas'] = df['Respuestas'].str.split(', ')
    
    # Luego, crea una nueva columna 'Categorias' que contiene las categorías individuales
    df['Categorias'] = df['Respuestas'].apply(lambda x: [cat for cat in x if cat in ['Positivas', 'Negativas', 'Neutras']])
    
    # Cuenta la frecuencia de cada categoría
    conteo_acuerdo = df['Categorias'].explode().value_counts()
    
    categories = ['Positivas', 'Negativas', 'Neutras']
    for category in categories:
        if category not in conteo_acuerdo:
            conteo_acuerdo[category] = 0

    plt.figure(i), plt.clf()


    plt.pie(conteo_acuerdo, labels = conteo_acuerdo.index, colors=colores, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})


    # Muestra el gráfico
    plt.show()

    plt.savefig(f'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/validacion_{tema[i]}_tipo_emocion.png', transparent=True)

#%%
presecial_tipo_emocion_aparte = {'Positivas': 48, 'Neutras': 15, 'Negativas':6}

datos_emociones = list(presecial_tipo_emocion_aparte.values())
etiquetas_emociones = list(presecial_tipo_emocion_aparte.keys())
plt.figure(3), plt.clf()
plt.pie(datos_emociones, labels=etiquetas_emociones, autopct='%1.1f%%', colors=colores, startangle=140, textprops={'fontsize': 14})

plt.show()

plt.savefig('C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/validacion_presecial_tipo_emocion.png', transparent=True)

#%%

plt.figure(1), plt.clf()
plt.hist(edades, color = color_celeste, edgecolor='k')  # Puedes ajustar el número de 'bins' según tus preferencias

# Agregar etiquetas y título
plt.xlabel('Edades', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostrar el histograma
plt.show()

plt.savefig('C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/validacion_edad.png', transparent=True)

#%%

# Calcular la cantidad de cada género
conteo_generos = genero.value_counts()

# Personaliza los colores para los géneros
colores = [color_celeste, color_celestito, color_palido, 'purple', 'orange']  # Agrega más colores si es necesario

# Crea el gráfico de torta
plt.figure(2), plt.clf()
plt.pie(conteo_generos, labels = conteo_generos.index, colors=colores, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})

#plt.legend(conteo_generos.index, title="Géneros", fontsize=12)

# Muestra el gráfico
plt.show()

plt.savefig('C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/validacion_genero.png', transparent=True)

