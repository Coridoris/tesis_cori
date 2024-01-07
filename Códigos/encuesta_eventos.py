# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:43:35 2023

@author: corir
"""

'''
aca vamos a hacer un analisis de las respuestas de la encuesta para elegir los 4 eventos mas relevantes del 2022
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

filename = 'C:/Users/Usuario/Desktop/Cori/Tesis/Encuestas/Encuesta para elegir eventos/Encuesta sobre eventos del 2022 (Respuestas) - Respuestas de formulario 1.csv'


# Carga del dataset
df = pd.read_csv(filename)

#print(df['Edad']<36) # esto me da una columna de verdaderos y falsos, es verdadero si al edad es menor igual a 35

df_edad35 = df[df['Edad']<36] #aca me quedo con el dataframe solo de menores a 35
df_edad35 = df

#%%
generos = df_edad35['Género'].value_counts()

dic_generos = {1 : 'Femenino', 2 : 'Masculino', 3 : 'Sin decir'}#, 4 : 'No binario', 5 : 'Otro'} # Esto lo saco de uno de los diccionarios que indican qué significa cada número


plt.figure(1), plt.clf()
plt.pie(generos.values / generos.values.sum(),
           colors = plt.get_cmap('Set2').colors, autopct='%1.1f%%')
plt.title('Género', size= 20)
plt.legend([dic_generos[g][:] for g in [1,2,3]], loc = (0,0.4), bbox_to_anchor=(1, 0, 0.5, 1))
#%%
def grafico_torta(nro_figura, columna, valores_respuesta = None, titulo = None, loc = None):
    columna_value_counts = columna.value_counts()
    if valores_respuesta == None:
        valores_respuesta = columna_value_counts.keys()
    if loc == None:
        loc = "best"
    if len(valores_respuesta)>10:
        autopct =  None
        color =  plt.cm.nipy_spectral(np.linspace(.1, .9,len(valores_respuesta)))
    if len(valores_respuesta)<10:
        autopct = '%1.1f%%'
        color = plt.get_cmap('Set2').colors
    plt.figure(nro_figura, figsize=(15,15)), plt.clf()        
    plt.pie(columna_value_counts.values / columna_value_counts.values.sum(),
               colors = color, autopct=autopct)
    plt.legend(valores_respuesta, loc = loc, bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(titulo, size= 20)
    return nro_figura

# def grafico_torta_diez(nro_figura, columna1, titulo1 = None, columna2, titulo2 = None, columna3, titulo3 = None, columna4, titulo4 = None, columna5, titulo5 = None, columna6, titulo6 = None, columna7, titulo7 = None, columna8, titulo8 = None, columna9, titulo9 = None, columna10, titulo10 = None):
#     columna_value_counts1 = columna1.value_counts()
#     valores_respuesta1 = columna_value_counts1.keys()
#     columna_value_counts2 = columna2.value_counts()
#     valores_respuesta2 = columna_value_counts2.keys()
#     columna_value_counts3 = columna3.value_counts()
#     valores_respuesta3 = columna_value_counts3.keys()
#     columna_value_counts4 = columna4.value_counts()
#     valores_respuesta4 = columna_value_counts4.keys()
#     columna_value_counts5 = columna5.value_counts()
#     valores_respuesta5 = columna_value_counts5.keys()
#     columna_value_counts6 = columna6.value_counts()
#     valores_respuesta6 = columna_value_counts6.keys()
#     columna_value_counts7 = columna7.value_counts()
#     valores_respuesta7 = columna_value_counts7.keys()
#     columna_value_counts8 = columna8.value_counts()
#     valores_respuesta8 = columna_value_counts8.keys()
#     columna_value_counts9 = columna9.value_counts()
#     valores_respuesta9 = columna_value_counts9.keys()
#     columna_value_counts10 = columna10.value_counts()
#     valores_respuesta10 = columna_value_counts10.keys()
#     columna = [[columna_value_counts1, columna_value_counts2, columna_value_counts3, columna_value_counts4, columna_value_counts5], [columna_value_counts6, columna_value_counts7, columna_value_counts8, columna_value_counts9, columna_value_counts10]]
#     autopct = '%1.1f%%'
#     color = plt.get_cmap('Set2').colors
#     #plt.figure(nro_figura, figsize=(15,15)), plt.clf()       
#     fig, ax = plt.subplots((2,5))
#     for i in range(2):
#         for j in range(5):
#             ax[i][j].pie(columna[i][j].values / columna[i][j].values.sum(), colors = color, autopct=autopct)
#             ax[legend]
#     plt.pie(columna_value_counts.values / columna_value_counts.values.sum(),
#                colors = color, autopct=autopct)
#     plt.legend(valores_respuesta, loc = loc, bbox_to_anchor=(1, 0, 0.5, 1))
#     plt.title(titulo, size= 20)
#     return nro_figura



def grafico_barras(nro_figura, columna, valores_respuesta =  None, titulo = None):
    columna_value_counts = columna.value_counts()
    if valores_respuesta == None:
        valores_respuesta = columna_value_counts.keys()
    plt.figure(nro_figura), plt.clf()
    
    plt.grid('on', linestyle = 'dashed', alpha = 0.5)
    plt.title(titulo)
    plt.ylabel('Total')
    plt.bar(x = [i for i in range(len(columna_value_counts))], # Definimos la ubicación de las barras a lo largo del eje horizontal
           height = columna_value_counts.values, # Definimos la altura de las barras
           color = plt.get_cmap('Set2').colors
           )
    plt.xticks(ticks = [i for i in range(len(columna_value_counts))], labels = valores_respuesta,
                       rotation = 90
                       )
    plt.tick_params(axis='both', which='major', labelsize = 12)
    return nro_figura


def tres_graficos_barras(nro_figura, columna1, columna2, columna3, titulo = None):
    columna1_value_counts = columna1.value_counts()
    valores_respuesta1 = columna1_value_counts.keys()
    columna2_value_counts = columna2.value_counts()
    valores_respuesta2 = columna2_value_counts.keys()
    columna3_value_counts = columna3.value_counts()
    valores_respuesta3 = columna3_value_counts.keys()
    columna_value_counts = [columna1_value_counts, columna2_value_counts, columna3_value_counts]
    valores_respuesta = [valores_respuesta1, valores_respuesta2, valores_respuesta3]
    fig, ax = plt.subplots(3)
    eje_y= ['Tot, ¿cuánto recordas?', 'Total', 'Intensidad sent, tot']
    fig.suptitle(titulo)
    for j in [0,1,2]:
        ax[j].grid('on', linestyle = 'dashed', alpha = 0.5)
        ax[j].set_ylabel(eje_y[j])
        ax[j].bar(x = [i for i in range(len(columna_value_counts[j]))], # Definimos la ubicación de las barras a lo largo del eje horizontal
               height = columna_value_counts[j].values, # Definimos la altura de las barras
               color = plt.get_cmap('Set2').colors
               )
        ax[j].set_xticks(ticks = [i for i in range(len(columna_value_counts[j]))], labels = valores_respuesta[j],
                           rotation = 0
                           )
       # ax[j].set_tick_params(axis='both', which='major', labelsize = 12)
    return nro_figura
#%% Cuanto recordas

evento = ['Fr vs Arg', 'CFK atentado', 'AS vs Arg', 'Coldplay', 'Reina', 'Guerra', 'Guzman', 'Censo',  'Presencial', 'Condena CFK']

recuerdo = [100, 92.9, 89.3, 85.7, 77.4, 76.2, 70.2, 64.3, 64.3, 61.9]

plt.figure(1), plt.clf()
    
plt.grid('on', linestyle = 'dashed', alpha = 0.5)
plt.title('¿Qué recuerdan sub 35?', fontsize = 20)
plt.ylabel('Total', fontsize = 15)
plt.bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = recuerdo, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )
plt.xticks(ticks = [i for i in range(len(evento))], labels = evento,
                   rotation = 0
                   )
plt.tick_params(axis='both', which='major', labelsize = 12)
#%% Cuánto recordas?

rec_AS = df_edad35['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'].mean()
rec_Fr = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'].mean()
rec_CFK_at = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'].mean()
rec_guerra = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'].mean()
rec_censo = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'].mean()
rec_reina = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'].mean()
rec_coldplay = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'].mean()
rec_guzman = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'].mean()
rec_presencial = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'].mean()
rec_CFKcondena = df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'].mean()
cuanto_rec = [rec_Fr, rec_CFK_at, rec_AS, rec_coldplay, rec_reina, rec_guerra, rec_guzman, rec_censo, rec_presencial, rec_CFKcondena]

plt.figure(2), plt.clf()
    
plt.grid('on', linestyle = 'dashed', alpha = 0.5)
plt.title('¿Cuánto recuerdan sub 35?', fontsize = 20)
plt.ylabel('Total', fontsize = 17)
plt.bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanto_rec, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )
plt.xticks(ticks = [i for i in range(len(evento))], labels = evento,
                   rotation = 0
                   )
plt.tick_params(axis='both', which='major', labelsize = 12)

#%% Intensidad emocion
int_AS = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'].mean()
int_Fr = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'].mean()
int_CFK_at = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'].mean()
int_guerra = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'].mean()
int_censo = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'].mean()
int_reina = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'].mean()
int_coldplay = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'].mean()
int_guzman = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'].mean()
int_presencial = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'].mean()
int_CFKcondena = df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'].mean()

cuanta_int = [int_Fr, int_CFK_at, int_AS, int_coldplay, int_reina, int_guerra, int_guzman, int_censo, int_presencial, int_CFKcondena]

plt.figure(3), plt.clf()
    
plt.grid('on', linestyle = 'dashed', alpha = 0.5)
plt.title('Intensidad emocion recuerdo sub 35', fontsize = 20)
plt.ylabel('Total', fontsize = 17)
plt.bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanta_int, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )
plt.xticks(ticks = [i for i in range(len(evento))], labels = evento,
                   rotation = 0
                   )
plt.tick_params(axis='both', which='major', labelsize = 12)
#%%
fig, ax = plt.subplots(4)
eje_y= ['Tot', 'Tot', 'Tot','Tot']
fig.suptitle('Sub 35')
for j in [0,1,2,3]:
    ax[j].grid('on', linestyle = 'dashed', alpha = 0.5)
    ax[j].set_ylabel(eje_y[j], fontsize = 10)
    ax[j].set_xticks(ticks = [i for i in range(len(evento))], labels = evento, rotation = 0)
    ax[j].tick_params(axis='both', which='major', labelsize = 12)

ax[0].set_title('¿Qué recuerdan?', fontsize = 15)
ax[0].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = recuerdo, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

ax[1].set_title('¿Cuánto recuerdan?', fontsize = 15)
ax[1].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanto_rec, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

ax[2].set_title('Intensidad emocion recuerdo', fontsize = 15)
ax[2].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanta_int, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

# definir los parámetros del gráfico
N = len(evento)
barWidth = .5
xloc = np. arange (N)

#create gráfico de barras apiladas
p1 = ax[3].bar(xloc, positivas, width = barWidth)
p2 = ax[3].bar(xloc, neutras, bottom = positivas, width = barWidth)
p3 = ax[3].bar(xloc, negativas, bottom = neutras+positivas, width = barWidth)

#add etiquetas, título, marcas de verificación y plt de leyenda

ax[3].title('Tipo de sentimiento', fontsize = 15)
#plt.xticks(xloc, evento)
#plt.yticks(np. arange (0, 80, 20))
#plt.ylim((-0.2,82))
ax[3].legend((p1[0], p2[0], p3[0]), ('positivas', 'neutras', 'negativas'))

#%% que emocion?
evento = ['Fr vs Arg', 'CFK atentado', 'AS vs Arg', 'Coldplay', 'Reina', 'Guerra', 'Guzman', 'Censo',  'Presencial', 'Condena CFK']
plt.figure(4), plt.clf()
#con la gente q respondio mal
#positivas = [78, 4, 12, 23, 4, 0, 4, 12, 39, 20]
#neutras = [8, 45, 34, 49, 57, 20, 5, 44, 14, 20]
#negativas = [3, 30, 36, 5, 7, 49, 19, 3, 6, 15 ]
#sin la gente que respondio mal
positivas = np.array([73, 4, 9, 19, 4, 0, 3, 7, 35, 19 ])
neutras = np.array([6, 44, 29, 45, 54, 15, 3, 40, 11, 17])
negativas = np.array([0, 29, 30, 4, 4, 44, 16, 2, 4, 13])
suma = positivas + neutras + negativas
positivas = positivas/suma
neutras = neutras/suma
negativas = negativas/suma

# definir los parámetros del gráfico
N = len(evento)
barWidth = .5
xloc = np. arange (N)

#create gráfico de barras apiladas
p1 = plt.bar(xloc, positivas, width = barWidth)
p2 = plt.bar(xloc, neutras, bottom = positivas, width = barWidth)
p3 = plt.bar(xloc, negativas, bottom = neutras+positivas, width = barWidth)

#add etiquetas, título, marcas de verificación y plt de leyenda
plt.ylabel('Total', fontsize = 17)
plt.title('Tipo de sentimiento', fontsize = 20)
plt.xticks(xloc, evento)
#plt.yticks(np. arange (0, 80, 20))
plt.ylim((-0.2,1.2))
plt.grid('on', linestyle = 'dashed', alpha = 0.5)
plt.legend((p1[0], p2[0], p3[0]), ('positivas', 'neutras', 'negativas'))

#display chart
plt.show()


#%%
# todo junto

fig, ax = plt.subplots(4)
eje_y= ['Tot, Tot', 'Tot','Tot']
fig.suptitle('Sub 35')
for j in [0,1,2,3]:
    ax[j].grid('on', linestyle = 'dashed', alpha = 0.5)
    ax[j].set_ylabel(eje_y[j], fontsize = 10)
    ax[j].set_xticks(ticks = [i for i in range(len(evento))], labels = evento, rotation = 0)
    ax[j].tick_params(axis='both', which='major', labelsize = 12)

ax[0].set_title('¿Qué recuerdan?', fontsize = 15)
ax[0].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = recuerdo, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

ax[1].settitle('¿Cuánto recuerdan?', fontsize = 15)
ax[1].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanto_rec, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

ax[2].set_title('Intensidad emocion recuerdo', fontsize = 15)
ax[2].bar(x = [i for i in range(len(evento))], # Definimos la ubicación de las barras a lo largo del eje horizontal
        height = cuanta_int, # Definimos la altura de las barras
        color = plt.get_cmap('Set2').colors
        )

# definir los parámetros del gráfico
N = len(evento)
barWidth = .5
xloc = np. arange (N)

#create gráfico de barras apiladas
p1 = ax[3].bar(xloc, positivas, width = barWidth)
p2 = ax[3].bar(xloc, neutras, bottom = positivas, width = barWidth)
p3 = ax[3].bar(xloc, negativas, bottom = neutras+positivas, width = barWidth)

#add etiquetas, título, marcas de verificación y plt de leyenda

ax[3].title('Tipo de sentimiento', fontsize = 15)
#plt.xticks(xloc, evento)
#plt.yticks(np. arange (0, 80, 20))
#plt.ylim((-0.2,82))
ax[3].legend((p1[0], p2[0], p3[0]), ('positivas', 'neutras', 'negativas'))


#%%graficos torta todo
grafico_torta(1, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?'], titulo = 'AS vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_torta(3, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.1'], titulo = 'Fr vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_torta(5, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.2'], titulo = 'Atentado CFK, te acordas? sub 35', loc = (0,0.4))

grafico_torta(7, df_edad35['¿Te acordas cuando sucedió o te enteraste este evento?'], titulo = 'Guerra, te acordas? sub 35', loc = (0,0.4))

grafico_torta(9, df_edad35['¿Recordas el día del censo y/o el momento en el que lo completaste digitalmente?'], titulo = 'Censo, te acordas? sub 35', loc = (0,0.4))

grafico_torta(11, df_edad35['¿Recordas cuando sucedió o te enteraste el evento?'], titulo = 'Muerte Reina, te acordas? sub 35', loc = (0,0.4))

grafico_torta(13, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?'], titulo = 'Coldplay, te acordas? sub 35', loc = (0,0.4))

grafico_torta(15, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?.1'], titulo = 'Guzman, te acordas? sub 35', loc = (0,0.4))

grafico_torta(17, df_edad35['¿Tuviste la vuelta a las clases presenciales en el 2022?'], titulo = 'Clase presencial, te acordas? sub 35', loc = (0,0.4))

evento = 'Condena CFK'

grafico_torta(19, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?.2'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))
#%% emotividad graficos barra todo
tres_graficos_barras(2, df_edad35['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'], titulo = 'AS. vs Arg')

tres_graficos_barras(4, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.1'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'], titulo = 'Fr vs Arg')

tres_graficos_barras(6, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.2'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'], titulo = 'Atentado CFK')

tres_graficos_barras(8, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.5'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'], titulo = 'Guerra')

tres_graficos_barras(10, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.6'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'], titulo = 'Censo')

tres_graficos_barras(12, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.7'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'], titulo = 'Muerte Reina')

tres_graficos_barras(14, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.8'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'], titulo = 'Coldplay')

tres_graficos_barras(16, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.9'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'], titulo = 'Guzman')

tres_graficos_barras(18, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.10'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'], titulo = 'Clase Presencial')

evento = 'Condena CFK'

tres_graficos_barras(20, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.18'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'], titulo = 'Condena CFK')
#%% Género y edad
grafico_torta(1, df_edad35['Género'], ['F', 'M', 'X'], titulo ='Generos sub 35', loc = (0,0.4))

grafico_torta(2, df['Género'], ['F', 'M', 'X'], 'Generos total', loc = (0,0.4))

grafico_torta(3, df_edad35['Edad'], titulo = 'Edades sub 35')

grafico_torta(4, df['Edad'], titulo = 'Edades total')
#%%
plt.figure(6), plt.clf()
plt.hist(df_edad35['Edad'])

#%% AS vs Arg
grafico_torta(5, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?'], titulo = 'AS vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Te acordas cuando sucedió o te enteraste del evento?'], titulo = 'AS vs Arg, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'AS vs Arg, cuánto te acordas? sub35')
 

grafico_barras(8, df['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'AS vs Arg, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?'], valores_respuesta = ['Neg', 'Neu', 'Pos', 'Neg, neu', 'neg, pos', 'neu, pos'], titulo = 'AS vs Arg, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?'], valores_respuesta = ['Neg', 'Neu', 'Pos', 'Neg, neu', 'neg, pos', 'neu, pos'], titulo = 'AS vs Arg, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'], titulo = 'AS vs Arg, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'], titulo = 'AS vs Arg, intensidad emoción total')

#%% Fr vs Arg

grafico_torta(5, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.1'], titulo = 'Fr vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Te acordas cuando sucedió o te enteraste del evento?.1'], titulo = 'Fr vs Arg, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'Fr vs Arg, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'Fr vs Arg, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.1'],valores_respuesta = ['Pos', 'Neu', 'Neg, Pos', 'Neu, Pos'], titulo = 'Fr vs Arg, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.1'], valores_respuesta = ['Pos', 'Neu', 'Neg, Pos', 'Neu, Pos'], titulo = 'Fr vs Arg, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'], titulo = 'Fr vs Arg, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'], titulo = 'Fr vs Arg, intensidad emoción total')

#%% Atentado CFK

grafico_torta(5, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.2'], titulo = 'Atentado CFK, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Te acordas cuando sucedió o te enteraste del evento?.2'], titulo = 'Atentado CFK, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'], titulo = 'Atentado CFK, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'], titulo = 'Atentado CFK, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.2'],valores_respuesta = ['Neu', 'Neg', 'Pos', 'Neg, Neu'], titulo = 'Atentado CFK, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.2'], valores_respuesta = ['Neu', 'Neg', 'Pos', 'Neg, Neu'], titulo = 'Atentado CFK, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'], titulo = 'Atentado CFK, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'], titulo = 'Atentado CFK, intensidad emoción total')

#%% GH


grafico_torta(5, df_edad35['¿Viste el comienzo?'], titulo = 'GH, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Viste el comienzo?'], titulo = 'GH, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..2'], titulo = 'GH, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..2'], titulo = 'GH, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.3'],valores_respuesta = ['Neu', 'Pos'], titulo = 'GH, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.3'], valores_respuesta = ['Neu', 'Pos'], titulo = 'GH, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..3'], titulo = 'GH, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..3'], titulo = 'GH, intensidad emoción total')

#%% Arg 1985


grafico_torta(5, df_edad35['¿Viste la pelicula?'], titulo = 'Arg 1985, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Viste la pelicula?'], titulo = 'Arg 1985, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando viste la pelicula? Donde 0 es nada y 5 es mucho.'], titulo = 'Arg 1985, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando viste la pelicula? Donde 0 es nada y 5 es mucho.'], titulo = 'Arg 1985, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.4'],valores_respuesta = ['pos', 'neu', 'neg', 'neg, pos', 'neg,neu,pos'], titulo = 'Arg 1985, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.4'], valores_respuesta = ['pos', 'neu', 'neg', 'neg, pos, neu', 'neg,pos'], titulo = 'Arg 1985, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..4'], titulo = 'Arg 1985, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..4'], titulo = 'Arg 1985, intensidad emoción total')


#%% Guerra

grafico_torta(5, df_edad35['¿Te acordas cuando sucedió o te enteraste este evento?'], titulo = 'Guerra, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Te acordas cuando sucedió o te enteraste este evento?'], titulo = 'Guerra, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'], titulo = 'Guerra, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'], titulo = 'Guerra, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.5'],valores_respuesta = ['neg', 'neu', 'neg, neu'], titulo = 'Guerra, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.5'], valores_respuesta = ['neg', 'neu', 'neg, neu'], titulo = 'Guerra, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'], titulo = 'Guerra, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'], titulo = 'Guerra, intensidad emoción total')


#%% Censo

grafico_torta(5, df_edad35['¿Recordas el día del censo y/o el momento en el que lo completaste digitalmente?'], titulo = 'Censo, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas el día del censo y/o el momento en el que lo completaste digitalmente?'], titulo = 'Censo, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'], titulo = 'Censo, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'], titulo = 'Censo, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.6'],valores_respuesta = ['neu', 'pos', 'neu, pos', 'neg', 'neg, pos'], titulo = 'Censo, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.6'], valores_respuesta = ['neu', 'pos', 'neg', 'neu, pos', 'neg, pos', 'neu, neg, pos'], titulo = 'Censo, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'], titulo = 'Censo, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'], titulo = 'Censo, intensidad emoción total')


#%% Muerte Reina

grafico_torta(5, df_edad35['¿Recordas cuando sucedió o te enteraste el evento?'], titulo = 'Muerte Reina, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas cuando sucedió o te enteraste el evento?'], titulo = 'Muerte Reina, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'], titulo = 'Muerte Reina, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'], titulo = 'Muerte Reina, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.7'],valores_respuesta = None, titulo = 'Muerte Reina, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.7'], valores_respuesta = None, titulo = 'Muerte Reina, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'], titulo = 'Muerte Reina, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'], titulo = 'Muerte Reina, intensidad emoción total')

#%% Coldplay

grafico_torta(5, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?'], titulo = 'Coldplay, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas cuando sucedió o te enteraste de este evento?'], titulo = 'Coldplay, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'], titulo = 'Coldplay, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'], titulo = 'Coldplay, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.8'],valores_respuesta = None, titulo = 'Coldplay, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.8'], valores_respuesta = None, titulo = 'Coldplay, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'], titulo = 'Coldplay, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'], titulo = 'Coldplay, intensidad emoción total')

#%% Guzman

grafico_torta(5, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?.1'], titulo = 'Guzman, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas cuando sucedió o te enteraste de este evento?.1'], titulo = 'Guzman, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'], titulo = 'Guzman, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'], titulo = 'Guzman, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.9'],valores_respuesta = None, titulo = 'Guzman, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.9'], valores_respuesta = None, titulo = 'Guzman, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'], titulo = 'Guzman, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'], titulo = 'Guzman, intensidad emoción total')

#%% Clase presencial

grafico_torta(5, df_edad35['¿Tuviste la vuelta a las clases presenciales en el 2022?'], titulo = 'Clase presencial, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Tuviste la vuelta a las clases presenciales en el 2022?'], titulo = 'Clase presencial, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'], titulo = 'Clase presencial, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'], titulo = 'Clase presencial, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.10'],valores_respuesta = None, titulo = 'Clase presencial, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.10'], valores_respuesta = None, titulo = 'Clase presencial, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'], titulo = 'Clase presencial, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'], titulo = 'Clase presencial, intensidad emoción total')


#%% Dojacat

grafico_torta(5, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?.2'], titulo = 'Dojacat, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas cuando sucedió o te enteraste de este evento?.2'], titulo = 'Dojacat, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..9'], titulo = 'Dojacat, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..9'], titulo = 'Dojacat, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.11'],valores_respuesta = None, titulo = 'Dojacat, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.11'], valores_respuesta = None, titulo = 'Dojacat, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..11'], titulo = 'Dojacat, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..11'], titulo = 'Dojacat, intensidad emoción total')


#%% Bizarrap

grafico_torta(5, df_edad35['¿Recordas cuando escuchaste la canción?'], titulo = 'Bizarrap, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recordas cuando escuchaste la canción?'], titulo = 'Bizarrap, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..10'], titulo = 'Bizarrap, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..10'], titulo = 'Bizarrap, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.12'],valores_respuesta = None, titulo = 'Bizarrap, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.12'], valores_respuesta = None, titulo = 'Bizarrap, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..12'], titulo = 'Bizarrap, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..12'], titulo = 'Bizarrap, intensidad emoción total')

#%% DrStrange

grafico_torta(5, df_edad35['¿Viste la pelicula?.1'], titulo = 'DrStrange, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Viste la pelicula?.1'], titulo = 'DrStrange, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..11'], titulo = 'DrStrange, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..11'], titulo = 'DrStrange, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.13'],valores_respuesta = None, titulo = 'DrStrange, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.13'], valores_respuesta = None, titulo = 'DrStrange, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..13'], titulo = 'DrStrange, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..13'], titulo = 'DrStrange, intensidad emoción total')


#%% Thor

evento = 'Thor'

grafico_torta(5, df_edad35['¿Viste la pelicula?.2'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Viste la pelicula?.2'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..12'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..12'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.14'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.14'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..14'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..14'], titulo = f'{evento}, intensidad emoción total')

#%% Black Panther

evento = 'Black Panther'

grafico_torta(5, df_edad35['¿Viste la pelicula?.3'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Viste la pelicula?.3'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..13'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..13'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.15'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.15'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..15'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..15'], titulo = f'{evento}, intensidad emoción total')

#%% Pele

evento = 'Pele'

grafico_torta(5, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recuerdas cuando sucedió o te enteraste del evento?'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..14'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..14'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.16'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.16'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..16'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..16'], titulo = f'{evento}, intensidad emoción total')

#%% Lula

evento = 'Lula'

grafico_torta(5, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?.1'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recuerdas cuando sucedió o te enteraste del evento?.1'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..15'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..15'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.17'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.17'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..17'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..17'], titulo = f'{evento}, intensidad emoción total')

#%% Condena CFK

evento = 'Condena CFK'

grafico_torta(5, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?.2'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recuerdas cuando sucedió o te enteraste del evento?.2'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.18'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.18'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'], titulo = f'{evento}, intensidad emoción total')

#%% Elon compra Tw

evento = 'Elon compra Tw'

grafico_torta(5, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?.3'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_torta(6, df['¿Recuerdas cuando sucedió o te enteraste del evento?.3'], titulo = f'{evento}, te acordas? total', loc = (0,0.4))

grafico_barras(7, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..17'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(8, df['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..17'], titulo = f'{evento}, cuánto te acordas? total')

grafico_barras(9, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.19'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')
 
grafico_barras(10, df['¿Qué tipo de emociones te genera el recuerdo?.19'], valores_respuesta = None, titulo = f'{evento}, emoción total')


grafico_barras(11, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..19'], titulo = f'{evento}, intensidad emoción sub35')
 
grafico_barras(12, df['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..19'], titulo = f'{evento}, intensidad emoción total')


#%%

Arabia_Saudista = df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?'].value_counts()

Campeones_mundo = df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.1'].value_counts()

Atentado_CFK = df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?'].value_counts()

#GH = df_edad35['¿Viste el comienzo?'].value_counts()

#Arg_1985 = df_edad35['¿Viste la pelicula?'].value_counts()

Guerra = df_edad35['¿Te acordas cuando sucedió o te enteraste este evento?'].value_counts()


#%% AS vs Arg
grafico_torta(1, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?'], titulo = 'AS vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'AS vs Arg, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?'], valores_respuesta = ['Neg', 'Neu', 'Pos', 'Neg, neu', 'neg, pos', 'neu, pos'], titulo = 'AS vs Arg, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'], titulo = 'AS vs Arg, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.'], titulo = 'AS. vs Arg')

#%% Fr vs Arg

grafico_torta(1, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.1'], titulo = 'Fr vs Arg, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], titulo = 'Fr vs Arg, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.1'],valores_respuesta = ['Pos', 'Neu', 'Neg, Pos', 'Neu, Pos'], titulo = 'Fr vs Arg, emoción sub35')
 
grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'], titulo = 'Fr vs Arg, intensidad emoción sub35')
 
tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.1'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1'], titulo = 'Fr vs Arg')

#%% Atentado CFK

grafico_torta(1, df_edad35['¿Te acordas cuando sucedió o te enteraste del evento?.2'], titulo = 'Atentado CFK, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'], titulo = 'Atentado CFK, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.2'],valores_respuesta = ['Neu', 'Neg', 'Pos', 'Neg, Neu'], titulo = 'Atentado CFK, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'], titulo = 'Atentado CFK, intensidad emoción sub35')
 
tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.2'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2'], titulo = 'Atentado CFK')

#%% Guerra

grafico_torta(1, df_edad35['¿Te acordas cuando sucedió o te enteraste este evento?'], titulo = 'Guerra, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'], titulo = 'Guerra, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.5'],valores_respuesta = ['neg', 'neu', 'neg, neu'], titulo = 'Guerra, emoción sub35')
 
grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'], titulo = 'Guerra, intensidad emoción sub35')
 
tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.5'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..5'], titulo = 'Guerra')

#%% Censo

grafico_torta(1, df_edad35['¿Recordas el día del censo y/o el momento en el que lo completaste digitalmente?'], titulo = 'Censo, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'], titulo = 'Censo, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.6'],valores_respuesta = ['neu', 'pos', 'neu, pos', 'neg', 'neg, pos'], titulo = 'Censo, emoción sub35')
 
grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'], titulo = 'Censo, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.6'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..6'], titulo = 'Censo')
 

#%% Muerte Reina

grafico_torta(1, df_edad35['¿Recordas cuando sucedió o te enteraste el evento?'], titulo = 'Muerte Reina, te acordas? sub 35', loc = (0,0.4))

#grafico_torta(2, df['¿Recordas cuando sucedió o te enteraste el evento?'], titulo = 'Muerte Reina, te acordas? total', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'], titulo = 'Muerte Reina, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.7'],valores_respuesta = None, titulo = 'Muerte Reina, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'], titulo = 'Muerte Reina, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.7'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..7'], titulo = 'Muerte Reina')


#%% Coldplay

grafico_torta(1, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?'], titulo = 'Coldplay, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'], titulo = 'Coldplay, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.8'],valores_respuesta = None, titulo = 'Coldplay, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'], titulo = 'Coldplay, intensidad emoción sub35')
 
tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.8'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..8'], titulo = 'Coldplay')

#%% Guzman

grafico_torta(1, df_edad35['¿Recordas cuando sucedió o te enteraste de este evento?.1'], titulo = 'Guzman, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'], titulo = 'Guzman, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.9'],valores_respuesta = None, titulo = 'Guzman, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'], titulo = 'Guzman, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..7'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.9'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..9'], titulo = 'Guzman')

#%% Clase presencial

grafico_torta(1, df_edad35['¿Tuviste la vuelta a las clases presenciales en el 2022?'], titulo = 'Clase presencial, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'], titulo = 'Clase presencial, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.10'],valores_respuesta = None, titulo = 'Clase presencial, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'], titulo = 'Clase presencial, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.10'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10'], titulo = 'Clase Presencial')

#%% Condena CFK

evento = 'Condena CFK'

grafico_torta(1, df_edad35['¿Recuerdas cuando sucedió o te enteraste del evento?.2'], titulo = f'{evento}, te acordas? sub 35', loc = (0,0.4))

grafico_barras(2, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'], titulo = f'{evento}, cuánto te acordas? sub35')
 
grafico_barras(3, df_edad35['¿Qué tipo de emociones te genera el recuerdo?.18'],valores_respuesta = None, titulo = f'{evento}, emoción sub35')

grafico_barras(4, df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'], titulo = f'{evento}, intensidad emoción sub35')

tres_graficos_barras(5, df_edad35['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..16'], df_edad35['¿Qué tipo de emociones te genera el recuerdo?.18'], df_edad35['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..18'], titulo = 'Condena CFK')

