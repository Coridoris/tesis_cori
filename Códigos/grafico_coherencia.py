# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 02:18:03 2023

@author: corir
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario'

def plot_two_histograms(data1, data2, titulo):
    # Crear una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Crear los histogramas en cada subgráfico
    ax1.hist(data1, edgecolor='black')
    ax2.hist(data2, edgecolor='black')


    # Etiquetas y título para cada subgráfico
    ax1.set_xlabel('Valores de coherencia')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Cada oración')
    ax1.legend()

    ax2.set_xlabel('Valores de coherencia')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Cada dos oraciones')
    ax2.legend()

    # Título general para los dos histogramas
    fig.suptitle(titulo, fontsize=16)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    
    #plt.savefig(path_imagenes + f'/{titulo}_hist_coherencia.png')

    # Mostrar los histogramas
    plt.show()
    
def plot_three_histograms(data1, data2, data3, titulo, title1, title2, title3, save, xlabel = None):
    # Crear una figura con dos subgráficos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # Crear los histogramas en cada subgráfico
    ax1.hist(data1, edgecolor='black')
    ax2.hist(data2, edgecolor='black')
    ax3.hist(data3, edgecolor='black')


    # Etiquetas y título para cada subgráfico
    ax1.set_xlabel('Num de sust')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title(title1)
    ax1.legend()

    ax2.set_xlabel('Num de verb')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(title2)
    ax2.legend()
    
    ax3.set_xlabel('Num de adj')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title(title3)
    ax3.legend()
    
    if xlabel != None:
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        ax3.set_xlabel(xlabel)        

    # Título general para los dos histogramas
    fig.suptitle(titulo, fontsize=16)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    
    #plt.savefig(path_imagenes + f'/{titulo}_hist_{save}.png')

    # Mostrar los histogramas
    plt.show()
    
def box_plot3(data1, ylabel, save, titulo):
    plt.figure()
    datos = {
        'Sust': data1[0],
        'Verb': data1[1],
        'Adj': data1[2]}   
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)
    
    # Título general para los dos histogramas
    plt.title(titulo, fontsize=16)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'
    

def box_plot(data1, ylabel, save):
    plt.figure()
    datos = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'Cfk': data1[2],
        'Arabia': data1[3]}   #Filler': data1[1]
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'

def box_plot5(data1, ylabel, save):
    plt.figure()
    datos = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'Cfk': data1[2],
        'Arabia': data1[3],
        'Filler': data1[4]}
    
    
    df = pd.DataFrame.from_dict(datos)

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df)
    
    plt.ylabel(ylabel)

    # Mostrar el gráfico
    plt.show()
    
    #plt.savefig(path_imagenes + f'/{save}_boxplot.png')
    
    return 'ok'
    
#%%

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antes_de_venir"]


for i in range(len(temas)):
    tema = temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia.csv'
            
    df_coherence = pd.read_csv(path)
    
    coherence1 = df_coherence['coherencia_cada_oracion']
    coherence2 = df_coherence['coherencia_cada_dos_oraciones']
    
    plot_two_histograms(coherence1, coherence2, tema)
    
    
#%% un boxplot con los 5 y con los 4

coherence1 = []
coherence2 = []
for i in range(len(temas)):
    tema = temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Coherencia/{tema}_coherencia.csv'
            
    df_coherence = pd.read_csv(path)
    
    coherence1.append(df_coherence['coherencia_cada_oracion'])
    coherence2.append(df_coherence['coherencia_cada_dos_oraciones'])
    
box_plot5(coherence1, 'Coher. cada una oración', 'coherencia_cada_oracion_filler')

box_plot5(coherence2, 'Coher. cada dos oraciones', 'coherencia_cada_dos_oraciones_filler')


# un boxplot sin antes de venir

box_plot(coherence1, 'Coher. cada una oración', 'coherencia_cada_oracion')

box_plot(coherence2, 'Coher. cada dos oraciones', 'coherencia_cada_dos_oraciones')

#%% histograma del número de sust verb y adj


temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antes_de_venir"]


for i in range(len(temas)-1):
    tema = temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv'
            
    df_conteo = pd.read_csv(path)
    
    sust = df_conteo['nro. sust']
    verb = df_conteo['nro. verb']
    adj = df_conteo['nro. adj']
    
    plot_three_histograms(sust, verb, adj, tema, 'Sustantivos', 'Verbos', 'Adjetivos', 'SustVerbAdj')
    
#%% hago un boxplot por tema

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"]

for i in range(len(temas)):
    tema = temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv'
            
    df_conteo = pd.read_csv(path)
    
    sust = df_conteo['nro. sust']
    verb = df_conteo['nro. verb']
    adj = df_conteo['nro. adj']
    
    data = [sust, verb, adj]
    
    box_plot3(data, 'Cantidad palabras', f'conteoSustVerbAdj_{tema}', f'{tema}')
    

    
#%% hago un boxplot por sust adj y verb


sust = []
verb = []
adj = []

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"]

for i in range(len(temas)):
    tema = temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/SustVerbAdj/{tema}_contando_sust_verb_adj.csv'
            
    df_conteo = pd.read_csv(path)
    
    sust.append(df_conteo['nro. sust'])
    verb.append(df_conteo['nro. verb'])
    adj.append(df_conteo['nro. adj'])
    

# un boxplot sin antes de venir

box_plot(sust, 'Cantidad sust', 'conteo_sust')

box_plot(verb, 'Cantidad verb', 'conteo_verb')

box_plot(adj, 'Cantidad adj', 'conteo_adj')

#%% boxplot, histograma y uno de barras apiladas de sentimiento

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"]

py_pos = []
py_neg = []
py_int = []
py_neu = []

for i in range(len(temas)):
    tema = temas[i]
    
    path_sentimiento = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Analisis_sentimiento/pysentimiento/{tema}_pysentimiento.csv'
            
    df_sent = pd.read_csv(path_sentimiento)
    
    df_sent['pysent_curdo'] = df_sent['pysent_curdo'].apply(eval)
    
    pysent = df_sent['pysent_curdo']
    
    pos = df_sent['pysent_curdo'].apply(lambda x: x['POS'])
    neg = pysent.apply(lambda x: x['NEG'])
    intensity = pysent.apply(lambda x: x['POS']) + pysent.apply(lambda x: x['NEG'])
    
    py_pos.append(pos)
    
    py_neg.append(neg)
    
    py_neu.append(pysent.apply(lambda x: x['NEU']))
    
    py_int.append(intensity)
    
    #histograma
    plot_three_histograms(pos, neg, intensity, tema, 'Positivo', 'Negativo', 'Intensidad', 'pysent', 'Probabilidad')

    
#boxplots

box_plot(py_pos, 'Pysent pos', 'pypos')

box_plot(py_neg, 'Pysent neg', 'pyneg')

box_plot(py_int, 'Pysent intensity', 'pyint')
#%%
#un grafico de barras apiladas con el promedio 

temas_label = ["Camp", "Pres", "CFK", "Arabia"]
plt.figure()

grupo1 = np.mean(py_neu, axis = 1)
grupo2 = np.mean(py_neg, axis = 1)
grupo3 = np.mean(py_pos, axis = 1)

# Coordenadas para la posición de las barras
posiciones = np.arange(len(grupo1))

# Crear el gráfico de barras apiladas
plt.bar(posiciones, grupo1, label = 'Neutro')
plt.bar(posiciones, grupo2, bottom=grupo1, label = 'Negativo')
plt.bar(posiciones, grupo3, bottom=np.add(grupo1, grupo2), label = 'Positivo')

# Etiquetas y título
plt.xlabel('Temas', fontsize = 15)
plt.ylabel('Probabilidad promedio', fontsize = 15)
#plt.title('Gráfico de Barras Apiladas')
plt.xticks(posiciones, labels = temas_label, fontsize = 15)
plt.yticks(fontsize = 12)
# Agregar leyenda fuera de la imagen
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)

# Mostrar el gráfico
plt.show()

#plt.savefig(path_imagenes + '/pysent_barras_apiladas.png')
#%% gráfico con barras una al lado de otra

colores = ["#011638", "#364156", "#9FA0A4"]

grupo1_mean = np.mean(py_int, axis = 1)
grupo2_mean = np.mean(py_neg, axis = 1)
grupo3_mean = np.mean(py_pos, axis = 1)

grupo1_std = np.std(py_int, axis = 1)/np.sqrt(len(py_int[0]))
grupo2_std = np.std(py_neg, axis = 1)/np.sqrt(len(py_neu[0]))
grupo3_std = np.std(py_pos, axis = 1)/np.sqrt(len(py_neu[0]))

condicion = ("Campeones", "Presencial", "CFK", "Arabia")#, "Filler")
detalles = {
    'Intensidad': tuple(grupo1_mean),
    'Negativo': tuple(grupo2_mean),
    'Positivo': tuple(grupo3_mean)
}

x = np.arange(len(condicion))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (15,10))

for (attribute, measurement, std_dev, color) in zip(detalles.keys(), detalles.values(), [grupo1_std, grupo2_std, grupo3_std], colores):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=std_dev, capsize=5, color=color)
    multiplier += 1
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probabilidad', fontsize =25)
ax.set_xticks(x + width, condicion)
ax.tick_params(labelsize = 25)
ax.legend(loc='upper right', fontsize = 20)

plt.show()

plt.savefig(path_imagenes+'/pysentimiento_seminario', transparent=True)
#%% correlacion de todo con todo :)))))))))))

#variables que tengo: cantidad de palabras únicas, cantidad de palabras totales, pysent pos neg e intensidad, 
#cantidad de sust, ver y adj, coherencia cada una y cada dos oraciones, data post entrevista (autopercep de recuerdo
# y sentimiento)
