# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:44:09 2023

@author: Usuario
"""

#%% librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as npl
from scipy.stats import sem
import ast 


#%% el santo trial

entrevista = 'Primera'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]
condicion = ("CFK", "Campeones", "Filler", "Presencial", "Arabia")
temas_label = ["CFK", "Campeones", "Filler", "Presencial", "Arabia"]

#%%colores y cosas imagenes
path_imagenes = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Graficos'

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

color_violeta = "#856084"

color_campeones = color_celeste
color_presencial = color_celestito
color_cfk = color_palido
color_arabia = color_violeta
color_filler = color_gris

color_violeta_fuerte = "#6A4480"

#%% funciones

def box_plot4o5(data1, cantidad, ylabel, save):
    
    npl.rcParams['figure.figsize'] = [10, 15]
    npl.rcParams["axes.labelsize"] = 20
    npl.rcParams['xtick.labelsize'] = 20
    npl.rcParams['ytick.labelsize'] = 20
    
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Presencial': data1[3],
            'Arabia': data1[4]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Filler': data1[2],
            'Presencial': data1[3],
            'Arabia': data1[4]}        
    
    df = pd.DataFrame.from_dict(datos)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]

    # Crear el boxplot utilizando seaborn
    sns.boxplot(data=df, palette = colors)
    
    plt.ylabel(ylabel)#, fontsize = 15)
    
    #plt.xticks(fontsize=12) 
    #plt.yticks(fontsize=12) 

    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = True)
    
    #plt.close()
    
    return 'ok'

def violin_plot4o5(data1, cantidad, ylabel, save):
    
    npl.rcParams['figure.figsize'] = [10, 15]
    npl.rcParams["axes.labelsize"] = 20
    npl.rcParams['xtick.labelsize'] = 20
    npl.rcParams['ytick.labelsize'] = 20
    
    sns.set(style="whitegrid")
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Presencial': data1[3],
            'Arabia': data1[4]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Filler': data1[2],
            'Presencial': data1[3],
            'Arabia': data1[4]}        
    
    df = pd.DataFrame.from_dict(datos)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]

    # Crear el boxplot utilizando seaborn
    sns.violinplot(data=df, palette=colors) #inner="quartile", cut=0
    
    plt.ylabel(ylabel)#, fontsize = 15)
    
    #plt.xticks(fontsize=12) 
    #plt.yticks(fontsize=12) 

    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(path_imagenes + f'/{save}_violinplot.png', transparent = True)
    
    #plt.close()
    
    return 'ok'

def scatter_plot4o5(data1, cantidad, ylabel, save):
    
    npl.rcParams['figure.figsize'] = [10, 15]
    npl.rcParams["axes.labelsize"] = 20
    npl.rcParams['xtick.labelsize'] = 20
    npl.rcParams['ytick.labelsize'] = 20
    
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Presencial': data1[3],
            'Arabia': data1[4]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Campeones': data1[1],
            'Filler': data1[2],
            'Presencial': data1[3],
            'Arabia': data1[4]}        
    
    df = pd.DataFrame.from_dict(datos)
    
    x = np.linspace(1, cantidad, cantidad)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]

    # Crear el scatter plot
    for i, condicion in enumerate(list(datos.keys())):
        plt.scatter([x[i]]*len(df[condicion]), df[condicion], label=condicion, color=colors[i])
    
    plt.ylabel(ylabel)
    plt.legend()
    
    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(path_imagenes + f'/{save}_scatterplot.png', transparent=True)
    
    return 'ok'


def ast_literal_eval_notnan(lista):
    if type(lista) == str:
        return ast.literal_eval(lista)
    else:
        return np.nan

#%% datos
df_del_tema = []

for tema in temas:    

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_conautopercepcion_{tema}.csv'
    
    df_del_tema.append(pd.read_csv(path))
    
#%% graficos de boxplot, violinplot o puntos

variables_label = list(df_del_tema[0].columns)
'''
['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop',
'Intensidad_autop', 'ValeInt_autop', 'num_palabras_unicas_norm',
'primera_persona_norm', 'tercera_persona_norm', 'num noun norm',
'num verb norm', 'num adj norm', 'num advs norm', 'num numeral norm',
'num propn norm', 'Positivo pysent', 'Negativo pysent',
'Intensidad pysent', 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3',
'num_nodes_norm', 'Comunidades_LSC', 'diámetro', 'k_mean',
'transitivity', 'ASP', 'average_CC', 'selfloops', 'L2', 'L3', 'density',
'Detalles internos norm', 'Detalles externos norm']
'''
variables = []
for j in range(len(variables_label)):
    variable = []
    for i, tema in enumerate(temas):
        variable.append(df_del_tema[i][variables_label[j]])
    variables.append(variable)
    
nro_p_cfk = nro_palabras_unicas_por_tema_para_normalizar[4] #hay que correr primeras celdas de variables contenido para tener esto definido
nro_p_cfk = np.where(np.array(nro_p_cfk) == 0, np.nan, nro_p_cfk)
#nro_p_cfk = variables[6][3]
print(len(nro_p_cfk))
sigma = np.std(nro_p_cfk)
mediana = np.nanmedian(nro_p_cfk)
nanindices = np.isnan(nro_p_cfk)
print(np.where(nanindices)[0])
#umbral = 3 * sigma
# Calcula la desviación absoluta de cada dato respecto a la mediana
desviaciones_absolutas = np.abs(nro_p_cfk - mediana)

# Calcula la mediana de las desviaciones absolutas para obtener la MAD
mad = np.nanmedian(desviaciones_absolutas)
umbral = 3*mad

datos_filtrados = [dato for dato in nro_p_cfk if abs(dato - mediana) <= umbral]
datos_filtrados = [dato if np.isnan(dato) or abs(dato - mediana) <= umbral else np.nan for dato in nro_p_cfk]
nanindices = np.isnan(datos_filtrados)
print(np.where(nanindices)[0])
    

#box_plot4o5(variables[6], 5, '$Z_{score}$ del núm de palabras', 'nro_palabras_unicas')
#violin_plot4o5(variables[6], 5, '$Z_{score}$ del núm de palabras', 'nro_palabras_unicas')