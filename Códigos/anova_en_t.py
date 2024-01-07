# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:46:47 2023

@author: corir
"""

import pandas as pd
import numpy as np

#para ANOVA
import pingouin as pg
from scipy.stats import f
#post ANOVA --> tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%% el santo trial y colores

entrevista = 'dos tiempos' #'Primera' 'Segunda' o 'dos tiempos'

no_autop = True #pone false si queres que las tenga en cuenta para el análisis

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

color_celeste = "#79b4b7ff"
color_gris = "#9fa0a4ff"

#%% path data 

if entrevista == 'dos tiempos':
    path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_dostiempos.csv'
    
    path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_dostiempos.csv'

#%% ANOVA

df = pd.read_csv(path_conautopercepcion_todas)

eliminamos_pysent = ['Valencia pysent','Valencia e intensidad pysent']

df = df.drop(eliminamos_pysent, axis=1)


mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3
}

df['Condición'] = df['Condición'].map(mapping)

condiciones_a_eliminar = [2,3,4,5]

valor_a_buscar = 1

for clave, valor in mapping.items():
    if valor == valor_a_buscar:
        condicion_que_evaluamos = clave
        break

# Filtramos las condiciones 
df = df[~df['Condición'].isin(condiciones_a_eliminar)]

variables_dependientes = list(df.columns)[3:] #sacamos sujeto condicion y tiempo
#%%
#aov = pg.rm_anova(dv = 'Nro palabras únicas', within = 'Condición', subject='Sujetos', data=df, detailed=True, effsize="np2")

# aov = df.rm_anova(dv = 'num_palabras_unicas_norm', within='Tiempo', subject='Sujetos',  detailed=False)

# # Definir los grados de libertad del numerador y del denominador
# df_between = aov['ddof1'][0]  # Grados de libertad del numerador
# df_within = aov['ddof2'][0]   # Grados de libertad del denominador

# # Definir el nivel de significancia (alfa)
# alfa = 0.05

# # Calcular el valor crítico de F
# f_critical = f.ppf(1 - alfa, df_between, df_within)

# print(f"Valor crítico de F: {f_critical}")

# print(f"Valor de F: {aov['F'][0]}")

# print(f"Valor de p: {aov['p-unc'][0]}")

# print(f"Valor de epsilon: {aov['eps'][0]}")

#print(f"Valor de p corregido: {aov['p-GG-corr'][0]}"

#%% veo si alguna variable da que no hay diferencias

# Definir el nivel de significancia (alfa)
alfa = 0.05

Fsignificativas = []
F = []
F_critico = []
psignificativas = []
p = []
epsilon = []
p_corr = []

vars_no_sig = []
vars_sig = []
for i, var in enumerate(variables_dependientes):
    '''
    si quiero sin campeones ni filler tengo que poner aca df_sin_camp_ni_filler en vez de df
    '''
    aov = df.rm_anova(dv = var, within='Tiempo', subject='Sujetos',  detailed=False)
    
    # Definir los grados de libertad del numerador y del denominador
    df_between = aov['ddof1'][0]  # Grados de libertad del numerador
    df_within = aov['ddof2'][0]   # Grados de libertad del denominador

    # Calcular el valor crítico de F
    f_critical = f.ppf(1 - alfa, df_between, df_within)
    
    F.append(aov['F'][0])
    F_critico.append(f_critical)
    epsilon.append(aov['eps'][0])
    p.append(aov['p-unc'][0])
    
    if f_critical > aov['F'][0]:
        #print('La variable ' + var + f" tiene un F ({aov['F'][0]}) que no supera el crítico ({f_critical}).")
        vars_no_sig.append(var)
        Fsignificativas.append(False)
    else:
        Fsignificativas.append(True)
    if 'p-GG-corr' not in np.array(aov.columns):
        pval = aov['p-unc'][0]
        p_corr.append(False)
    else:
        pval = aov['p-GG-corr'][0]
        p_corr.append(aov['p-GG-corr'][0])
    if pval > alfa:
       # print("La variable " + var + f" tiene un pval corregido ({pval}) que no supera el nivel de significancia ({alfa}).")
        psignificativas.append(False)
        #vars_no_sig.append(var)
    else:
        psignificativas.append(True)
    if pval < alfa:
        if f_critical < aov['F'][0]:
            vars_sig.append(var)
        
#Acomodo todo en un csv

df_resultados = pd.DataFrame({
    'Variable': variables_dependientes,
    'F': F,
    'F_critico': F_critico,
    'F es significativo?': Fsignificativas,
    'P': p,
    'Epsilon': epsilon,
    'P_corr': p_corr,
    'P significativo?': psignificativas,
})

print(f"la cantidad de vars significativas en {condicion_que_evaluamos} es {len(vars_sig)}")

# Guardar el DataFrame en un archivo CSV
#df_resultados.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ANOVA y correlacion/ANOVAS.csv', index=False)
# para guardar sin camp ni filler busca si quiero sin campeones ni filler tengo que poner aca df_sin_camp_ni_filler en vez de df
#%% ahora el test de tukey para las variables donde dió significativa ver donde es

# comparaciones = 5*4/2 # n(n-1)/2 donde n es el número de grupos

# from scipy.stats import t

# def valor_critico_t(grados_de_libertad):
#     alpha = 0.05/10 #Nivel de significancia
#     return t.ppf(1 - alpha / 2, df=grados_de_libertad)

# # Realizar la prueba de Tukey como prueba post hoc
# for i, var in enumerate(vars_sig):
#     print(var)
#     posthoc_result = pg.pairwise_tests(dv= var, within='Tiempo', subject='Sujetos', data=df) #alpha=0.05/comparaciones correction='bonferroni' DA LO MISMO PONER ESTO
#     # Calcula el valor crítico de T para una prueba t de dos colas
#     posthoc_result['T critico'] = posthoc_result['dof'].apply(valor_critico_t)
#     posthoc_result['p pasa'] = np.where(posthoc_result['p-unc'] < 0.05/10, True, np.nan)
#     posthoc_result['t pasa'] = np.where(abs(posthoc_result['T']) > posthoc_result['T critico'], True, np.nan)
#     #print(posthoc_result)
#     print(posthoc_result[['A','B','p pasa', 't pasa']])