# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:43 2023

@author: corir
"""
#%% librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as npl
import seaborn as sns
from matplotlib.image import imread
from matplotlib.patches import Rectangle, Polygon
import matplotlib.ticker as plticker
import re
import scipy.stats
from matplotlib.gridspec import GridSpec



#PCs
# Clase para realizar componentes principales
from sklearn.decomposition import PCA
# Estandarizador (transforma las variables en z-scores)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler() # Creamos el estandarizador para usarlo posteriormente


#para ANOVA
import pingouin as pg
from scipy.stats import f
#post ANOVA --> tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%% funciones

def PCA_estandarizando(data, n_components = None, graph_var = True, max_var = 0.7, graph_PCs = True, n_graph_PCs = 7):
    '''
    le das el dataframe al que queres que le haga PCs 
    estandariza y te da las PCs, le podes pedir el gráfico de la varianza y el de las
    primeras n PCs
    '''

    X = data.to_numpy()
    
    # Ajustamos el estandarizador
    std_scale.fit(X)

    # Aplicamos el estandarizador y obtenemos la matriz de features escaleados
    X_scaled = std_scale.transform(X)

    # Creación del modelo. Si el número de componentes no se específica, 
    # se obtienen tantas componentes principales como features en nuestro dataset.
    pca = PCA(n_components=n_components)

    # Ajustamos el modelo a los datos escaleados
    pca.fit(X_scaled)

    # Obtenemos la descripción de los datos en el espacio de componentes principales
    X_pca = pca.transform(X_scaled)

    evr = pca.explained_variance_ratio_
    
    if graph_var == True:
        # con .explained_variance_ratio_ vemos la fracción de información que aporta cada componente
    
        fig, ax = plt.subplots(1, 1, figsize = (18, 10))
    
        # Calculamos el acumulado con la función cumsum de numpy 
        varianza_acumulada = np.cumsum(evr)
        max_comp = np.where(varianza_acumulada > max_var)[0][0] + 1
    
        ax.plot(range(1, len(evr) + 1), varianza_acumulada, '.-', markersize = 20, color = color_celeste, zorder = 5)
        ax.set_ylabel('Fracción acumulada de varianza explicada')
        ax.set_xlabel('Cantidad de componentes principales')
        ax.axhline(y=max_var, color=color_gris, linestyle='--', linewidth = 4, label=f'{max_var*100}%')
        ax.axvline(x = max_comp, color=color_gris, linestyle='--', linewidth = 4)
        ax.grid(True)
        plt.legend()

    if graph_PCs == True:

        npl.rcParams["axes.labelsize"] = 20
        npl.rcParams['xtick.labelsize'] = 20
        npl.rcParams['ytick.labelsize'] = 20


        variables = list(data.columns)
        
        componentes_principales = [pca.components_[i] for i in range(0, n_graph_PCs)]
    
        # Crea un diccionario con las componentes principales y las variables
        data_pcs = {f"PC{i+1}": componentes_principales[i] for i in range(len(componentes_principales))}
    
        # Crea el DataFrame
        df_vars_1 = pd.DataFrame(data_pcs, index=variables)
    
        #df_vars_1.to_csv('C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/PCA y matriz de corr/primeras_6_componentes.csv')
    
        center_value= 0
        plt.figure(figsize = (30, len(variables)))
        sns.heatmap(df_vars_1, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, linecolor="black", center = center_value) #cbar_kws={"shrink": 0.75}) #"YlGnBu,  annot=True
    
    
        plt.yticks(rotation=0) #plt.yticks(variables)
        plt.xticks(rotation=0)
        #plt.xlabel("Componentes Principales")
    
    
        # Muestra el gráfico
        plt.show()
    return X_pca, pca, evr

def rgb_to_hex(rgb):
    # Asegurarse de que los valores estén en el rango [0, 1]
    rgb = tuple(max(0, min(1, x)) for x in rgb)

    # Convertir los valores RGB a enteros en el rango [0, 255]
    rgb_int = tuple(int(x * 255) for x in rgb)

    # Formatear el color en formato hexadecimal
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_int)

    return hex_color

def lighten_color_hex(hex_color, factor=0.4):
    # Asegurarse de que el factor está en el rango [0, 1]
    factor = max(0, min(factor, 1))

    # Convertir el color hexadecimal a RGB
    rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    # Aclarar el color en formato RGB
    lightened_rgb = tuple(min(1, x + (1 - x) * factor) for x in rgb_color)

    # Convertir el nuevo color RGB a formato hexadecimal
    lightened_hex_color = "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in lightened_rgb))

    return lightened_hex_color

def darken_color(color, factor=0.7):
    """
    Oscurece un color ajustando manualmente sus componentes RGB.

    Parameters:
    - color (str): Código hexadecimal del color.
    - factor (float): Factor de oscurecimiento (0 a 1).

    Returns:
    - str: Código hexadecimal del color oscurecido.
    """
    # Obtener componentes RGB
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # Aplicar el factor de oscurecimiento
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Asegurarse de que los valores estén en el rango correcto (0-255)
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))

    # Convertir los componentes oscurecidos de nuevo a código hexadecimal
    darkened_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

    return darkened_color

def interval(datas):
    min_datas = []
    max_datas = []
    means = []
    std_data = []
    
    for i in range(0, len(datas)):
        min_datas.append(min(datas[i]))
        max_datas.append(max(datas[i]))
        means.append(np.mean(datas[i]))
        #std_data.append(np.std(datas[i]))
        std_data.append(scipy.stats.sem(datas[i],ddof=1))
            
    return min_datas, max_datas, means, std_data



def plot_interval(min0, min1, max0, max1, mean, std, xlim=None, y=0, thickness=0.4, std_interval = True, color1='k', color2 = 'k', alpha = 0.7, ax=None, time2 = "up"):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.tick_params(labelbottom=True)  # to get tick labels on all axes
    ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(which='both', direction='in')`  # tick marks above instead below the axis
    #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1)) # major ticks in steps of 10
    #ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))  # minor ticks in steps of 1
    ax.set_ylim(-1.5,.5)
    if std_interval == True:
        if xlim is None:
            max_tot = max(mean[0]+std[0], mean[1]+std[1])
            min_tot = min(mean[0]-std[0], mean[1]-std[1])
            interval_data = max_tot - min_tot
            xlim =  (min_tot-interval_data*0.1, max_tot+interval_data*0.1)
        ax.set_xlim(xlim)
        ax.add_patch(Rectangle((mean[0]-std[0], y), mean[0]+std[0]-(mean[0]-std[0]), thickness, linewidth=0, color=color1, alpha = alpha))
        ax.plot(mean[0], y + thickness/2, "o", color = darken_color(color1), markersize=6)
        if time2 == "up":
            ax.add_patch(Rectangle((mean[1]-std[1], y-thickness), 2*std[1], thickness, linewidth=0, color= color2, alpha = alpha))
            ax.plot(mean[1], thickness/2, "o", color = darken_color(color2), markersize=6)
        else:
            #se grafica por abajo del eje x
            ax.add_patch(Rectangle((mean[1]-std[1], y-thickness), 2*std[1], thickness, linewidth=0, color= color2, alpha = alpha))
            #sino hago mas fino thinkness (y dejo y = 0 y thinkess 0.4: el segundo param es -thinkness/2
            ax.plot(mean[1], thickness/2, "o", color = darken_color(color2), markersize=6) #edgecolor = 'k' no funca
    else:
        if xlim is None:
            max_tot = max(max0, max1)
            min_tot = min(min0, min1)
            interval_data = max_tot - min_tot
            xlim =  (min_tot-interval_data*0.1, max_tot+interval_data*0.1)
        ax.set_xlim(xlim)
        ax.add_patch(Rectangle((min0, y), max0-min0, thickness, linewidth=0, color=color1, alpha = alpha))
        ax.plot(mean[0], y + thickness/2, "o", color = darken_color(color1), markersize=6)
        if time2 == "up":
            ax.add_patch(Rectangle((min1, y), max1-min1, thickness, linewidth=0, color= color2, alpha = alpha))
            ax.plot(mean[1], thickness/2, "o", color = darken_color(color2), markersize=6)
        else:
            #se grafica por abajo del eje x
            ax.add_patch(Rectangle((min1, y-thickness), max1-min1, thickness, linewidth=0, color= color2, alpha = alpha))
            #sino hago mas fino thinkness (y dejo y = 0 y thinkess 0.4: el segundo param es -thinkness/2
            ax.plot(mean[1], thickness/2, "o", color = darken_color(color2), markersize=6)
    c = 0
    triangle1 = [(xlim[0] - (xlim[1]-xlim[0])*0.03, c), (xlim[0], c-thickness), (xlim[0], c+thickness)]
    ax.add_patch(Polygon(triangle1, linewidth=0, color='black', clip_on=False))
    triangle2 = [(xlim[1] + (xlim[1]-xlim[0])*0.03, c), (xlim[1], c-thickness), (xlim[1], c+thickness)]
    ax.add_patch(Polygon(triangle2, linewidth=0, color='black', clip_on=False))
    return ax

#%% el santo trial y colores

entrevista = 'Primera'

no_autop = True #pone false si queres que las tenga en cuenta para el análisis

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]

condicion = temas[0]

color_celeste = "#79b4b7ff"
color_gris = "#9fa0a4ff"

drop_12 = True

eliminando_outliers = True

path_imagenes = "C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Figuras_finales/Dos tiempos"

#kmeans_TF = False #si es true kmeans es el metodo 0, k medoids el 1 y asi
#%% path data primera entrevista para hacer PCs que optimizan a la misma

path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

if eliminando_outliers == True:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv'

    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv'

#%% mas variables de interés

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3, #3 
}

eliminamos_pysent = ['Valencia pysent', 'Valencia e intensidad pysent']#, 'Valencia2 pysent', 'Valencia e intensidad2 pysent']


#valencia2 pysent kmeans
nro_pcs_kmeans = 2 #4 
vars_no_imp_kmeans = ['primera_persona_norm', 'num verb norm', 'cohe_norm_d=3', 'diámetro', 'transitivity', 'average_CC', 'selfloops']
#valencia 2 average jerarquico
nro_pcs_average = 12
vars_no_imp_average = ['primera_persona_norm', 'tercera_persona_norm', 'num noun norm', 'num verb norm', 'num numeral norm', 'num propn norm', 'Intensidad pysent', 'cohe_norm_d=3', 'selfloops']
#valencia 2 kmedoids
vars_no_imp_kmedoids = ['primera_persona_norm']
nro_psc_kmedoids = 13

vars_no_imp_metodos = [vars_no_imp_kmeans, vars_no_imp_average, vars_no_imp_kmedoids]
nro_pcs_metodos = [nro_pcs_kmeans, nro_pcs_average, nro_psc_kmedoids]

metodo = 0 #0 kmeans, 1 average, 2 kmedoids
vars_no_imp = vars_no_imp_metodos[metodo] 
nro_pcs = nro_pcs_metodos[metodo]

#%% PCs que separan de manera óptima la primera entrevista con el método seleccionado
df = pd.read_csv(path_conautopercepcion_todas)

df = df.dropna()

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df[~df['Condición'].isin([5,2])]

#condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_conautopercepcion_todas, condiciones = [5, 2])
               
#df = df.drop(['Sujetos', 'Condición'] + eliminamos_pysent, axis=1)

if no_autop == True:
    df = pd.read_csv(path_conautopercepcion_todas)
    
    if drop_12 == True:
        df = df.dropna()
    
    df['Condición'] = df['Condición'].map(mapping)
    
    df_vars_crudas = df

    df = df[~df['Condición'].isin([5,2])]

    df = df.drop(['Sujetos', 'Condición', 'Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'] + eliminamos_pysent, axis = 1) 

    if drop_12 != True:
        df = df.dropna()
    
        #condicion_labels, indices_camp_ar_cfk = indices_condiciones(path_sinautopercepcion_todas, condiciones = [5, 2], drop = vars_no_imp_n[n])
    
df = df.drop(vars_no_imp, axis = 1)

X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  nro_pcs, max_var = 0.2, graph_var = True, graph_PCs = True, n_graph_PCs = nro_pcs)

nro_variables = len(df.columns)
for i in range(nro_pcs):
    orden_invertida_pcs = np.argsort(abs(pca1.components_[i]))
    orden_pcs = orden_invertida_pcs[::-1]
    print(f"PARA LA PC {i+1}")
    for j in range(nro_variables):
        print(f"{df.columns[orden_pcs[j]]} pesa {round(pca1.components_[i][orden_pcs[j]], 2)}")
    

#a = set(np.argsort(abs(pca1.components_[0]))[-round(3):])
#si quisiera chequear que el R es el correcto y eso debería subir las funciones R_clausterizacion indices_condiciones y varias mas que usan estas y aca podría ver que efectivamente esta todo bn
#k = 2
#R = R_clausterizacion(X_pca, k, condicion_labels, indices_camp_ar_cfk, kmeans = kmeans_TF, etiquetas_print = True)

#%% para dos tiempos path
entrevista = 'dos tiempos'

if entrevista == 'dos tiempos':
    path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_dostiempos.csv'
    
    path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_dostiempos.csv'
else:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_todos_temas.csv'
    
    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_todos_temas.csv'

if eliminando_outliers == True:
    if entrevista == 'dos tiempos':
        path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_dostiempos.csv'
        
        path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_dostiempos.csv'

    


#%% dataframe con PCs en primer y segundo tiempo con transformación para optimizar R de la primer entrevista

save = True

vars_sig_PC1 = ["num_nodes_LSC", "Comunidades_LSC", "density", "num_palabras_unicas_norm"] #si no queda mal agregaria k_mean, L3, L2, pesan bastante
vars_sig_PC2 = ["Valencia e intensidad2 pysent", "Negativo pysent", "Positivo pysent", "Intensidad pysent"] #si no queda mal Intensidad pysent
vars_sig_PC3 = ["Detalles externos norm", "Detalles internos norm"]
vars_sig_PC4 = ["cohe_norm_d=1", "cohe_norm_d=2"]

path_image = 'C:/Users/Usuario/Desktop/Cori/Tesis/Imagenes_tesis/Resultados/Comparacion_t/auxiliar/'

path_image_final = 'C:/Users/Usuario/Desktop/Cori/Tesis/Imagenes_tesis/Resultados/Comparacion_t/'

save_image_con_fondo = True

sns.set_context("paper")

# Definir los colores originales
color_1 = "#d896ff"#"#fb2e01"
color_2 = "#800080"#"6fcb9f"

import matplotlib.patheffects as mpe

outline=mpe.withStroke(linewidth=8, foreground='black')


rainbow_palette = sns.color_palette("icefire_r", n_colors=5)

#rainbow_palette = sns.color_palette("autumn_r", n_colors=2)



# Asignar colores a las variables
color_1 = rgb_to_hex(rainbow_palette[0])
color_2 = rgb_to_hex(rainbow_palette[1])
#darkened_color_2 = sns.dark_palette(color_2)


darkened_color_1 = lighten_color_hex(color_1, factor = 0.4)
darkened_color_2 = lighten_color_hex(color_2, factor = 0.4)

# Oscurecer los colores
darkened_color_1 = darken_color(color_1)
darkened_color_2 = darken_color(color_2, factor = 0.95)

k = 2

cond_elim = [1,2,3,4]

valor_a_buscar = 5

condiciones_elim = [[2,3,4,5], [1,3,4,5],[1,2,4,5],[1,2,3,5], [1,2,3,4]]

valor_a_buscar_list = [1,2,3,4,5]


for l, (cond_elim, valor_a_buscar) in enumerate(zip(condiciones_elim, valor_a_buscar_list), 1):

    df = pd.read_csv(path_conautopercepcion_todas)
    
    #df = df.dropna()
    
    # Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
    df['Condición'] = df['Condición'].map(mapping)
    
    df = df[~df['Condición'].isin(cond_elim)]
    
    #condicion_labels, indices_ = indices_condiciones(path_conautopercepcion_todas, condiciones = cond_elim)
    
    
    if no_autop == True:
        df = df.drop(['Recuerdo_autop', 'Valencia_autop', 'Intensidad_autop', 'ValeInt_autop'], axis = 1) 
        #condicion_labels, indices_ = indices_condiciones(path_sinautopercepcion_todas, condiciones = cond_elim)
    
        
    df = df.dropna()
    
    df_vars_crudas = df
    
    df_vars_tiempo_1 = df_vars_crudas[df_vars_crudas['Tiempo'] == 1]
    df_vars_tiempo_2 = df_vars_crudas[df_vars_crudas['Tiempo'] == 2]

    columna_tiempos = df['Tiempo']
    columna_sujetos = df['Sujetos']
    columna_condicion = df['Condición']
    
    #tengo que tirar esto despues de dopear nans sino queda dim diferente de columna t que después agrego y de 
    # X_pca
    df = df.drop(['Sujetos', 'Condición', 'Tiempo'] + eliminamos_pysent, axis=1) 
    
    df_vars_tiempo_1_filtrado = df_vars_tiempo_1.drop(['Sujetos', 'Condición', 'Tiempo'] + eliminamos_pysent + vars_no_imp, axis=1) 
    
    df_vars_tiempo_2_filtrado = df_vars_tiempo_2.drop(['Sujetos', 'Condición', 'Tiempo'] + eliminamos_pysent + vars_no_imp, axis=1)
          
    df = df.drop(vars_no_imp, axis = 1)
    
    X = df.to_numpy()
    X_t1 = df_vars_tiempo_1_filtrado.to_numpy()
    X_t2 = df_vars_tiempo_2_filtrado.to_numpy()
    
    # Ajustamos el estandarizador
    std_scale.fit(X) 
    # Aplicamos el estandarizador y obtenemos la matriz de features escaleados
    X_scaled = std_scale.transform(X)
    X_pca = pca1.transform(X_scaled)
    
    #std_scale.fit(X_t1)
    X_scaled_t1 = std_scale.transform(X_t1)
    X_pca_t1 = pca1.transform(X_scaled_t1)
    
    #std_scale.fit(X_t2)
    X_scaled_t2 = std_scale.transform(X_t2)
    X_pca_t2 = pca1.transform(X_scaled_t2)
    
    # Crear un DataFrame con las columnas PC1, PC2, PC3, ...
    columns_pca = [f'PC{i}' for i in range(1, X_pca.shape[1] + 1)]
    df_pca = pd.DataFrame(data=X_pca, columns=columns_pca)
    
    df_pca_t1 = pd.DataFrame(data=X_pca_t1, columns=columns_pca)
    df_pca_t2 = pd.DataFrame(data=X_pca_t2, columns=columns_pca)
    '''
    si usas df_pca_ts da lo mismo, ojo cuando haces el X_scaled, no es de cada una el std_scale
    porque sino te normaliza todo al mismo intervalo... debería transformas las pcs por separado pero no el X_scaled
    '''
    #df_pca_ts!!! abajo
    df_pca_ts = pd.concat([df_pca_t1, df_pca_t2], ignore_index=True)#df_pca_ts!!! este
    #df_pca_ts!!! arriba
    
    df_pca.index = df.index #sino cuando inserte las columnas de abajo como tienen indice va a poner nans en los que no coincidan
   # df_pca_ts.index = df.index
    
    df_pca.insert(0, 'Tiempo', columna_tiempos)
    df_pca.insert(0, 'Condición', columna_condicion)
    df_pca.insert(0, 'Sujetos', columna_sujetos)
    
   # df_pca_ts.insert(0, 'Tiempo', columna_tiempos)
   # df_pca_ts.insert(0, 'Condición', columna_condicion)
   # df_pca_ts.insert(0, 'Sujetos', columna_sujetos)
    
    #% ANOVA
    
    variables_dependientes = list(df_pca.columns)[3:] #sacamos sujeto condicion y tiempo
    
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
        
        aov = df_pca.rm_anova(dv = var, within='Tiempo', subject='Sujetos',  detailed=False)
        
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
    
    for clave, valor in mapping.items():
        if valor == valor_a_buscar:
            clave_correspondiente = clave
            break
        
    print(f"la cantidad de vars significativas en {clave_correspondiente} es {len(vars_sig)}")
    print(f"las variables significativas son {vars_sig}")
    
    if vars_sig != []:
        #las figuras
        for j, var_sig in enumerate(vars_sig):
            
            if var_sig in ["PC1", "PC2"]: #PC3 las que siguen
            
            
            #plt.figure(l*10+j), plt.clf()
            #plt.figure()
                nrows = 8 #va a tener 4 subplots
                fig = plt.figure(figsize=(12,6.6))
                ax1 = plt.subplot2grid((nrows, 5), (0, 0), rowspan = nrows, colspan=3)
                ax2 = plt.subplot2grid((nrows, 5), (0, 3), rowspan = 1, colspan=1)
                ax3 = plt.subplot2grid((nrows, 5), (1, 3), rowspan = 1, colspan=1)
                ax4 = plt.subplot2grid((nrows, 5), (2, 3), rowspan = 1, colspan=1)
                ax5 = plt.subplot2grid((nrows, 5), (3, 3), rowspan = 1, colspan=1)
                #plt.tight_layout()
                
                df_pca_tiempo_1 = df_pca[df_pca['Tiempo'] == 1]
                df_pca_tiempo_2 = df_pca[df_pca['Tiempo'] == 2]
                # Crear un histograma
                median1 = np.nanmedian(df_pca_tiempo_1[var_sig])
                median2 = np.nanmedian(df_pca_tiempo_2[var_sig])
                mean1 = np.nanmean(df_pca_tiempo_1[var_sig])
                mean2 = np.nanmean(df_pca_tiempo_2[var_sig])
                hist1, bins1, m = ax1.hist(df_pca_tiempo_1[var_sig], edgecolor='black', color = color_1, alpha = 0.7, label = 'Primer tiempo', zorder = 15)
                hist2, bins2, m = ax1.hist(df_pca_tiempo_2[var_sig], edgecolor='black', color = color_2, alpha = 0.7, label = 'Segundo tiempo', zorder = 10)
                max_ylim = max(max(hist1), max(hist2))
                #plt.vlines(median1, 0, max_ylim, color='black', linestyle='-', linewidth=2, label = "Tiempo 1")
                #plt.vlines(median2, 0, max_ylim, color='red', linestyle='-', linewidth=2,  label = "Tiempo 2")
                ax1.vlines(mean1, 0, max_ylim, color= darkened_color_1, linestyle='--', linewidth=5, zorder = 20, path_effects=[outline])
                ax1.vlines(mean2, 0, max_ylim, color= darkened_color_2, linestyle='--', linewidth=5, zorder = 20, path_effects=[outline])

                nro = re.search(r'\d+', var_sig).group()
                
                ax1.set_yticks([0, 3, 6, 9, 12])
                ax1.tick_params(axis='both', labelsize=15)
                #plt.gca().set_zorder(10)
                ax1.set_xlabel(f"Componente principal {nro}", fontsize = 18)
                ax1.set_ylabel("Cuentas", fontsize = 18)
                #plt.title(f'{clave_correspondiente}')
                ax1.legend(fontsize = 15)
                ax1.grid(True, zorder = 5)
                #plt.subplots_adjust(bottom=0.15, left=0.12)

    
                if var_sig == "PC1":
                    var1 = vars_sig_PC1[0]
                    var2 = vars_sig_PC1[1]
                    var3 = vars_sig_PC1[2]
                    var4 = vars_sig_PC1[3]
                    vars_label = ["Núm nodos", "Comunidades", "Densidad", "Núm palabras"]
                    data1 = df_vars_tiempo_1[vars_sig_PC1[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC1[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC1[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC1[1]]
                    data5 = df_vars_tiempo_1[vars_sig_PC1[2]]
                    data6 = df_vars_tiempo_2[vars_sig_PC1[2]]
                    data7 = df_vars_tiempo_1[vars_sig_PC1[3]]
                    data8 = df_vars_tiempo_2[vars_sig_PC1[3]]
                
                if var_sig == "PC2":
                    var1 = vars_sig_PC2[0]
                    var2 = vars_sig_PC2[1]
                    var3 = vars_sig_PC2[2]
                    var4 = vars_sig_PC2[3]
                    vars_label = ["Val. e inten.", "Negativo", "Positivo", "Intensidad"]
                    data1 = df_vars_tiempo_1[vars_sig_PC2[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC2[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC2[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC2[1]]
                    data5 = df_vars_tiempo_1[vars_sig_PC2[2]]
                    data6 = df_vars_tiempo_2[vars_sig_PC2[2]]
                    data7 = df_vars_tiempo_1[vars_sig_PC2[3]]
                    data8 = df_vars_tiempo_2[vars_sig_PC2[3]]
                    
                if var_sig == "PC3":
                    var1 = "num advs norm"
                    var2 = "ASP"
                    var3 = "k_mean"
                    var4 = "num noun norm"
                    
                    data1 = df_vars_tiempo_1[var1]
                    data2 = df_vars_tiempo_2[var1]
                    data3 = df_vars_tiempo_1[var2]
                    data4 = df_vars_tiempo_2[var2]
                    data5 = df_vars_tiempo_1[var3]
                    data6 = df_vars_tiempo_2[var3]
                    data7 = df_vars_tiempo_1[var4]
                    data8 = df_vars_tiempo_2[var4]
                
                datas = [data1, data2, data3, data4, data5, data6, data7, data8]
                
                min_datas, max_datas, means, std_data = interval(datas)

                n_plots = int(len(datas)/2)
                dist_between_axis_in_inches = 0.4

                #fig, axs = plt.subplots(n_plots, sharex=False, figsize=(10, dist_between_axis_in_inches*len(datas)))
                ax2.set_xlabel(vars_label[0], fontsize = 16)
                ax3.set_xlabel(vars_label[1], fontsize = 16)
                ax4.set_xlabel(vars_label[2], fontsize = 16)
                ax5.set_xlabel(vars_label[3], fontsize = 16)
                axs = [ax2, ax3, ax4, ax5]
                for i in range(0, int(len(datas)/2)):
                    axs[i] = plot_interval(min_datas[i*2], min_datas[i*2+1], max_datas[i*2], max_datas[i*2+1], means[i*2:i*2+2], std_data[i*2:i*2+2], thickness=0.2, y = 0.2, ax=axs[i], color1= color_1, color2 = color_2, time2 = "down")
                    axs[i].xaxis.set_label_coords(1.6, 0.85)
                

                plt.subplots_adjust(wspace=3.7)
                
                plt.tight_layout()
                plt.show()
                
                if save == True:
                    print("estoy guardando")
                    plt.savefig(path_imagenes + f'/Transparentes/{clave}_{var_sig}_tansparente.png', transparent = True)
                    plt.savefig(path_imagenes + f'/No transparentes/{clave}_{var_sig}.png', transparent = False)
                    plt.savefig(path_imagenes + f'/No transparentes/{clave}_{var_sig}.pdf')

            
            if var_sig in ["PC3", "PC4"]: #PC2 las que siguen
            
                nrows = 8 #va a tener 4 subplots
                fig = plt.figure(figsize=(12,6.6))
                ax1 = plt.subplot2grid((nrows, 5), (0, 0), rowspan = nrows, colspan=3)
                ax2 = plt.subplot2grid((nrows, 5), (0, 3), rowspan = 1, colspan=1)
                ax3 = plt.subplot2grid((nrows, 5), (1, 3), rowspan = 1, colspan=1)

                #plt.tight_layout()
                
                df_pca_tiempo_1 = df_pca[df_pca['Tiempo'] == 1]
                df_pca_tiempo_2 = df_pca[df_pca['Tiempo'] == 2]
                # Crear un histograma
                median1 = np.nanmedian(df_pca_tiempo_1[var_sig])
                median2 = np.nanmedian(df_pca_tiempo_2[var_sig])
                mean1 = np.nanmean(df_pca_tiempo_1[var_sig])
                mean2 = np.nanmean(df_pca_tiempo_2[var_sig])
                hist1, bins1, m = ax1.hist(df_pca_tiempo_1[var_sig], edgecolor='black', color = color_1, alpha = 0.7, label = 'Primer tiempo', zorder = 15)
                hist2, bins2, m = ax1.hist(df_pca_tiempo_2[var_sig], edgecolor='black', color = color_2, alpha = 0.7, label = 'Segundo tiempo', zorder = 10)
                max_ylim = max(max(hist1), max(hist2))
                #plt.vlines(median1, 0, max_ylim, color='black', linestyle='-', linewidth=2, label = "Tiempo 1")
                #plt.vlines(median2, 0, max_ylim, color='red', linestyle='-', linewidth=2,  label = "Tiempo 2")
                ax1.vlines(mean1, 0, max_ylim, color= darkened_color_1, linestyle='--', linewidth=5, zorder = 20, path_effects=[outline])
                ax1.vlines(mean2, 0, max_ylim, color= darkened_color_2, linestyle='--', linewidth=5, zorder = 20, path_effects=[outline])

                nro = re.search(r'\d+', var_sig).group()
                
                ax1.set_yticks([0, 3, 6, 9, 12])
                ax1.tick_params(axis='both', labelsize=15)
                #plt.gca().set_zorder(10)
                ax1.set_xlabel(f"Componente principal {nro}", fontsize = 18)
                ax1.set_ylabel("Cuentas", fontsize = 18)
                #plt.title(f'{clave_correspondiente}')
                ax1.legend(fontsize = 15)
                ax1.grid(True, zorder = 5)
                
                if var_sig == "PC3":

                    var1 = vars_sig_PC3[0]
                    var2 = vars_sig_PC3[1]
                    vars_label = ["Detalles externos", "Detalles internos"]
                    data1 = df_vars_tiempo_1[vars_sig_PC3[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC3[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC3[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC3[1]]
                
                if var_sig == "PC4":
                    var1 = vars_sig_PC4[0]
                    var2 = vars_sig_PC4[1]
                    vars_label = ["Coherencia d = 1", "Coherencia d = 2"]
                    data1 = df_vars_tiempo_1[vars_sig_PC4[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC4[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC4[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC4[1]]
                    
                datas = [data1, data2, data3, data4]
                
                min_datas, max_datas, means, std_data = interval(datas)

                n_plots = int(len(datas)/2)
                dist_between_axis_in_inches = 0.4
                
                ax2.set_xlabel(vars_label[0], fontsize = 16)
                ax3.set_xlabel(vars_label[1], fontsize = 16)

                axs = [ax2, ax3]

                #fig, axs = plt.subplots(n_plots, sharex=False, figsize=(10, dist_between_axis_in_inches*len(datas)))
                for i in range(0, int(len(datas)/2)):
                    axs[i] = plot_interval(min_datas[i*2], min_datas[i*2+1], max_datas[i*2], max_datas[i*2+1], means[i*2:i*2+2], std_data[i*2:i*2+2], thickness=0.2, y = 0.2, ax=axs[i], color1= color_1, color2 = color_2, time2 = "down")
                    axs[i].xaxis.set_label_coords(1.6, 0.85)
                    axs[i].tick_params(axis='x', labelsize=14) 
                    
                plt.subplots_adjust(wspace=3.7)

                # Ajustar el diseño para evitar solapamientos
                plt.tight_layout()
                
                plt.show()
                if save == True:
                    print("estoy guardando")
                    plt.savefig(path_imagenes + f'/Transparentes/{clave}_{var_sig}_tansparente.png', transparent = True)
                    plt.savefig(path_imagenes + f'/No transparentes/{clave}_{var_sig}.png', transparent = False)
                    plt.savefig(path_imagenes + f'/No transparentes/{clave}_{var_sig}.pdf')

