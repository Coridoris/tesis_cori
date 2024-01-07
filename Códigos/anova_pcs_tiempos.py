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
    
    mins_d = []
    maxs_d = []
    min_datas = []
    max_datas = []
    for i in range(0, len(datas), 2):
        min_datas.append(min(datas[i]))
        min_datas.append(min(datas[i+1]))
        max_datas.append(max(datas[i]))
        max_datas.append(max(datas[i+1]))
        mins_d.append([min(datas[i]), min(datas[i+1])])
        maxs_d.append([max(datas[i]), max(datas[i+1])])
    
    x_datas = []
    for i in range(0, int(len(datas)/2)):
        max_d = max(maxs_d[i])
        min_d = min(mins_d[i])
        int_data = max_d - min_d
        min_x = min_d - 0.03 * int_data #le resto el 3% del intervalo
        max_x = max_d + 0.03 * int_data
        x_datas.append(np.linspace(min_x, max_x, 100))
        x_datas.append(np.linspace(min_x, max_x, 100))
    

    num_masks = len(datas)
    
    masks = [np.zeros_like(x_datas[0], dtype=bool) for _ in range(num_masks)]
    for i in range(num_masks):
        masks[i] += (x_datas[i] >= min_datas[i]) * (x_datas[i] <= max_datas[i])
    
    #masks = masks[::-1] # reverse to get the masks plotted from bottom to top      
    return masks, x_datas 



def bool2extreme(mask, times) :
    """return xmins and xmaxs for intervals in times"""
    binary = 1*mask
    slope = np.diff(binary)

    extr = (slope != 0)
    signs = slope[extr]
    mins = list(times[1:][slope==1])
    maxs = list(times[:-1][slope==-1])
    if signs[0]==-1:
        mins = [times[0]] + mins
    if signs[-1]==1:
        maxs = maxs + [times[-1]]
    return mins, maxs



def plot_interval(two_masks, times, xlim=None, y=0, thickness=0.4, color1='k', color2 = 'k', alpha = 0.7, ax=None, time2 = "up"):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.tick_params(labelbottom=True)  # to get tick labels on all axes
    # ax.tick_params(which='both', direction='in')`  # tick marks above instead below the axis
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1)) # major ticks in steps of 10
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))  # minor ticks in steps of 1
    ax.set_ylim(-1.5,.5)
    if xlim is None:
        xlim = (times[0]-0.9, times[-1]+0.9)
    ax.set_xlim(xlim)
    xmins, xmaxs = bool2extreme(two_masks[0], times)
    xmins1, xmaxs1 = bool2extreme(two_masks[1], times)
    
    ax.add_patch(Rectangle((xmins[0], y), xmaxs[0]-xmins[0], thickness, linewidth=0, color=color1, alpha = alpha))
    if time2 == "up":
        ax.add_patch(Rectangle((xmins1[0], y), xmaxs1[0]-xmins1[0], thickness, linewidth=0, color= color2, alpha = alpha))
    else:
        #se grafica por abajo del eje x
        ax.add_patch(Rectangle((xmins[0], y-thickness), xmaxs[0]-xmins[0], thickness, linewidth=0, color= color2, alpha = alpha))
    triangle1 = [(xlim[0]*1.05, y), (xlim[0], y-thickness), (xlim[0], y+thickness)]
    ax.add_patch(Polygon(triangle1, linewidth=0, color='black', clip_on=False))
    triangle2 = [(xlim[1]*1.05, y), (xlim[1], y-thickness), (xlim[1], y+thickness)]
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
nro_pcs_kmeans = 4 
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

X_pca, pca1, evr1 = PCA_estandarizando(df, n_components =  nro_pcs, max_var = 0.6, graph_var = True, graph_PCs = True, n_graph_PCs = nro_pcs)

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

# Oscurecer los colores
darkened_color_1 = darken_color(color_1)
darkened_color_2 = darken_color(color_2)

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
          
    df = df.drop(vars_no_imp, axis = 1)
    
    X = df.to_numpy()
    
    # Ajustamos el estandarizador
    std_scale.fit(X)
    
    # Aplicamos el estandarizador y obtenemos la matriz de features escaleados
    X_scaled = std_scale.transform(X)
    X_pca = pca1.transform(X_scaled)
    
    # Crear un DataFrame con las columnas PC1, PC2, PC3, ...
    columns_pca = [f'PC{i}' for i in range(1, X_pca.shape[1] + 1)]
    df_pca = pd.DataFrame(data=X_pca, columns=columns_pca)
    
    df_pca.index = df.index #sino cuando inserte las columnas de abajo como tienen indice va a poner nans en los que no coincidan
    
    df_pca.insert(0, 'Tiempo', columna_tiempos)
    df_pca.insert(0, 'Condición', columna_condicion)
    df_pca.insert(0, 'Sujetos', columna_sujetos)
    
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
            #plt.figure(l*10+j), plt.clf()
            plt.figure()
            
            df_pca_tiempo_1 = df_pca[df_pca['Tiempo'] == 1]
            df_pca_tiempo_2 = df_pca[df_pca['Tiempo'] == 2]
            # Crear un histograma
            median1 = np.nanmedian(df_pca_tiempo_1[var_sig])
            median2 = np.nanmedian(df_pca_tiempo_2[var_sig])
            mean1 = np.nanmean(df_pca_tiempo_1[var_sig])
            mean2 = np.nanmean(df_pca_tiempo_2[var_sig])
            hist1, bins1, m = plt.hist(df_pca_tiempo_1[var_sig], edgecolor='black', color = color_1, alpha = 0.7, label = 't1', zorder = 10)
            hist2, bins2, m = plt.hist(df_pca_tiempo_2[var_sig], edgecolor='black', color = color_2, alpha = 0.7, label = 't2', zorder = 10)
            max_ylim = max(max(hist1), max(hist2))
            #plt.vlines(median1, 0, max_ylim, color='black', linestyle='-', linewidth=2, label = "Tiempo 1")
            #plt.vlines(median2, 0, max_ylim, color='red', linestyle='-', linewidth=2,  label = "Tiempo 2")
            plt.vlines(mean1, 0, max_ylim, color= darkened_color_1, linestyle='--', linewidth=3, zorder = 10)
            plt.vlines(mean2, 0, max_ylim, color= darkened_color_2, linestyle='--', linewidth=3, zorder = 10)

            nro = re.search(r'\d+', var_sig).group()
            
            plt.yticks([0, 3, 6, 9, 12])
            plt.gca().set_zorder(10)
            plt.xlabel(f"Componente principal {nro}")
            plt.ylabel("Cuentas")
            #plt.title(f'{clave_correspondiente}')
            plt.legend(fontsize = 15)
            plt.grid(True, zorder = 5)
            plt.subplots_adjust(bottom=0.15, left=0.12)
            plt.show()
            
            path_image_hist_png = path_image + f"hist_{clave_correspondiente}_{var_sig}.png"
            path_image_hist_sinfondo_png = path_image + f"hist_{clave_correspondiente}_{var_sig}_sinfondo.png"
            path_image_hist_pdf = path_image + f"hist_{clave_correspondiente}_{var_sig}.pdf"
            if save_image_con_fondo == True:
                plt.savefig(path_image_hist_png)
                plt.savefig(path_image_hist_pdf)
            else:
                plt.savefig(path_image_hist_sinfondo_png, transparent = True)
                
            
            
            if var_sig in ["PC1", "PC2"]: #PC3 las que siguen
                if var_sig == "PC1":
                    var1 = vars_sig_PC1[0]
                    var2 = vars_sig_PC1[1]
                    var3 = vars_sig_PC1[2]
                    var4 = vars_sig_PC1[3]
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
                
                fig, axs = plt.subplots(4, 1, figsize=(8, 6))

                # Figura 1
                ax1 = axs[0]
                boxplot1_1 = ax1.boxplot(data1, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_1, linewidth=3), boxprops=dict(linewidth=3, color='black'))
                boxplot1_2 = ax1.boxplot(data2, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_2, linewidth=3), boxprops=dict(linewidth=3, color='black'))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot1_1, boxplot1_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter1_1 = ax1.scatter(data1, np.random.uniform(low=0.9, high=1.1, size=len(data1)), color= color_1, alpha=0.5)
                scatter1_2 = ax1.scatter(data2, np.random.uniform(low=1.9, high=2.1, size=len(data2)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax1.set_yticks([1, 2])
                ax1.set_yticklabels(['t1', 't2'])
                ax1.set_xlabel(f'{var1} para {clave_correspondiente}')

                # Figura 2
                ax2 = axs[1]
                boxplot2_1 = ax2.boxplot(data3, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_1, linewidth=3), boxprops=dict(linewidth=3, color='black'))
                boxplot2_2 = ax2.boxplot(data4, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_2, linewidth=3), boxprops=dict(linewidth=3, color='black'))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot2_1, boxplot2_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter2_1 = ax2.scatter(data3, np.random.uniform(low=0.9, high=1.1, size=len(data3)), color= color_1, alpha=0.5)
                scatter2_2 = ax2.scatter(data4, np.random.uniform(low=1.9, high=2.1, size=len(data4)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax2.set_yticks([1, 2])
                ax2.set_yticklabels(['t1', 't2'])
                ax2.set_xlabel(f'{var2} para {clave_correspondiente}')
                
                # Figura 3
                ax3 = axs[2]
                boxplot3_1 = ax3.boxplot(data5, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_1, linewidth=3), boxprops=dict(linewidth=3, color='black'))
                boxplot3_2 = ax3.boxplot(data6, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_2, linewidth=3), boxprops=dict(linewidth=3, color='black'))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot3_1, boxplot3_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter3_1 = ax3.scatter(data5, np.random.uniform(low=0.9, high=1.1, size=len(data5)), color= color_1, alpha=0.5)
                scatter3_2 = ax3.scatter(data6, np.random.uniform(low=1.9, high=2.1, size=len(data6)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax3.set_yticks([1, 2])
                ax3.set_yticklabels(['t1', 't2'])
                ax3.set_xlabel(f'{var3} para {clave_correspondiente}')

                # Figura 4
                ax4 = axs[3]
                boxplot4_1 = ax4.boxplot(data7, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_1, linewidth=3), boxprops=dict(linewidth=3, color='black'))
                boxplot4_2 = ax4.boxplot(data8, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_2, linewidth=3), boxprops=dict(linewidth=3, color='black'))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot4_1, boxplot4_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter4_1 = ax4.scatter(data7, np.random.uniform(low=0.9, high=1.1, size=len(data7)), color= color_1, alpha=0.5)
                scatter4_2 = ax4.scatter(data8, np.random.uniform(low=1.9, high=2.1, size=len(data8)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax4.set_yticks([1, 2])
                ax4.set_yticklabels(['t1', 't2'])
                ax4.set_xlabel(f'{var4} para {clave_correspondiente}')   

                # Ajustar el diseño para evitar solapamientos
                plt.tight_layout()

                # Mostrar las figuras
                plt.show()
                
                path_image_boxplot_png = path_image + f"boxplot_{clave_correspondiente}_{var_sig}.png"
                path_image_boxplot_sinfondo_png = path_image + f"boxplot_{clave_correspondiente}_{var_sig}_sinfondo.png"
                path_image_boxplot_pdf = path_image + f"boxplot_{clave_correspondiente}_{var_sig}.pdf"
                if save_image_con_fondo == True:
                    plt.savefig(path_image_boxplot_png)
                    plt.savefig(path_image_boxplot_pdf)
                else:
                    plt.savefig(path_image_boxplot_sinfondo_png, transparent = True)
            
            if var_sig in ["PC3", "PC4"]:
                if var_sig == "PC3":
                    var1 = vars_sig_PC3[0]
                    var2 = vars_sig_PC3[1]
                    data1 = df_vars_tiempo_1[vars_sig_PC3[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC3[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC3[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC3[1]]
                
                if var_sig == "PC4":
                    var1 = vars_sig_PC4[0]
                    var2 = vars_sig_PC4[1]
                    data1 = df_vars_tiempo_1[vars_sig_PC4[0]]
                    data2 = df_vars_tiempo_2[vars_sig_PC4[0]]
                    data3 = df_vars_tiempo_1[vars_sig_PC4[1]]
                    data4 = df_vars_tiempo_2[vars_sig_PC4[1]]
                    
                fig, axs = plt.subplots(2, 1, figsize=(8, 6))

                # Figura 1
                ax1 = axs[0]
                boxplot1_1 = ax1.boxplot(data1, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_1, linewidth=2))
                boxplot1_2 = ax1.boxplot(data2, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color= darkened_color_2, linewidth=2))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot1_1, boxplot1_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter1_1 = ax1.scatter(data1, np.random.uniform(low=0.9, high=1.1, size=len(data1)), color= color_1, alpha=0.5)
                scatter1_2 = ax1.scatter(data2, np.random.uniform(low=1.9, high=2.1, size=len(data2)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax1.set_yticks([1, 2])
                ax1.set_yticklabels(['t1', 't2'])
                ax1.set_xlabel(f'{var1} para {clave_correspondiente}')

                # Figura 2
                ax2 = axs[1]
                boxplot2_1 = ax2.boxplot(data3, positions=[1], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color = darkened_color_1, linewidth=2))
                boxplot2_2 = ax2.boxplot(data4, positions=[2], widths=0.6, patch_artist=True, vert=False, medianprops=dict(color = darkened_color_2, linewidth=2))

                # Personalizar los boxplots desactivando el relleno
                for box in [boxplot2_1, boxplot2_2]:
                    for patch in box['boxes']:
                        patch.set_facecolor('none')  # Color de relleno desactivado

                # Agregar scatter plot para los puntos de fondo
                scatter2_1 = ax2.scatter(data3, np.random.uniform(low=0.9, high=1.1, size=len(data3)), color= color_1, alpha=0.5)
                scatter2_2 = ax2.scatter(data4, np.random.uniform(low=1.9, high=2.1, size=len(data4)), color= color_2, alpha=0.5)

                # Configurar ejes y etiquetas
                ax2.set_yticks([1, 2])
                ax2.set_yticklabels(['t1', 't2'])
                ax2.set_xlabel(f'{var2} para {clave_correspondiente}')

                # Ajustar el diseño para evitar solapamientos
                plt.tight_layout()

                # Mostrar las figuras
                plt.show()
                
                path_image_boxplot_png = path_image + f"boxplot_{clave_correspondiente}_{var_sig}.png"
                path_image_boxplot_pdf = path_image + f"boxplot_{clave_correspondiente}_{var_sig}.pdf"
                if save_image_con_fondo == True:
                    plt.savefig(path_image_boxplot_png)
                    plt.savefig(path_image_boxplot_pdf)
                else:
                    plt.savefig(path_image_boxplot_sinfondo_png, transparent = True)
                
                
                path_imag_hist_fondo = [path_image_hist_png]#, path_image_hist_pdf]
                path_imag_boxplot_fondo = [path_image_boxplot_png]#, path_image_boxplot_pdf]
                path_im_tot_fondo = [f"{clave_correspondiente}_{var_sig}.png"]#, f"{clave_correspondiente}_{var_sig}.pdf"]
                # Leer las imagenes
                if save_image_con_fondo == True:
                    for w in range(len(path_imag_hist_fondo)):
                        imagen1 = imread(path_imag_hist_fondo[w])
                        imagen2 = imread(path_imag_boxplot_fondo[w])
                
                        # Crear una figura con dos subgráficos (uno al lado del otro)
                        plt.figure(figsize=(10, 5))  # Puedes ajustar el tamaño de la figura según tus necesidades
                        
                        # Subgráfico 1
                        plt.subplot(1, 2, 1)
                        plt.imshow(imagen1)
                        plt.axis('off')  # Desactivar los ejes para mejorar la presentación
                        
                        # Subgráfico 2
                        plt.subplot(1, 2, 2)
                        plt.imshow(imagen2)
                        plt.axis('off')  # Desactivar los ejes para mejorar la presentación
                        
                        plt.subplots_adjust(wspace=0.05)
                        
                        # Mostrar la figura con ambas imágenes una al lado de la otra
                        plt.show()
                        
                        plt.savefig(path_image_final + path_im_tot_fondo[w])
                else:
                    imagen1 = imread(path_image_hist_sinfondo_png)
                    imagen2 = imread(path_image_boxplot_sinfondo_png)
                    
                    # Crear una figura con dos subgráficos (uno al lado del otro)
                    plt.figure(figsize=(10, 5))  # Puedes ajustar el tamaño de la figura según tus necesidades
                    
                    # Subgráfico 1
                    plt.subplot(1, 2, 1)
                    plt.imshow(imagen1)
                    plt.axis('off')  # Desactivar los ejes para mejorar la presentación
                    
                    # Subgráfico 2
                    plt.subplot(1, 2, 2)
                    plt.imshow(imagen2)
                    plt.axis('off')  # Desactivar los ejes para mejorar la presentación
                    
                    plt.subplots_adjust(wspace=0.05)
                    # Mostrar la figura con ambas imágenes una al lado de la otra
                    plt.show()
                    
                    plt.savefig(path_image_final + f"{clave_correspondiente}_{var_sig}_sinfondo.png")
                



