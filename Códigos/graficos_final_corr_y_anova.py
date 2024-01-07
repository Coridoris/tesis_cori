# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:52:00 2023

@author: corir
"""

#%% librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as npl
from scipy.stats import sem
import ast 

#para ANOVA
import pingouin as pg
from scipy.stats import f
#post ANOVA --> tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t

#%% el santo trial

entrevista = 'Primera'

nro_sujetos = 65

Sujetos = ['0']*nro_sujetos
for j in range(nro_sujetos):
    Sujetos[j] = f"Sujeto {j+1}"
    
temas = ["cfk", "campeones_del_mundo", "antesdevenir", "presencial", "arabia"]
condicion = ("CFK", "Campeones", "Filler", "Presencial", "Arabia")
temas_label = ["CFK", "Campeones", "Filler", "Presencial", "Arabia"]

temas = ["cfk",  "arabia", "campeones_del_mundo", "presencial", "antesdevenir"]
condicion = ("CFK", "Arabia", "Campeones", "Presencial", "Control")
temas_label = ["CFK", "Arabia", "Campeones", "Presencial", "Control"]

eliminando_outliers = True

control = 4 #5 seria con control, 4 no

#%% funciones


def valor_critico_t(grados_de_libertad):
    alpha = 0.05/10 #Nivel de significancia
    return t.ppf(1 - alpha / 2, df=grados_de_libertad)


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


def box_plot4o5(data1, cantidad, ylabel, save, path_imagenes = None, color = None, scatter = True):
    
    npl.rcParams['figure.figsize'] = [15, 10]
    npl.rcParams["axes.labelsize"] = 30
    npl.rcParams['xtick.labelsize'] = 30
    npl.rcParams['ytick.labelsize'] = 30
    
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3],
            'Control': data1[4]}        
    
    df = pd.DataFrame.from_dict(datos)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]
    
    if color != None:
        
        colors = color

    
    if scatter == True:
        # Personalizar los bordes del boxplot y quitar el relleno
        boxprops = {'edgecolor': 'black', 'linewidth': 2.5, 'fill': False}
        #personalizar la media bigotes
        medianprops = {'color': 'black', 'linewidth': 2.5}
        whiskerprops = {'color': 'black', 'linewidth': 2}
        #outliers
        #flierprops = {'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8}
        #barrita post bigote
        capprops = {'color': 'black', 'linewidth': 2}
        
                
        # Añadir un scatter plot sobre el boxplot con jitter
        sns.stripplot(data=df, palette=colors, size=8, edgecolor='black', linewidth=0.5, alpha=0.7, jitter=0.3, zorder = 0)
        

        # Crear el boxplot con personalizaciones
        sns.boxplot(data=df, palette=colors, boxprops=boxprops, width=0.75, zorder=30,
                    medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, sym='')


    else:
        # Crear el boxplot utilizando seaborn
        sns.boxplot(data=df, palette = colors)
        
    plt.ylabel(ylabel, fontsize = 30)
    plt.tick_params(axis='x', labelsize=30)
    
    plt.xticks(fontsize=30) 
    plt.yticks(fontsize=25) 
    
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
    
    plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
    plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
    plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')
    
    #plt.close()
    
    return plt.gcf()


def box_subplot4o5(data1, cantidad, ylabel, lettersize = 20, abreviar = False, color = None, scatter = True, ax = None, text_str = None, text_y = 0.94, text_x = 0.08):

    npl.rcParams['figure.figsize'] = [10, 10]
    npl.rcParams["axes.labelsize"] = lettersize
    npl.rcParams['xtick.labelsize'] = lettersize
    npl.rcParams['ytick.labelsize'] = lettersize
    
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3],
            'Control': data1[4]}    
        
    if abreviar != False:
        
        if cantidad == 4:
            datos = {
                'CFK': data1[0],
                'Arabia': data1[1],
                'Camp.': data1[2],
                'Pres.': data1[3]}
        elif cantidad == 5:
            datos = {
                'CFK': data1[0],
                'Arabia': data1[1],
                'Camp.': data1[2],
                'Pres.': data1[3],
                'Control': data1[4]}  
        
    
    df = pd.DataFrame.from_dict(datos)
    
    colors = [color_celeste, color_celestito, color_palido, color_violeta, color_gris]
    
    if color != None:
        
        colors = color

    ax.tick_params(axis='x', labelsize=lettersize - 3)
    ax.tick_params(axis='y', labelsize=lettersize - 3)
    
    if scatter == True:
        # Personalizar los bordes del boxplot y quitar el relleno
        boxprops = {'edgecolor': 'black', 'linewidth': 2.5, 'fill': False}
        #personalizar la media bigotes
        medianprops = {'color': 'black', 'linewidth': 3.5}
        whiskerprops = {'color': 'black', 'linewidth': 2}
        #outliers
        #flierprops = {'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8}
        #barrita post bigote
        capprops = {'color': 'black', 'linewidth': 2}

        
        # Añadir un scatter plot sobre el boxplot con jitter
        ax = sns.stripplot(data=df, palette=colors, size=8, edgecolor='black', linewidth=0.5, alpha=0.7, jitter=0.3, zorder = 0, ax = ax)
        
        
        # Crear el boxplot con personalizaciones
        ax = sns.boxplot(data=df, palette=colors, boxprops=boxprops, width=0.75, zorder=30,
                    medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, sym='', ax=ax)

        
        
        plt.close()
        
    else:
        # Crear el boxplot utilizando seaborn
        ax = sns.boxplot(data=df, palette = colors, ax = ax)
   
    if text_str != None:
        #text_x = 0.07  # Adjust the x-coordinate for horizontal positioning
        #text_y = 0.94  # Adjust the y-coordinate for vertical positioning
    
        ax.text(text_x, text_y, text_str, transform=ax.transAxes,
                fontsize= lettersize+4, color='k', ha='right', va='bottom', zorder =30)   
    
    ax.set_ylabel(ylabel, fontsize = lettersize)
    ax.tick_params(axis='x', labelsize=lettersize)
    
    return ax


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
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3],
            'Control': data1[4]}        
    
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
    
    npl.rcParams['figure.figsize'] = [10, 12]
    npl.rcParams["axes.labelsize"] = 23
    npl.rcParams['xtick.labelsize'] = 23
    npl.rcParams['ytick.labelsize'] = 23
    
    plt.figure()
    
    if cantidad == 4:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3]}
    elif cantidad == 5:
        datos = {
            'CFK': data1[0],
            'Arabia': data1[1],
            'Campeones': data1[2],
            'Presencial': data1[3],
            'Control': data1[4]}        
    
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


#%%colores y cosas imagenes
#path_imagenes = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Graficos'

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

rainbow_palette = sns.color_palette("rainbow", n_colors=7)

#rainbow_palette = sns.color_palette("autumn_r", n_colors=2)



# Asignar colores a las variables
color_1 = rgb_to_hex(rainbow_palette[0])
color_2 = rgb_to_hex(rainbow_palette[2])
color_3 = rgb_to_hex(rainbow_palette[4])
color_4 = rgb_to_hex(rainbow_palette[5])
color_5 = rgb_to_hex(rainbow_palette[6])


colores = [color_1, color_2, color_3, color_4, color_5]

color_campeones = color_1
color_presencial = color_2
color_cfk = color_3
color_arabia = color_4
color_filler = color_5

sent_palette = sns.color_palette("GnBu", n_colors=5)

#gist_stern_r 2 3 4
#mako_r 2 3 4 

colorsent_1 = rgb_to_hex(sent_palette[0])
colorsent_2 = rgb_to_hex(sent_palette[1])
colorsent_3 = rgb_to_hex(sent_palette[2])

colorsent = [colorsent_1,colorsent_2, colorsent_3]

#darkened_color_2 = sns.dark_palette(color_2)


# darkened_color_1 = lighten_color_hex(color_1, factor = 0.4)
# darkened_color_2 = lighten_color_hex(color_2, factor = 0.4)

# # Oscurecer los colores
# darkened_color_1 = darken_color(color_1)
# darkened_color_2 = darken_color(color_2, factor = 0.95)


#%% datos

if eliminando_outliers == True:
    path_sinautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_sinautopercepcion_sinoutliers_todos_temas.csv'

    path_conautopercepcion_todas = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_todos_temas.csv'


df_del_tema = []

for tema in temas:    

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ELcsv/ELcsv_conautopercepcion_{tema}.csv'
    
    df_del_tema.append(pd.read_csv(path))
    
df_todas = pd.read_csv(path_conautopercepcion_todas)
    
#%% graficos de boxplot, violinplot o puntos

variables_label = list(df_todas.columns)
'''
['Sujetos','Condición','Recuerdo_autop','Valencia_autop', 'Intensidad_autop', 'ValeInt_autop', 
 'num_palabras_unicas_norm', 'primera_persona_norm', 'tercera_persona_norm', 'num noun norm', 
 'num verb norm', 'num adj norm', 'num advs norm', 'num numeral norm', 'num propn norm', 
 'Positivo pysent', 'Negativo pysent', 'Intensidad pysent', 'Valencia pysent', 
 'Valencia e intensidad pysent', 'Valencia2 pysent', 'Valencia e intensidad2 pysent', 
 'cohe_norm_d=1', 'cohe_norm_d=2', 'cohe_norm_d=3', 'num_nodes_LSC', 'Comunidades_LSC', 
 'diámetro', 'k_mean', 'transitivity', 'ASP', 'average_CC','selfloops', 'L2', 'L3', 'density', 
 'Detalles internos norm', 'Detalles externos norm']
'''
variables = []
# for j in range(len(variables_label)):
#     variable = []
#     for i, tema in enumerate(temas):
#         variable.append(df_del_tema[i][variables_label[j]])
#     variables.append(variable)

for j in range(len(variables_label)):
    variable = []
    for i, tema in enumerate(temas):
        df_filtrado = df_todas[df_todas['Condición'] == tema]
        variable.append(df_filtrado[variables_label[j]])
    variables.append(variable)
        
#%% contenido

path_imagenes = f"C:/Users/Usuario/Desktop/Cori/Tesis/Figuras_finales/{entrevista}_entrevista/Presentacion variables"

fig, axs = plt.subplots(1,2 , sharex=False, figsize=(15,7.5))

# Crear el primer subplot (1 fila, 2 columnas, primer subplot)
#ax1 = fig.add_subplot(1, 2, 1)
axs[0] =box_subplot4o5(variables[6], control, '$Z_{score}$ del núm de palabras', text_str = "(a)", color = colores, scatter=True, ax = axs[0])

# Crear el segundo subplot (1 fila, 2 columnas, segundo subplot)
#ax2 = fig.add_subplot(1, 2, 2)
axs[1] = box_subplot4o5(variables[11], control, 'Núm. de adjetivos norm.',text_str = "(b)", color = colores, scatter=True, ax = axs[1])

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'contenido'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')

fig, axs = plt.subplots(1,3 , sharex=False, figsize=(20,7.5))

axs[0] =box_subplot4o5(variables[6], control, '$Z_{score}$ del núm de palabras', lettersize = 25, abreviar = True, text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.12, text_y = 0.93)

axs[1] = box_subplot4o5(variables[11], control, 'Núm. de adjetivos norm.', lettersize = 25, abreviar = True, text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.12, text_y = 0.93)

axs[2] = box_subplot4o5(variables[7], control, 'Palabras en primera persona norm.', lettersize = 25, abreviar = True, text_str = "(c)", color = colores, scatter=True, ax = axs[2], text_x = 0.12, text_y = 0.93)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'contenido_op2'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')

#%%
path_imagenes = f"C:/Users/Usuario/Desktop/Cori/Tesis/Figuras_finales/{entrevista}_entrevista/Presentacion variables"
box_plot4o5(variables[6], control, '$Z_{score}$ del núm de palabras', f'{entrevista}_nro_palabras_unicas', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[7], control, 'Palabras en primera persona norm.', f'{entrevista}_primera_persona', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[10], control, 'Núm. de verbos norm.', f'{entrevista}_verb', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[11], control, 'Núm. de adjetivos norm.', f'{entrevista}_adj', path_imagenes = path_imagenes, color = colores)
#%%sentimiento
box_plot4o5(variables[15], control, 'Positivo', f'{entrevista}_positivo', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[17], control, 'Intensidad', f'{entrevista}_intensidad', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[20], control, 'Valencia', f'{entrevista}_valencia', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[21], control, 'Intensidad y valencia', f'{entrevista}_intyval', path_imagenes = path_imagenes, color = colores)


#%% sentimiento subplots
fig, axs = plt.subplots(1,2 , sharex=False, figsize=(15,7.5))

axs[0] =box_subplot4o5(variables[15], control, 'Positivo', text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.08)

axs[1] = box_subplot4o5(variables[21], control, 'Intensidad y valencia',text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.08)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'sentimiento'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')

fig, axs = plt.subplots(1,3 , sharex=False, figsize=(18,7.5))

axs[0] =box_subplot4o5(variables[15], control, 'Positivo', lettersize = 25, abreviar = True, text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.13, text_y = 0.93)

axs[1] = box_subplot4o5(variables[21], control, 'Intensidad y valencia', lettersize = 25, abreviar = True, text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.13, text_y = 0.93)

axs[2] = box_subplot4o5(variables[17], control, 'Intensidad', lettersize = 25, abreviar = True, text_str = "(c)", color = colores, scatter=True, ax = axs[2], text_x = 0.12, text_y = 0.93)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'sentimiento_op2'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')


#%%coherencia
box_plot4o5(variables[22], control, 'Coherencia a d = 1', f'{entrevista}_cohed1', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[23], control, 'Coherencia a d = 2', f'{entrevista}_cohed2', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[24], control, 'Coherencia a d = 3', f'{entrevista}_cohed3', path_imagenes = path_imagenes, color = colores)
#%%redes
box_plot4o5(variables[25], control, 'Núm. nodos', f'{entrevista}_nodos', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[26], control, 'Núm. comunidades', f'{entrevista}_comunidades', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[28], control, 'Grado medio', f'{entrevista}_kmean', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[30], control, 'Camino mas corto promedio', f'{entrevista}_ASP', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[31], control, 'Coeficiente de clustering promedio', f'{entrevista}_averageCC', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[32], control, 'Selfloops', f'{entrevista}_selfloops', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[33], control, 'Loops de dos', f'{entrevista}_L2', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[34], control, 'Loops de tres', f'{entrevista}_L3', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[35], control, 'Densidad', f'{entrevista}_densidad', path_imagenes = path_imagenes, color = colores)

#%% estructurales y memoria

fig, axs = plt.subplots(1,2 , sharex=False, figsize=(15,7.5))

axs[0] =box_subplot4o5(variables[25], control, 'Núm. nodos', text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.08)

axs[1] = box_subplot4o5(variables[26], control, 'Núm. comunidades',text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.08)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'estructurales1'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')

fig, axs = plt.subplots(1,3 , sharex=False, figsize=(18,7.5))

axs[0] =box_subplot4o5(variables[24], control, 'Coherencia a d = 3', lettersize = 25, abreviar = True, text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.12, text_y = 0.93)

axs[1] = box_subplot4o5(variables[25], control, 'Núm. nodos', lettersize = 25, abreviar = True, text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.12, text_y = 0.93)

axs[2] = box_subplot4o5(variables[26], control, 'Núm. comunidades', lettersize = 25, abreviar = True, text_str = "(c)", color = colores, scatter=True, ax = axs[2], text_x = 0.12, text_y = 0.93)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'estructurales1op2'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')


box_plot4o5(variables[36], control, 'Detalles internos', 'memoria', path_imagenes = path_imagenes, color = colores)

fig, axs = plt.subplots(1,3 , sharex=False, figsize=(18,7.5))

axs[0] =box_subplot4o5(variables[35], control, 'Densidad', lettersize = 25, abreviar = True, text_str = "(a)", color = colores, scatter=True, ax = axs[0], text_x = 0.13, text_y = 0.93)

axs[1] = box_subplot4o5(variables[31], control, 'Coeficiente de clustering promedio', lettersize = 25, abreviar = True, text_str = "(b)", color = colores, scatter=True, ax = axs[1], text_x = 0.13, text_y = 0.93)

axs[2] = box_subplot4o5(variables[36], control, 'Detalles internos', lettersize = 25, abreviar = True, text_str = "(c)", color = colores, scatter=True, ax = axs[2], text_x = 0.12, text_y = 0.93)

plt.tight_layout()

# Mostrar el gráfico
plt.show()

save = 'estructurales2op2'

plt.savefig(path_imagenes + f'/{save}_boxplot_transparente.png', transparent = True)
plt.savefig(path_imagenes + f'/{save}_boxplot.png', transparent = False)
plt.savefig(path_imagenes + f'/{save}_boxplot.pdf')





#%% memoria
box_plot4o5(variables[36], control, 'Detalles internos', f'{entrevista}_internos', path_imagenes = path_imagenes, color = colores)
box_plot4o5(variables[37], control, 'Detalles externos', f'{entrevista}_externos', path_imagenes = path_imagenes, color = colores)


#%% gráfico con barras una al lado de otra de pysentimiento

colores = ["#011638", "#364156", "#9FA0A4"]

colores = colorsent
control = 4 #asi no tiene control, sino va 5
grupo1_mean = np.nanmean(variables[17][:control], axis = 1)
grupo2_mean = np.nanmean(variables[16][:control], axis = 1)
grupo3_mean = np.nanmean(variables[15][:control], axis = 1)

grupo1_std = sem(variables[17][:control], axis=1, nan_policy='omit')
grupo2_std = sem(variables[16][:control], axis=1, nan_policy='omit')
grupo3_std = sem(variables[15][:control], axis=1, nan_policy='omit')

detalles = {
    'Intensidad': tuple(grupo1_mean),
    'Negativo': tuple(grupo2_mean),
    'Positivo': tuple(grupo3_mean)
}

x = np.arange(len(condicion[:control]))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (15,10))

for (attribute, measurement, std_dev, color) in zip(detalles.keys(), detalles.values(), [grupo1_std, grupo2_std, grupo3_std], colores):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=std_dev, capsize=5, color=color)
    multiplier += 1
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probabilidad', fontsize =25)
ax.set_xticks(x + width, condicion[:control])
ax.tick_params(labelsize = 25)
ax.legend(loc='upper right', fontsize = 20)
plt.tight_layout()
plt.show()

plt.savefig(path_imagenes + '/pysentimiento_tresbarras_transparente.png', transparent=True)
plt.savefig(path_imagenes + '/pysentimiento_tresbarras.png')
plt.savefig(path_imagenes + '/pysentimiento_tresbarras.pdf')

#%% preeliminares del grafico coherencia promedio vs cada x cantidad de oraciones
coherence1 = []
coherence2 = []
coherence3 = []
coherence4 = []
coherence5 = []
coherence1_err = []
coherence2_err = []
coherence3_err = []
coherence4_err = []
coherence5_err = []

for i in range(len(temas)):
    tema =  temas[i]
    
    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion.csv'
    
    df_coherence_evolution = pd.read_csv(path)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a listas
    df_coherence_evolution['coherencia_evolucion'] = df_coherence_evolution['coherencia_evolucion'].apply(ast_literal_eval_notnan)
    

    coherence1.append(sum(v[0] for v in df_coherence_evolution['coherencia_evolucion'].dropna()) / len(df_coherence_evolution.dropna()))
    valores1 = [v[0] if type(v) == list else np.nan for v in df_coherence_evolution['coherencia_evolucion']] 
    coherence1_err.append(np.std(valores1)/np.sqrt(len(df_coherence_evolution)))
    coherence2.append(sum(v[1] for v in df_coherence_evolution['coherencia_evolucion'].dropna()) / len(df_coherence_evolution.dropna()))
    valores2 = [v[1] if type(v) == list else np.nan for v in df_coherence_evolution['coherencia_evolucion']]
    coherence2_err.append(np.std(valores2)/np.sqrt(len(df_coherence_evolution)))
    valid_vectors3 = [v for v in df_coherence_evolution['coherencia_evolucion'].dropna() if len(v) >= 3]
    coherence3.append(sum(v[2] for v in valid_vectors3) / len(valid_vectors3))
    valores3 = [v[2] if type(v) == list else np.nan for v in valid_vectors3]
    coherence3_err.append(np.std(valores3)/np.sqrt(len(valid_vectors3)))
    valid_vectors4 = [v for v in df_coherence_evolution['coherencia_evolucion'].dropna() if len(v) >= 4]
    coherence4.append(sum(v[3] for v in valid_vectors4) / len(valid_vectors4))
    valores4 = [v[3] if type(v) == list else np.nan for v in valid_vectors4]
    coherence4_err.append(np.std(valores4)/np.sqrt(len(valid_vectors4)))
    valid_vectors5 = [v for v in df_coherence_evolution['coherencia_evolucion'].dropna() if len(v) >= 5]
    coherence5.append(sum(v[4] for v in valid_vectors5) / len(valid_vectors5))
    valores5 = [v[4] if type(v) == list else np.nan for v in valid_vectors5]
    coherence5_err.append(np.std(valores5)/np.sqrt(len(valid_vectors5)))
 
#modelo nulo

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)
#hice mil mezclas de textos de cada sujeto y le calculé la coherencia
coherence1_random = []
coherence2_random = []
coherence3_random = []
coherence4_random = []
coherence5_random = []

for i in range(0,1000):

    df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
    # Calcular el promedio del primer elemento de los vectores
    coherence1_random.append(sum(v[0] for v in df_nulo[f'coherencia_evolucion_de_random_{i}']) / len(df_nulo))
    
#buscamos los cuantiles

q_coherence_rand_1 = np.percentile(coherence1_random, 95)

print("cuartil 95 cohe1rand:", q_coherence_rand_1)
#%%#el grafico

# Crear una figura con dos subgráficos
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

color_campeones = color_celeste
color_presencial = color_celestito
color_cfk = color_palido
color_arabia = color_violeta
color_filler = color_gris

color_violeta_fuerte = "#6A4480"

t = np.linspace(1, 5, 5)
coherence_por_cant_oraciones = np.array([np.array(coherence1), np.array(coherence2), np.array(coherence3), np.array(coherence4), np.array(coherence5)])
#coherence1 es un vector que en el primer elemento tiene la coherencia entre oraciones contiguas de 
#para el tema 0, en el segundo elemento para el tema 1, en el tercero para el tema 2 y asi
coherence_por_tema = np.transpose(coherence_por_cant_oraciones)
#esto debería tener en el primer elemento un vector que tenga la coherencia en los distintos t del tema 0

plt.errorbar(t, coherence_por_tema[0]/q_coherence_rand_1, yerr = coherence1_err/q_coherence_rand_1, fmt='o', color = darken_color(colores[0]), linewidth =3, zorder = 1)
plt.errorbar(t, coherence_por_tema[1]/q_coherence_rand_1, yerr = coherence2_err/q_coherence_rand_1, fmt='o', color = darken_color(colores[1]), linewidth = 3, zorder = 1)
plt.errorbar(t, coherence_por_tema[2]/q_coherence_rand_1, yerr = coherence3_err/q_coherence_rand_1, fmt='o', color = darken_color(colores[2]), linewidth = 3, zorder = 1)#, zorder = 1)
plt.errorbar(t, coherence_por_tema[3]/q_coherence_rand_1, yerr = coherence4_err/q_coherence_rand_1, fmt='o', color = darken_color(colores[3]), linewidth = 3, zorder = 1)
#plt.errorbar(t, coherence_por_tema[4]/q_coherence_rand_1, yerr = coherence5_err/q_coherence_rand_1, fmt='o', color = darken_color(colores[4]), linewidth = 3, zorder = 1)


# Crear los histogramas en cada subgráfico
ax.scatter(t, coherence_por_tema[0]/q_coherence_rand_1, s = 100, color = colores[0], label = temas_label[0], zorder = 5)
ax.scatter(t, coherence_por_tema[1]/q_coherence_rand_1, s = 100, color = colores[1], label = temas_label[1], zorder = 5)
ax.scatter(t, coherence_por_tema[2]/q_coherence_rand_1, s = 100, color = colores[2], label = temas_label[2], zorder = 5)#, zorder = 5) #edgecolors = 'k',
ax.scatter(t, coherence_por_tema[3]/q_coherence_rand_1, s = 100, color = colores[3], label = temas_label[3], zorder = 5)
#ax.scatter(t, coherence_por_tema[4]/q_coherence_rand_1, s = 100, color = colores[4], label = temas_label[4], zorder = 5)




ax.axhline(y = 1, color='k', linestyle='--', linewidth = 3, label= 'Modelo nulo')


# Etiquetas y título para cada subgráfico
ax.set_xlabel('Distancia', fontsize = 18)
ax.set_xticks([1,2,3,4,5], )
ax.tick_params(labelsize=15)
ax.set_ylabel('Coherencia', fontsize = 18)
plt.grid(True)

ax.legend(loc='upper right',fontsize = 15, bbox_to_anchor=(1.5, 1.0))


# Título general para los dos histogramas
#fig.suptitle('Coherencia en el tiempo', fontsize=16)

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

plt.savefig(path_imagenes + '/coherencia_vs_t_conmodelonulo_transparente.png', transparent = True)
plt.savefig(path_imagenes + '/coherencia_vs_t_conmodelonulo.png')
plt.savefig(path_imagenes + '/coherencia_vs_t_conmodelonulo.pdf')

# Mostrar los histogramas
plt.show()
#%% contador de cuántos sujetos superan el umbral de coherencia

'''
esta es la nomeclatura
coherence{cada_cuantas_oraciones}[nro_tema][nro_sujeto]
q_modelo_nulo[nro_sujeto]
'''   


coherence1_norm = []
coherence2_norm = []
coherence3_norm = []
coherence4_norm = []
coherence5_norm = []

for i in range(len(temas)):
    tema =  temas[i]
    path_cohe = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/{tema}_coherencia_evolucion_norm.csv'
    
    df_coherence_evolution = pd.read_csv(path_cohe)
    
    # Convertir los valores de la columna coherencia_evolucion de cadena a listas
    df_coherence_evolution['coherencia_evolucion_normalizada'] = df_coherence_evolution['coherencia_evolucion_normalizada'].apply(ast_literal_eval_notnan)

    coherence1_norm.append([v[0] if type(v) == list else np.nan for v in df_coherence_evolution['coherencia_evolucion_normalizada']])
    coherence2_norm.append([v[1] if type(v) == list else np.nan for v in df_coherence_evolution['coherencia_evolucion_normalizada']])
    coherence3_tema_norm = []
    for v in df_coherence_evolution['coherencia_evolucion_normalizada']:
        if type(v) == list:
            if len(v) >= 3:
                coherence3_tema_norm.append(v[2])
            else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
                coherence3_tema_norm.append(np.nan)
        else:
            coherence3_tema_norm.append(np.nan)
    coherence3_norm.append(coherence3_tema_norm)
    coherence4_tema_norm = []
    for v in df_coherence_evolution['coherencia_evolucion_normalizada']:
        if type(v) == list:
            if len(v) >= 4:
                coherence4_tema_norm.append(v[3])
            else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
                coherence4_tema_norm.append(np.nan)
        else:
            coherence4_tema_norm.append(np.nan)
    coherence4_norm.append(coherence4_tema_norm)
    coherence5_norm = []
    for v in df_coherence_evolution['coherencia_evolucion_normalizada']:
        if type(v) == list:
            if len(v) >= 5:
                coherence5_norm.append(v[4])
            else: #si quiero mantener el largo en 30 (para q cada sujeto tenga su posicion) tengo que hacer hacer esto
                coherence5_norm.append(np.nan)
        else:
            coherence5_norm.append(np.nan)
    coherence5_norm.append(coherence5_norm)
    
#modelo nulo

path = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/Coherencia/coherencia_evolucion_random.csv'

df_nulo = pd.read_csv(path)

modelo_nulo = []
q_modelo_nulo = []
for j in range(0, len(df_nulo)): #recorre sobre sujetos
    modelo_nulo_sujeto_j = []
    for i in range(0,1000): #recorre sobre las mil iteraciones de modelo nulo
        if j == 0:
            df_nulo[f'coherencia_evolucion_de_random_{i}'] = df_nulo[f'coherencia_evolucion_de_random_{i}'].apply(ast.literal_eval)
        modelo_nulo_sujeto_j.append(df_nulo[f'coherencia_evolucion_de_random_{i}'][j][0])
    modelo_nulo.append(modelo_nulo_sujeto_j)
    q_modelo_nulo_j  = np.percentile(modelo_nulo_sujeto_j, 95)
    q_modelo_nulo.append(q_modelo_nulo_j)

count = 0
count_camp = 0
sujetos_camp = []
count_pres = 0
sujetos_pres = []
count_cfk = 0
sujetos_cfk = []
count_arab = 0
sujetos_arab = []
count_antes = 0
sujetos_antes = []
sujeto_coherente = []
for j in range(len(Sujetos)):
    for i in range(len(temas)):
        if coherence1_norm[i][j] > 1:
            count += 1
            sujeto_coherente.append(j)
            if i == 0:
                count_camp += 1
                sujetos_camp.append(j)
            elif i == 1:
                count_pres += 1
                sujetos_pres.append(j)
            elif i == 2:
                count_cfk += 1
                sujetos_cfk.append(j)
            elif i == 3:
                count_arab += 1
                sujetos_arab.append(j)
            elif i == 4:
                count_antes += 1
                sujetos_antes.append(j)
                
            
count_tema_norm = {"campeones_del_mundo": count_camp, "presencial": count_pres, "cfk": count_cfk, "arabia": count_arab, "antesdevenir": count_antes}

sujetos_tema_norm = {"campeones_del_mundo": sujetos_camp, "presencial": sujetos_pres, "cfk": sujetos_cfk, "arabia": sujetos_arab, "antesdevenir": sujetos_antes}


#%% detalles internos y externos, barras una al lado de otra

colores = ["#13505B", "#9FA0A4"]

detalles_int_x_tema = np.nanmean(variables[32], axis = 1)
detalles_ext_x_tema = np.nanmean(variables[33], axis = 1)

detalles_int_x_tema_err = sem(variables[32], axis=1, nan_policy='omit')
detalles_ext_x_tema_err = sem(variables[33], axis=1, nan_policy='omit')

detalles = {
    'Detalles internos': tuple(detalles_int_x_tema),
    'Detalles externos': tuple(detalles_ext_x_tema),
}

x = np.arange(len(condicion))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (11,6))

for i, (attribute, measurement, err, color) in enumerate(zip(detalles.keys(), detalles.values(), [detalles_int_x_tema_err, detalles_ext_x_tema_err], colores)):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=err, capsize=5, color=color)  # Asigna el color personalizado
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Número de detalles normalizado', fontsize =15)
ax.set_xticks(x + width, condicion)
ax.tick_params(labelsize = 15)
ax.legend(loc='upper right', fontsize = 13)
plt.savefig(path_imagenes+'/' + 'detalles_int_ext', transparent=True)
plt.show()

#%% primera y tercera persona grafico barras

colores = ["#13505B", "#9FA0A4"]

primera = np.nanmean(variables[7], axis = 1)
tercera = np.nanmean(variables[8], axis = 1)

primera_err = sem(variables[7], axis=1, nan_policy='omit')
tercera_err = sem(variables[8], axis=1, nan_policy='omit')

detalles = {
    'Primera persona': tuple(primera),
    'Tercera persona': tuple(tercera),
}

x = np.arange(len(condicion))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (11,6))

for i, (attribute, measurement, err, color) in enumerate(zip(detalles.keys(), detalles.values(), [primera_err, tercera_err], colores)):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=err, capsize=5, color=color)  # Asigna el color personalizado
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Núm. de palabras primera/tercera normalizado', fontsize =15)
ax.set_xticks(x + width, condicion)
ax.tick_params(labelsize = 15)
ax.legend(loc='upper right', fontsize = 13)
plt.savefig(path_imagenes+'/' + 'primera_tercera_persona', transparent=True)
plt.show()


#%% ANOVA

#path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/ELcsv nuevo/ELcsv_conautopercepcion_sinoutliers_dostiempos.csv'

df = pd.read_csv(path_conautopercepcion_todas)

df = df.drop(['Valencia pysent', 'Valencia e intensidad pysent'], axis = 1)

cond_elim = ['antesdevenir']

tiempo_elim = [4]

df_sin_filler = df[~df['Condición'].isin(cond_elim)]

#df = df[~df['Tiempo'].isin(tiempo_elim)]

#condiciones_a_eliminar = ['campeones_del_mundo', 'antesdevenir']

# Filtramos las condiciones que creo que dan las diferencias de las medias
#df_sin_camp_ni_filler = df[~df['Condición'].isin(condiciones_a_eliminar)]

variables_dependientes = list(df_sin_filler.columns)[2:]

#aov = pg.rm_anova(dv = 'Nro palabras únicas', within = 'Condición', subject='Sujetos', data=df, detailed=True, effsize="np2")

vars_1t = ['num_palabras_unicas_norm', 'num adj norm', 'primera_persona_norm','Positivo pysent',
           'Valencia e intensidad2 pysent',  'Intensidad pysent', 'cohe_norm_d=3', 'num_nodes_LSC',
           'Comunidades_LSC', 'density', 'average_CC', 'Detalles internos norm']
           
var = vars_1t[4]

print(var)

aov = df_sin_filler.rm_anova(dv = var, within='Condición', subject='Sujetos',  detailed=False)

print(aov)

# Definir los grados de libertad del numerador y del denominador
df_between = aov['ddof1'][0]  # Grados de libertad del numerador
df_within = aov['ddof2'][0]   # Grados de libertad del denominador

# Definir el nivel de significancia (alfa)
alfa = 0.05

# Calcular el valor crítico de F
f_critical = f.ppf(1 - alfa, df_between, df_within)

print(f"Valor crítico de F: {f_critical}")

print(f"Valor de F: {aov['F'][0]}")

print(f"Valor de p: {aov['p-unc'][0]}")

print(f"Valor de epsilon: {aov['eps'][0]}")

print(f"Valor de eta es: {aov['ng2']}")

if 'p-GG-corr' in np.array(aov.columns):
    print(f"Valor de p corregido: {aov['p-GG-corr'][0]}")
    
comparaciones = 4*3/2 # n(n-1)/2 donde n es el número de grupos

posthoc_result = pg.pairwise_tests(dv= var, within='Condición', subject='Sujetos', data=df_sin_filler) #alpha=0.05/comparaciones correction='bonferroni' DA LO MISMO PONER ESTO
# Calcula el valor crítico de T para una prueba t de dos colas
posthoc_result['T critico'] = posthoc_result['dof'].apply(valor_critico_t)
posthoc_result['p pasa'] = np.where(posthoc_result['p-unc'] < 0.05/comparaciones, True, np.nan)
posthoc_result['t pasa'] = np.where(abs(posthoc_result['T']) > posthoc_result['T critico'], True, np.nan)


sin_nans = posthoc_result.dropna()

columnas_deseadas = ['A', 'B', 'p-unc']

df_seleccionado = sin_nans[columnas_deseadas]

# Ahora df_seleccionado contiene solo las columnas que especificaste
print(df_seleccionado)
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
    aov = df.rm_anova(dv = var, within='Condición', subject='Sujetos',  detailed=False)
    
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
        vars_no_sig.append(var)
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

# Guardar el DataFrame en un archivo CSV
df_resultados.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ANOVA y correlacion/ANOVAS.csv', index=False)
# para guardar sin camp ni filler busca si quiero sin campeones ni filler tengo que poner aca df_sin_camp_ni_filler en vez de df
#%% ahora el test de tukey para las variables donde dió significativa ver donde es

# Realizar la prueba de Tukey como prueba post hoc
for i, var in enumerate(vars_sig):
    print(var)
    posthoc_result = pg.pairwise_tests(dv= var, within='Condición', subject='Sujetos', data=df) #alpha=0.05/comparaciones correction='bonferroni' DA LO MISMO PONER ESTO
    # Calcula el valor crítico de T para una prueba t de dos colas
    posthoc_result['T critico'] = posthoc_result['dof'].apply(valor_critico_t)
    posthoc_result['p pasa'] = np.where(posthoc_result['p-unc'] < 0.05/10, True, np.nan)
    posthoc_result['t pasa'] = np.where(abs(posthoc_result['T']) > posthoc_result['T critico'], True, np.nan)
    #print(posthoc_result)
    print(posthoc_result[['A','B','p pasa', 't pasa']])


#%% para lo que sigue no queremos estas columnas

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df.drop(['Sujetos', 'Condición'], axis=1)

#%% correlacion

variables_corr = list(df.columns)
variables_x = [var[:6] for var in variables_corr]

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear un mapa de calor (heatmap) de la matriz de correlación
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5, xticklabels = variables_x, yticklabels = variables_corr)
plt.rc('font', size=25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
# Mostrar el mapa de calor
plt.show()

#plt.savefig(path_imagenes + "/matriz_corr.png", bbox_inches='tight', pad_inches=0, transparent=True)

# Crear listas para almacenar las variables y sus p-values asociados
variable_pairs = []
p_values = []

from scipy.stats import pearsonr

df_sinnans = df.dropna()

# Iterar a través de las filas y columnas de la matriz de correlación
for row in correlation_matrix.index:
    for col in correlation_matrix.columns:
        # Evitar duplicados y la diagonal principal
        if row != col and (col, row) not in variable_pairs:
            # Calcular la correlación de Pearson y el p-value
            corr, p_value = pearsonr(df_sinnans[row], df_sinnans[col])
            # Almacenar el par de variables y su p-value
            variable_pairs.append((row, col))
            p_values.append(p_value)

# Darle forma a las listas para que coincidan con la longitud de la matriz de correlación
variable_pairs = np.array(variable_pairs)
p_values = np.array(p_values)

#defino la significancia como 0.05 / numero de comparaciones = len(nro_variables**2 /2)
#(len(df.columns)*(len(df.columns)-1)) = len(p_values)
significancia = 0.05/len(p_values)

significativo = np.where(p_values < significancia, np.nan, False) #asi tiramos todas las significativas y dejamos las que queremos eliminar

#%
# Crear un DataFrame para mostrar las variables y sus p-values
result_df = pd.DataFrame({"Variable 1": variable_pairs[:, 0], "Variable 2": variable_pairs[:, 1], "P-Value": p_values, "Dio significativo corrigiendo p?": significativo})


result_df_sinnans = result_df.dropna()
# Mostrar el DataFrame
result_df_sinnans = result_df_sinnans[['Variable 1', 'Variable 2']] #son las que queremos eliminar en la matriz de correlacion

result_df_sinnans['Tuplas'] = list(zip(result_df_sinnans['Variable 1'], result_df_sinnans['Variable 2']))

#vars_posicion = list(correlation_matrix.columns)
#position_dict = {value: index for index, value in enumerate(vars_posicion)}

#result_df_sinnans['Variable 1'] = result_df_sinnans['Variable 1'].map(position_dict)
#result_df_sinnans['Variable 2'] = result_df_sinnans['Variable 2'].map(position_dict)

for i, var1 in enumerate(correlation_matrix):
    for j, var2 in enumerate(correlation_matrix):
        if (var1, var2) in list(result_df_sinnans['Tuplas']):
            correlation_matrix[var1][var2] = 0
            correlation_matrix[var2][var1] = 0
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5, xticklabels = variables_x, yticklabels = variables_corr)
plt.rc('font', size=25)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
# Mostrar el mapa de calor
plt.show()

plt.savefig(path_imagenes + "/matriz_corr_significativa.png", bbox_inches='tight', pad_inches=0, transparent=True)
# Guardar el DataFrame en un archivo CSV
#result_df.to_csv(f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/ANOVA y correlacion/p_val_matriz_corr.csv', index=False)
