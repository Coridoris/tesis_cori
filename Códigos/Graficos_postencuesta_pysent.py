# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:39:50 2023

@author: corir
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from scipy.stats import spearmanr
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib as npl

#%% colores y parámetros

color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

color_violeta = "#856084"



#esto es para que todos los gráficos sean iguales

#tamaño del gráfico
npl.rcParams['figure.figsize'] = [10, 10]
#se ajusta el gráfico para que no haya espacios en blanco en la figura
#npl.rcParams["figure.autolayout"]= 'true'
#tamaño de las lineas
npl.rcParams["lines.linewidth"]=2
#tamaño de letras de leyenda
npl.rcParams["legend.fontsize"]=26
#tamaño de letras en ejes
npl.rcParams["axes.labelsize"] = 26
#tamaño de los números de los ejes
npl.rcParams['xtick.labelsize'] = 26
npl.rcParams['ytick.labelsize'] = 26

#%%
def violin_plot(data1, ylabel, nombre_guardado = None):
    npl.rcParams["axes.labelsize"] = 16
    npl.rcParams['xtick.labelsize'] = 16
    npl.rcParams['ytick.labelsize'] = 16
    data = {
        'Campeones': data1[0],
        'Presencial': data1[1],
        'Cfk': data1[2],
        'Arabia': data1[3]
    }   #Filler': data1[1]
    
    df_ = pd.DataFrame(data)
    
    # Crear el violin plot utilizando catplot
    catplot = sns.catplot(data=df_, kind='violin', palette='Set2')
    
    catplot.set_ylabels(ylabel)

    # Mostrar el gráfico
    plt.show()
    
    if nombre_guardado != None:
        plt.savefig(path_imagenes + nombre_guardado, transparent=True)
    
    npl.rcParams["axes.labelsize"] = 26
    npl.rcParams['xtick.labelsize'] = 26
    npl.rcParams['ytick.labelsize'] = 26
    return 'ok'

def get_max_sentiment(diccionario):
    max_key = max(diccionario, key=diccionario.get)
    
    if max_key == 'NEU':
        return 0
    elif max_key == 'POS':
        return 1
    elif max_key == 'NEG':
        return -1

#%%
'''
Normalizamos número de palabras con el zscore = (datos - su media) / su desviación estandar
Usamos el número de palabras únicas (es equivalente a palabras totales)
'''

#path donde vamos a guardar las imagenes

path_imagenes = 'C:/Users/Usuario/Desktop/Cori/Tesis/Seminario/'

#%% correlacion nro palabras unicas y percepcion
#toda la datita que necesitas para los graficos


Sujetos = [f'Sujeto {i}' for i in range(1, 31)]

nro_sujeto = np.linspace(1, 30, 30)

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"] #"antesdevenir"

temas_tabla = ["campeones", "presencial", "cfk", "arabia"]

percepcion_recuerdo_por_tema = []
pysent_pos_por_tema = []
pysent_neg_por_tema = []
pysent_neu_por_tema = []
pysent_valencia_por_tema = []
intensidad_sentimiento_por_tema = [] #es la autopercibida
tipo_sentimiento_por_tema = [] #es la autopercibida de -1 a 1
nro_palabras_unicas_por_tema = []
nro_palabras_por_tema = []
nro_palabras_unicas_por_tema_para_normalizar = []
nro_palabras_por_tema_para_normalizar = []
intensidad_menos5_5_por_tema = [] #es la autopercibida de -5 a 5
palabra_unicas = []

intensidad_por_tema = []


palabras = ['totales', 'unicas']


for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/pysentimiento_contando_{tema}.csv'
    
    
    df = pd.read_csv(path)
    
    nro_palabras_unicas_para_normalizar = df["num_tot_palabras_unicas"]

    nro_palabras_unicas_por_tema_para_normalizar.append(nro_palabras_unicas_para_normalizar)

    nro_palabras_para_normalizar = df["num_tot_palabras"]

    nro_palabras_por_tema_para_normalizar.append(nro_palabras_para_normalizar)
    

nro_palabras_unicas_x_sujeto = list(zip(*nro_palabras_unicas_por_tema_para_normalizar))
mean_palabras_unicas_x_sujeto = np.mean(nro_palabras_unicas_x_sujeto, axis = 1)
std_palabras_unicas_x_sujeto = np.std(nro_palabras_unicas_x_sujeto, axis = 1)

nro_palabras_x_sujeto = list(zip(*nro_palabras_por_tema_para_normalizar))
mean_palabras_x_sujeto = np.mean(nro_palabras_x_sujeto, axis = 1)
std_palabras_x_sujeto = np.std(nro_palabras_x_sujeto, axis = 1)

nro_palabras_unicas_norm = []

for palab_unicas, media, std in zip(nro_palabras_unicas_x_sujeto, mean_palabras_unicas_x_sujeto, std_palabras_unicas_x_sujeto):
    nro_palabras_unicas_norm.append([(x-media) / std for x in palab_unicas])
    
nro_palabras_norm = []

for palab, media, std in zip(nro_palabras_x_sujeto, mean_palabras_x_sujeto, std_palabras_x_sujeto):
    nro_palabras_norm.append([(x-media) / std for x in palab])

for ind, tema in enumerate(temas):

    path = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Contando_y_sentimiento/pysentimiento_contando_{tema}.csv'
    
    df = pd.read_csv(path) #sentimiento y contando palabras

    #SI NO QUIERO NORMALIZAR COMENTO ESTA LINEA
    
    df["num_tot_palabras_unicas"] = (df["num_tot_palabras_unicas"]-mean_palabras_unicas_x_sujeto)/std_palabras_unicas_x_sujeto
    
    df = df.dropna()
    
    percepcion_recuerdo = df["cuanto_recordaste"]
    
    nro_palabras_unicas = df["num_tot_palabras_unicas"]
    

    intensidad_sentimiento = df["intensidad_emocion"]
    
    tipo_sentimiento = df["tipo_emocion_num"]
    
    intensidad_menos5_5 = df["intensidad_y_tipo"]

    df["pysent_curdo"] = df["pysent_curdo"].apply(eval)
    
    df['valencia_pysent'] = df['pysent_curdo'].apply(get_max_sentiment)
    
    py_pos = df["pysent_curdo"].apply(lambda x: x['POS'])
    
    py_neg = df["pysent_curdo"].apply(lambda x: x['NEG'])
    
    py_neu = df["pysent_curdo"].apply(lambda x: x['NEU'])
    
    py_int = df["pysent_curdo"].apply(lambda x: x['POS']) + df["pysent_curdo"].apply(lambda x: x['NEG'])
            

    percepcion_recuerdo_por_tema.append(percepcion_recuerdo)
    nro_palabras_unicas_por_tema.append(nro_palabras_unicas)
    intensidad_sentimiento_por_tema.append(intensidad_sentimiento)
    tipo_sentimiento_por_tema.append(tipo_sentimiento)
    intensidad_menos5_5_por_tema.append(intensidad_menos5_5) 

    intensidad_por_tema.append(py_int)
    
    pysent_pos_por_tema.append(py_pos)
    pysent_neg_por_tema.append(py_neg)
    pysent_neu_por_tema.append(py_neu)
    pysent_valencia_por_tema.append(df['valencia_pysent'])
    
    palabra_unicas.append(df['num_tot_palabras_unicas'])


#%% Gráfico palabras únicas vs palabras totales y su correlación

palabras = nro_palabras_norm

palabras_unicas = nro_palabras_unicas_norm

correlation_unicas_totales = []

markers = ['o', 's', 'x', 'v']#misma cantidad que el len(temas) '+'


plt.figure(1), plt.clf()
for i in range(len(nro_palabras_x_sujeto)):
    for j in range(len(nro_palabras_x_sujeto[0])):
        color = cm.tab20(i)
        plt.plot(palabras[i][j], palabras_unicas[i][j],  marker = markers[j], color =  color)
    
    #Calcular el coeficiente de correlación
    correlation_matrix = np.corrcoef(palabras[i], palabras_unicas[i])
    correlation_unicas_totales.append(correlation_matrix[0, 1])
    
# Crear líneas ficticias para la leyenda
lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]

# Agregar las líneas ficticias y las etiquetas a la leyenda
plt.legend(lines, temas)

#coeficiente de correlación

columnas = nro_sujeto
valores = correlation_unicas_totales

data = [valores]
df_sujetos = pd.DataFrame(data, columns=columnas)


todas_palabras = []

todas_palabras_unicas = []

for lista in palabras:
    todas_palabras.extend(lista)
    
for lista in palabras_unicas:
    todas_palabras_unicas.extend(lista)


#Calcular el coeficiente de correlación
correlation_matrix = np.corrcoef(todas_palabras, todas_palabras_unicas)

# Imprimir el resultado
print("Coeficiente de correlación de Pearson:", correlation_matrix[0, 1])

plt.text(0.85, 0.05, f'R = {correlation_matrix[0, 1]:,.2f}', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


plt.grid(True)
plt.xlabel('Palabras totales', fontsize = 15)
plt.ylabel('Palabras únicas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#plt.savefig(path_imagenes + '/palabras_unicas_vs_totales.png')

#%% Gráfico correlacion nro palabras unicas y percepcion recuerdo

correlation_unicas__recuerdo_por_tema = []
pvals_unicas_recuerdo_por_tema = []

plt.figure(3), plt.clf()
for i in range(len(temas)):
    for j in range(len(percepcion_recuerdo_por_tema[0])):
       if j in percepcion_recuerdo_por_tema[i].index:
            color = cm.tab20(j)
            plt.plot(percepcion_recuerdo_por_tema[i][j], nro_palabras_unicas_por_tema[i][j], marker = markers[i], color = color)
    
    # Calcular el coeficiente de correlación de Spearman
    correlation, p_value = spearmanr(percepcion_recuerdo_por_tema[i], nro_palabras_unicas_por_tema[i])
    correlation_unicas__recuerdo_por_tema.append(correlation)
    pvals_unicas_recuerdo_por_tema.append(p_value)
    
# Crear líneas ficticias para la leyenda
lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]

# Agregar las líneas ficticias y las etiquetas a la leyenda
plt.legend(lines, temas)



columnas = temas_tabla
valores = correlation_unicas__recuerdo_por_tema

data = [valores, pvals_unicas_recuerdo_por_tema]
df_sperman_tema = pd.DataFrame(data, columns=columnas)

indice = ['sper', 'pval']

df_sperman_tema = df_sperman_tema.set_index(pd.Index(indice))


# coeficiente se sperman total

percepcion = []

nro_palabras_unicas = []

for lista in percepcion_recuerdo_por_tema:
    percepcion.extend(lista)
    
for lista in nro_palabras_unicas_por_tema:
    nro_palabras_unicas.extend(lista)


# Calcular el coeficiente de correlación de Spearman
correlation, p_value = spearmanr(percepcion, nro_palabras_unicas)

# Imprimir el resultado
print("Coeficiente de correlación de Spearman:", correlation)
print("Valor p:", p_value)

plt.grid(True)
plt.xlabel('Percepción', fontsize = 15)
plt.ylabel('Palabras únicas', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.text(0.65, 1.05, fr'$\rho$ = {correlation:,.2f}, pval = {p_value:,.5f}', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

print(df_sperman_tema)

#path_imagen = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/Encuesta_postentrevista/Imagenes/Palabras_unicas_vs_percepcion_recuerdo.png'
plt.savefig(path_imagenes + '/Palabras_unicas_vs_percepcion_recuerdo.png')


#con la tabla en la figura


fig = plt.figure(figsize=(8, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1])  # Ajusta los valores de height_ratios según tus necesidades


imagen = plt.imread(path_imagenes + '/Palabras_unicas_vs_percepcion_recuerdo.png')  # Reemplaza 'ruta_de_la_imagen.jpg' con la ruta de tu imagen

ax_imagen = fig.add_subplot(gs[0])
ax_imagen.imshow(imagen)
ax_imagen.axis('off')  # Para ocultar los ejes de la imagen

df_sperman_tema = df_sperman_tema.round(3)


df_sperman_tema.insert(0, '.', df_sperman_tema.index)

ax_tabla = fig.add_subplot(gs[1])
tabla = ax_tabla.table(cellText=df_sperman_tema.values,
                       colLabels=df_sperman_tema.columns,
                       cellLoc='center',
                       loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.5)  # Escala el tamaño de la tabla
tabla.show_index = True  # Mostrar el índice en la tabla
ax_tabla.axis('off')  # Para ocultar los ejes de la tabla


fig.tight_layout()

plt.show()

#plt.savefig(path_imagenes + '/Palabras_unicas_vs_percepcion_recuerdo_tabla.png')

# sns.violinplot(data=data)
    


#%% Histograma de número de palabras por tema (de todos los sujetos)

markers = ['o', 's', 'P', 'v', 'H']

plt.figure(4), plt.clf()
for nro_tema in range(len(temas)):
    histograma = np.histogram(nro_palabras_unicas_por_tema[nro_tema].values) # alpha = 1 - nro_tema/8, edgecolor= color, color = color, linewidth = 2, label = temas[nro_tema]
    alturas = histograma[0]
    bordes = histograma[1]
    color = cm.tab20(2*nro_tema)
    plt.plot((bordes[1:]+bordes[:-1])/2, alturas, marker = markers[nro_tema], color = color, label = f'{temas[nro_tema]}, {np.median(nro_palabras_unicas_por_tema[nro_tema].values):,.2f}')
    
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.savefig(path_imagenes + '/Histogramas_temas_1.png')

plt.figure(5), plt.clf()
for nro_tema in range(len(temas)):
    plt.hist(nro_palabras_unicas_por_tema[nro_tema].values, edgecolor='None', alpha = 0.5 - nro_tema/16, color = color, linewidth = 2, label = f'{temas[nro_tema]}, {np.median(nro_palabras_unicas_por_tema[nro_tema].values):,.2f}')
    plt.hist(nro_palabras_unicas_por_tema[nro_tema].values, edgecolor= color, fill = False, linewidth = 2)
    color = cm.tab20(2*nro_tema)
    
plt.legend(fontsize = 13)
plt.grid(True)
plt.xlabel('Palabras únicas', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.savefig(path_imagenes + '/Histogramas_temas_2.png')

#%% hagamos un violin plot

#violin_plot(percepcion_recuerdo_por_tema, "Autopercepción de recuerdo por tema")


violin_plot(nro_palabras_unicas_por_tema, "Nro. palabras únicas por tema", nombre_guardado = "contenido_seminario")

#violin_plot(pysent_valencia_por_tema, "Valencia pysent por tema")

#%%

data1 = nro_palabras_unicas_por_tema

ylabel = "$Z_{score}$ nro. palabras únicas"

nombre_guardado = "contenido_seminario"

npl.rcParams['figure.figsize'] = [10, 15]
npl.rcParams["axes.labelsize"] = 16
npl.rcParams['xtick.labelsize'] = 16
npl.rcParams['ytick.labelsize'] = 16
data = {
    'Campeones': data1[0],
    'Presencial': data1[1],
    'CFK': data1[2],
    'Arabia': data1[3]
}   #Filler': data1[1]

df_ = pd.DataFrame(data)

colors = [color_celeste, color_celestito, color_palido, color_violeta]
#plt.figure(figsize=(15,10)), plt.clf()
# Crear el violin plot utilizando catplot
catplot = sns.catplot(data=df_, kind='violin', palette=colors, height=6, aspect=1.5)

catplot.set_ylabels(ylabel)

# Mostrar el gráfico
plt.show()

plt.gcf().subplots_adjust(bottom=0.1)
plt.subplots_adjust(left=0.1) 

plt.savefig(path_imagenes+'/' + nombre_guardado, transparent=True)




#%%
'''
Análisis de sentimiento
'''
#%% función para gráficos

def imagenes_comparacion_sent(analisis_sent_1, analisis_sent_2, string_analisis_sent_1, string_analisis_sent_2, escala = False):

    markers = ['o', 's', 'x', 'v', '+']#misma cantidad que len(temas)
    
    correlation_por_tema = []
    pvals_por_tema = []

    
    plt.figure(8), plt.clf()
    for i in range(len(temas)):
        for j in range(len(analisis_sent_1[0])):
            if j in analisis_sent_1[i].index:
                color = cm.tab20(j)
                plt.plot(analisis_sent_1[i][j], analisis_sent_2[i][j], marker = markers[i], markersize = 5, color = color)
        
        # Calcular el coeficiente de correlación de Spearman
        correlation, p_value = spearmanr(analisis_sent_1[i], analisis_sent_2[i])
        correlation_por_tema.append(correlation)
        pvals_por_tema.append(p_value)
        
    # Crear líneas ficticias para la leyenda
    lines = [plt.Line2D([], [], color='black', marker=m, linestyle='None') for m in markers]
    
    # Agregar las líneas ficticias y las etiquetas a la leyenda
    plt.legend(lines, temas)
    
    
    analisis_sent_1_ = []
    
    analisis_sent_2_ = []
    
    for lista in analisis_sent_1:
        analisis_sent_1_.extend(lista)
        
    for lista in analisis_sent_2:
        analisis_sent_2_.extend(lista)
    
    
    # Calcular el coeficiente de correlación de Spearman
    correlation, p_value = spearmanr(analisis_sent_1_, analisis_sent_2_)
    
    # Imprimir el resultado
    print("Coeficiente de correlación de Spearman:", correlation)
    print("Valor p:", p_value)
    
    plt.text(0.65, 1.05, fr'$\rho$ = {correlation:,.2f}, pval = {p_value:,.5f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), fontsize=18)
    
    plt.grid(True)
    plt.xlabel(string_analisis_sent_1, fontsize = 20)
    plt.ylabel(string_analisis_sent_2, fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    if escala == 'log':
        plt.xscale("log")
    plt.show()
    
    columnas = temas_tabla
    valores = correlation_por_tema
    
    data = [valores, pvals_por_tema]
    df_sperman_tema = pd.DataFrame(data, columns=columnas)
    
    indice = ['sper', 'pval']
    
    df_sperman_tema = df_sperman_tema.set_index(pd.Index(indice))
    
    
    print(df_sperman_tema)
    
    
    plt.savefig(path_imagenes + f'/{string_analisis_sent_1}_{string_analisis_sent_2}.png')
    
    #con la tabla en la figura


    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])  # Ajusta los valores de height_ratios según tus necesidades


    imagen = plt.imread(path_imagenes + f'/{string_analisis_sent_1}_{string_analisis_sent_2}.png')  # Reemplaza 'ruta_de_la_imagen.jpg' con la ruta de tu imagen
    ax_imagen = fig.add_subplot(gs[0])
    ax_imagen.imshow(imagen)
    ax_imagen.axis('off')  # Para ocultar los ejes de la imagen

    df_sperman_tema = df_sperman_tema.round(3)


    df_sperman_tema.insert(0, '.', df_sperman_tema.index)

    ax_tabla = fig.add_subplot(gs[1])
    tabla = ax_tabla.table(cellText=df_sperman_tema.values,
                           colLabels=df_sperman_tema.columns,
                           cellLoc='center',
                           loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)  # Escala el tamaño de la tabla
    tabla.show_index = True  # Mostrar el índice en la tabla
    ax_tabla.axis('off')  # Para ocultar los ejes de la tabla


    fig.tight_layout()

    plt.show()

    plt.savefig(path_imagenes + f'/{string_analisis_sent_1}_{string_analisis_sent_2}_tabla.png')
    
    #volvemos al tamaño del gráfico
    npl.rcParams['figure.figsize'] = [10, 10]
    #se ajusta el gráfico para que no haya espacios en blanco en la figura
    npl.rcParams["figure.autolayout"]= 'true'
    #tamaño de las lineas
    npl.rcParams["lines.linewidth"]=2
    #tamaño de letras de leyenda
    npl.rcParams["legend.fontsize"]=26
    #tamaño de letras en ejes
    npl.rcParams["axes.labelsize"] = 26
    #tamaño de los números de los ejes
    npl.rcParams['xtick.labelsize'] = 26
    npl.rcParams['ytick.labelsize'] = 26

    return 'ok'

def barra_comp_sent(analisis_sent_1, analisis_sent_2, string_analisis_sent_1, string_analisis_sent_2, scale = False):
    npl.rcParams.update(npl.rcParamsDefault)
    temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"] #"antesdevenir"
    for i, tem in enumerate(temas):
        corrimiento = np.ones(len(analisis_sent_1[i]))*0.4
        
        fig, ax1 = plt.subplots()
        
        # Configurar la primera serie (eje y izquierdo)
        color = 'tab:red'
        ax1.set_xlabel('Nro. sujetos')
        ax1.set_xticks(analisis_sent_1[i].index + 1)
        ax1.set_xticklabels(analisis_sent_1[i].index)
        ax1.set_ylabel(string_analisis_sent_1, color=color)
        ax1.bar(analisis_sent_1[i].index, analisis_sent_1[i], color=color, width=0.4)
        #ax1.bar(nro_tema, nro_palabras_unicas_sujeto, '.', color='lightcoral', label = 'unicas')
        ax1.legend()
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Configurar la segunda serie (eje y derecho)
        ax2 = ax1.twinx()  # Crea un segundo eje y
        color = 'tab:blue'
        ax2.set_ylabel(string_analisis_sent_2, color=color)
        ax2.bar(analisis_sent_1[i].index + corrimiento, analisis_sent_2[i], color=color, width=0.4)
        ax2.tick_params(axis='y', labelcolor=color)
        
        if scale == 'log':
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        
        #titulo
        plt.title(f"{temas[i]}")
        
        # Ajustar el tamaño del gráfico
        fig.tight_layout()
        
        
        # Mostrar el gráfico
        plt.show()
        
        plt.savefig(path_imagenes + f'/hist_{string_analisis_sent_1}_{string_analisis_sent_2}_{temas[i]}.png')

    return 'ok'


def barras_sent(analisis_sent_1, string_analisis_sent_1):
    temas = ["campeones_del_mundo", "presencial", "cfk", "arabia"] #"antesdevenir",
    
    for i, tem in enumerate(temas):
        color = 'tab:red'
        plt.figure(9), plt.clf()
        plt.xticks(analisis_sent_1[i].index , analisis_sent_1[i].index + 1)
        plt.bar(analisis_sent_1[i].index, analisis_sent_1[i], color=color, width=0.4)
        
        
        #plt.legend(fontsize = 13)
        plt.grid(True)
        plt.xlabel('Nro. sujetos', fontsize = 15)
        plt.ylabel(string_analisis_sent_1, fontsize = 15)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title(f"{temas[i]}")
        plt.tight_layout()
        
        plt.show()
        
        plt.savefig(path_imagenes + f'/barra_{string_analisis_sent_1}_{temas[i]}.png')

    return 'ok'

#%% violin plots

violin_plot(pysent_pos_por_tema, "Pysent positivo")

#%%
violin_plot(intensidad_menos5_5_por_tema, "Sentimiento autopercibido de -5 a 5")
violin_plot(tipo_sentimiento_por_tema, "Sentimiento autopercibida de -1 a 1")
violin_plot(intensidad_sentimiento_por_tema, "Intensidad autopercibido de 0 a 5")
violin_plot(intensidad_por_tema, "Intensidad con pysentimiento")

#%%
violin_plot(pysent_neg_por_tema, "Pysent negativo")
violin_plot(pysent_neu_por_tema, "Pysent neutro")

    
#%% Gráficos de psentcrudo vs lemm

imagenes_comparacion_sent(intensidad_por_tema, intensidad_sentimiento_por_tema, "Intensidad con pysentimiento", "Intensidad autopercibido de 0 a 5")

barra_comp_sent(intensidad_por_tema, intensidad_sentimiento_por_tema, "Intensidad con pysentimiento", "Intensidad autopercibido de 0 a 5")
#%%
imagenes_comparacion_sent(pysent_valencia_por_tema, tipo_sentimiento_por_tema, "Valencia con pysentimiento", "Valencia autopercibida")

imagenes_comparacion_sent(pysent_valencia_por_tema, palabra_unicas, "Valencia con pysentimiento", "Cantidad de palabras")


