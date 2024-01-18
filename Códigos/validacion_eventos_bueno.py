# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:06:01 2024

@author: corir
"""

#%%librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats

#%% funciones


def rgb_to_hex(rgb):
    # Asegurarse de que los valores estén en el rango [0, 1]
    rgb = tuple(max(0, min(1, x)) for x in rgb)

    # Convertir los valores RGB a enteros en el rango [0, 255]
    rgb_int = tuple(int(x * 255) for x in rgb)

    # Formatear el color en formato hexadecimal
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_int)

    return hex_color

def darken_color(color, factor=0.6):
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

#%%paleta de colores y path imagenes
color_hex = "#79b4b7ff"
color_celeste = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5, 7)]

color_celestito = "#afd1d3ff"

color_palido = "#f8f0dfff"

color_gris = "#9fa0a4ff"

palette = sns.color_palette("PuBu", n_colors=5) #

#palette = sns.color_palette("autumn_r", n_colors=2)

# Asignar colores a las variables
color_1 = rgb_to_hex(palette[2])
color_2 = rgb_to_hex(palette[3])
color_3 = rgb_to_hex(palette[4])

color= [color_1, color_2, color_3]

rainbow_palette = sns.color_palette("rainbow", n_colors=7)

#rainbow_palette = sns.color_palette("autumn_r", n_colors=2)



# Asignar colores a las variables
color_1 = rgb_to_hex(rainbow_palette[0])
color_2 = rgb_to_hex(rainbow_palette[2])
color_3 = rgb_to_hex(rainbow_palette[4])
color_4 = rgb_to_hex(rainbow_palette[5])
color_5 = rgb_to_hex(rainbow_palette[6])

color_iv = [color_1, color_2, color_3, color_4, color_5]

color_iv_dark = [darken_color(color_1), darken_color(color_2), darken_color(color_3), darken_color(color_4)]


color_femenino = "#688db9"
color_masculino1 = "#ffa424"
color_masculino2 = "#ff9800" 
path_imagenes = "C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Figuras_finales/Encuestas/Validacion"

#%%
path = "C:/Users/Usuario/Desktop/Cori/git tesis/tesis_cori/Encuestas/Validacion/Encuesta sobre eventos del 2022 (Respuestas) - Respuestas de formulario 1.csv"

df_validacion = pd.read_csv(path)

df_validacion["Género"][69] = "Masculino"
df_validacion["Género"][52] = "Masculino"

#%%

contador_femenino_edad_rango = []
contador_masculino_edad_rango = []

edades_rango = [(18,22), (23,25), (26, 32), (33, 45), (46,60), (61,90)]
for rango in edades_rango:
    edad_min, edad_max = rango
    
    # Filtrar el DataFrame y contar elementos
    contador_femenino_edad_rango.append(len(df_validacion[(df_validacion['Género'] == 'Femenino') & (df_validacion['Edad'].between(edad_min, edad_max))]))
    contador_masculino_edad_rango.append(len(df_validacion[(df_validacion['Género'] == 'Masculino') & (df_validacion['Edad'].between(edad_min, edad_max))]))
    
edades = ['18-22', '23-25', '26-32', '33-45', '46-60', '61+']
poblacion_hombres = contador_masculino_edad_rango
poblacion_mujeres = contador_femenino_edad_rango

fig, ax = plt.subplots(1,1, figsize = (11,5))

# Barras de hombres
ax.barh(edades, poblacion_hombres, color=color_masculino1,label='Hombres', align='center', alpha = 0.4)

# Barras de mujeres a la izquierda del eje y
ax.barh(edades, [-x for x in poblacion_mujeres], left=0, color=color_femenino,label='Mujeres',  align='center', alpha = 0.4)

# Barras de hombres con relleno transparente y bordes con patrón
ax.barh(edades, poblacion_hombres, color='none', edgecolor=color_masculino1, linewidth=1.5, align='center')

# Barras de mujeres a la izquierda del eje y con relleno transparente y bordes con patrón
ax.barh(edades, [-x for x in poblacion_mujeres], left=0, color='none', edgecolor=color_femenino, linewidth=1.5, align='center')


# Añadir etiquetas y título
ax.set_xlabel('Cantidad', fontsize = 20)
ax.set_ylabel('Edades', fontsize = 20)
ax.tick_params(axis='both', which='both', labelsize=18)

# Configurar el eje y para que las edades estén en el centro
ax.set_yticks(np.arange(len(edades)))
ax.set_yticklabels(edades)

ticks_x = [-20, -15, -10, -5, 0, 5, 10, 15]
ax.set_xticks(ticks_x)
ax.set_xticklabels([abs(tick) for tick in ticks_x])  # Hacer los ticks positivos

# Añadir leyenda
ax.legend(fontsize = 18)
plt.tight_layout()
plt.show()

plt.savefig(path_imagenes + '/edad_genero_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/edad_genero.png') 
plt.savefig(path_imagenes + '/edad_genero.pdf') 

#%%
df_validacion_filtrado = df_validacion[df_validacion['Edad'] < 32]

recordas_arabia = df_validacion['¿Te acordas cuando sucedió o te enteraste del evento?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_francia = df_validacion['¿Te acordas cuando sucedió o te enteraste del evento?.1'].value_counts().get('Si', 0)/len(df_validacion)
recordas_CFK = df_validacion['¿Te acordas cuando sucedió o te enteraste del evento?.2'].value_counts().get('Si', 0)/len(df_validacion)
recordas_GH = df_validacion['¿Viste el comienzo?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_Arg1985 = df_validacion['¿Viste la pelicula?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_guerra = df_validacion['¿Te acordas cuando sucedió o te enteraste este evento?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_censo = df_validacion['¿Recordas el día del censo y/o el momento en el que lo completaste digitalmente?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_reina = df_validacion['¿Recordas cuando sucedió o te enteraste el evento?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_coldplay = df_validacion['¿Recordas cuando sucedió o te enteraste de este evento?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_guzman = df_validacion['¿Recordas cuando sucedió o te enteraste de este evento?.1'].value_counts().get('Si', 0)/len(df_validacion)
recordas_presencial = df_validacion_filtrado['¿Tuviste la vuelta a las clases presenciales en el 2022?'].value_counts().get('Si', 0)/len(df_validacion_filtrado)
recordas_dojacat = df_validacion['¿Recordas cuando sucedió o te enteraste de este evento?.2'].value_counts().get('Si', 0)/len(df_validacion)
recordas_biza = df_validacion['¿Recordas cuando escuchaste la canción?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_drstrange = df_validacion['¿Viste la pelicula?.1'].value_counts().get('Si', 0)/len(df_validacion)
recordas_thor = df_validacion['¿Viste la pelicula?.2'].value_counts().get('Si', 0)/len(df_validacion)
recordas_blackpanter = df_validacion['¿Viste la pelicula?.3'].value_counts().get('Si', 0)/len(df_validacion)
recordas_pele = df_validacion['¿Recuerdas cuando sucedió o te enteraste del evento?'].value_counts().get('Si', 0)/len(df_validacion)
recordas_lula = df_validacion['¿Recuerdas cuando sucedió o te enteraste del evento?.1'].value_counts().get('Si', 0)/len(df_validacion)
recordas_condenaCFK = df_validacion['¿Recuerdas cuando sucedió o te enteraste del evento?.2'].value_counts().get('Si', 0)/len(df_validacion)
recordas_elonmusk = df_validacion['¿Recuerdas cuando sucedió o te enteraste del evento?.3'].value_counts().get('Si', 0)/len(df_validacion)

recordas = [recordas_arabia, recordas_francia, recordas_CFK, recordas_GH, recordas_Arg1985, recordas_guerra, recordas_censo, recordas_reina, recordas_coldplay, recordas_guzman, recordas_presencial, recordas_dojacat, recordas_biza, recordas_drstrange, recordas_thor, recordas_blackpanter, recordas_pele, recordas_lula, recordas_condenaCFK, recordas_elonmusk]


temas = ["Arabia", "Campeones", "CFK", "GH", "1985", "Guerra", "Censo", "Reina", "Coldplay", "Guzman", "Presencial", "Doja", "Biza", "Peli1", "Peli2", "Peli3", "Pele", "Lula", "CFK2", "Tw"]

recordas, temas = zip(*sorted(zip(recordas, temas), reverse=True))

plt.figure(figsize = (12,8))
# Crear el gráfico de barras
plt.bar(temas[:8], recordas[:8], color = color_femenino, alpha = 0.85)

# Añadir etiquetas y título
#plt.xlabel('Temas')
plt.ylabel('Cantidad que recuerda', fontsize = 20)
plt.tick_params(axis='both', which='both', labelsize=18)

# Rotar etiquetas del eje X para mejorar la legibilidad
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
# Mostrar el gráfico
plt.show()
#%%
cuanto_recordas_arabia = df_validacion['¿Cuánto recordas lo que viviste cuando sucedió/te enteraste de este evento? Donde 0 es nada y 5 es mucho.']
cuanto_recordas_francia = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho.']
cuanto_recordas_CFK = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..1']
cuanto_recordas_guerra = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..3']
cuanto_recordas_censo = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..4']
cuanto_recordas_reina = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..5']
cuanto_recordas_coldplay = df_validacion['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..6']
cuanto_recordas_presencial = df_validacion_filtrado['¿Cuánto recordas lo que viviste vos cuando te enteraste de este evento? Donde 0 es nada y 5 es mucho..8']

cuanto_recordas = [cuanto_recordas_francia, cuanto_recordas_CFK, cuanto_recordas_arabia, cuanto_recordas_coldplay, cuanto_recordas_guerra, cuanto_recordas_reina, cuanto_recordas_censo, cuanto_recordas_presencial]
temas = ["Campeones", "CFK", "Arabia", "Coldplay", "Reina", "Guerra", "Censo", "Presencial"]

means = []
std = []
for cuanto in cuanto_recordas:
    means.append(np.nanmean(cuanto))
    std.append(np.nanstd(cuanto)/np.sqrt(len(cuanto)))
    
plt.figure(figsize = (12,8))
# Crear el gráfico de barras
plt.bar(temas, means, yerr = std, color = color_femenino, alpha = 0.85)

# Añadir etiquetas y título
#plt.xlabel('Temas')
plt.ylabel('Cuanto recuerdan promedio', fontsize = 20)
plt.tick_params(axis='both', which='both', labelsize=18)

# Rotar etiquetas del eje X para mejorar la legibilidad
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
# Mostrar el gráfico
plt.show()

#%% figura de ambas juntas

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 7))

ax1.bar(temas[:8], np.asarray(recordas[:8])*100, color=color_femenino, alpha=0.85)
# ax1.set_xlabel('Temas')  # Puedes descomentar esta línea si es necesario
ax1.set_ylabel('Porcentaje que recuerda (%)', fontsize=20)
ax1.tick_params(axis='both', which='both', labelsize=18)
# Rotar etiquetas del eje X para mejorar la legibilidad
ax1.set_xticklabels(temas[:8], rotation=45, ha="right")


ax2.bar(temas, means, yerr=std, color=color_femenino, alpha=0.85)
# ax2.set_xlabel('Temas')  # Puedes descomentar esta línea si es necesario
ax2.set_ylabel('Cuánto recuerdan promedio', fontsize=20)
ax2.tick_params(axis='both', which='both', labelsize=18)
# Rotar etiquetas del eje X para mejorar la legibilidad
ax2.set_xticklabels(temas, rotation=45, ha="right")

ax1.text(0.97, 0.97, '(a)', transform=ax1.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

ax2.text(0.97, 0.97, '(b)', transform=ax2.transAxes,
        fontsize=28, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + '/validacion_recuerdo_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/validacion_recuerdo.png') 
plt.savefig(path_imagenes + '/validacion_recuerdo.pdf') 

#%%
intensidad_arabia = df_validacion['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho.']
intensidad_francia = df_validacion['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..1']
intesidad_CFK = df_validacion['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..2']
intensidad_presencial = df_validacion_filtrado['¿Con que intensidad sentis esta emoción? Donde 1 es nada, 5 es mucho..10']

mapeo_emociones = {'Negativas': -1, 'Neutras': 0, 'Positivas': 1}
valencia_arabia = df_validacion['¿Qué tipo de emociones te genera el recuerdo?'].map(mapeo_emociones)
valencia_francia = df_validacion['¿Qué tipo de emociones te genera el recuerdo?.1'].map(mapeo_emociones)
valencia_CFK = df_validacion['¿Qué tipo de emociones te genera el recuerdo?.2'].map(mapeo_emociones)
valencia_presencial = df_validacion_filtrado['¿Qué tipo de emociones te genera el recuerdo?.10'].map(mapeo_emociones)

cond = ["CFK", "Arabia", "Campeones", "Presencial"]


valencia1 = [valencia_CFK.dropna(), valencia_arabia.dropna(), valencia_francia.dropna(), valencia_presencial]
intensidad1 = [intesidad_CFK.dropna(), intensidad_arabia.dropna(), intensidad_francia.dropna(), intensidad_presencial.dropna()]


    
fig, ax2 = plt.subplots(1, 1, figsize = (14, 5))


for i in range(len(valencia1)):
    mean_intensidad1 = np.nanmean(intensidad1[i])
    mean_valencia1 = np.nanmean(valencia1[i])
    
    std_intensidad1 = np.nanstd(intensidad1[i]/np.sqrt(len(intensidad1[i])))
    std_valencia1 = np.nanstd(valencia1[i]/np.sqrt(len(valencia1[i])))

    ax2.scatter(mean_intensidad1, mean_valencia1, marker="o", s=100, c=color_iv[i], label=cond[i], alpha = 0.7)
    ax2.errorbar(mean_intensidad1, mean_valencia1, xerr=std_intensidad1, yerr=std_valencia1, fmt='none', ecolor=color_iv[i], elinewidth=2, capsize=5, alpha = 0.7)


ax2.set_xlabel('Media intensidad', fontsize = 20)
ax2.set_ylabel('Media valencia', fontsize = 20)
ax2.legend(fontsize = 18, loc = "lower right")
ax2.tick_params(axis='x', labelsize=18)  
ax2.tick_params(axis='y', labelsize=18)


plt.tight_layout()
# Mostrar el gráfico
plt.show()

plt.savefig(path_imagenes + '/validacion_mediaIntvsVal_transparente.png', transparent = True) 
plt.savefig(path_imagenes + '/validacion_mediaIntvsVal.png') 
plt.savefig(path_imagenes + '/validacion_mediaIntvsVal.pdf') 
