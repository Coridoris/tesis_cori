# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:53:30 2023

@author: corir
"""
#%% librerias
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
#%% data

entrevista = "Primera"

temas = ["campeones_del_mundo", "presencial", "cfk", "arabia", "antesdevenir"]

condicion = temas[0]

path_sinautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_todos_temas.csv'

path_conautopercepcion_todas = 'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_todos_temas.csv'

path_autopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_conautopercepcion_{condicion}.csv'

path_sinautopercepcion_tema = f'C:/Users/Usuario/Desktop/Cori/Tesis/Primera_entrevista/ELcsv/ELcsv_sinautopercepcion_{condicion}.csv'


df = pd.read_csv(path_sinautopercepcion_todas)

mapping = {
    'antesdevenir': 5,
    'arabia': 4,
    'campeones_del_mundo': 1,
    'presencial': 2,
    'cfk': 3,
}

# Aplicar el mapeo a la columna 'Texto' y crear una nueva columna 'Numerico'
df['Condición'] = df['Condición'].map(mapping)

df = df.dropna()

y = np.asarray(df['Condición'])


df = df.drop(['Sujetos', 'Condición'], axis=1)

X = df.to_numpy()

#%%


# Aplicación de t-SNE para reducción de dimensionalidad a 2 dimensiones
n_components=3
perplexity=20.0 #the number of nearest neighbors that is used in other manifold learning algorithms. Different values can result in significantly different results
early_exaggeration=20.0
learning_rate=40
n_iter=10000
tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = None)
#si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
X_embedded = tsne.fit_transform(X)

#%%
# Graficar los datos reducidos a 2 dimensiones
plt.figure(2), plt.clf()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", len(set(y))))
plt.title('t-SNE Embedding')

cbar = plt.colorbar(scatter, ticks = np.linspace(1, len(temas), len(temas)), label='Classes')
temas_dict = {i: tema for i, tema in enumerate(temas)}
# Asignar nombres a los ticks del colorbar
cbar.set_ticklabels([temas_dict[i] for i in range(len(temas))])
# Agregar cuadro de texto con las variables
textstr = '\n'.join((
    f'n_components={n_components}',
    f'perplexity={perplexity}',
    f'early_exaggeration={early_exaggeration}',
    f'learning_rate={learning_rate}',
    f'n_iter={n_iter}'
))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
plt.show()
#%%
n_components=3
perplexity = 8
early_exaggeration=10.0 #Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical
#learning_rate=500 #The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
n_iter=10000
for learning_rate in tqdm(np.linspace(50, 715, 8)):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = None)
    #si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(learning_rate), plt.clf()
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", len(set(y))))
    plt.title('t-SNE Embedding')

    cbar = plt.colorbar(scatter, ticks = np.linspace(1, len(temas), len(temas)), label='Classes')
    temas_dict = {i: tema for i, tema in enumerate(temas)}
    # Asignar nombres a los ticks del colorbar
    cbar.set_ticklabels([temas_dict[i] for i in range(len(temas))])
    # Agregar cuadro de texto con las variables
    textstr = '\n'.join((
        f'n_components={n_components}',
        f'perplexity={perplexity}',
        f'early_exaggeration={early_exaggeration}',
        f'learning_rate={learning_rate}',
        f'n_iter={n_iter}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.show()
    
    path_imagenes = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/TSNE'
    
    plt.savefig(path_imagenes + f'/perplexity8_early10_learningrate{learning_rate}.png', transparent = False)
    
#%%
n_components=3
perplexity=8.0 #the number of nearest neighbors that is used in other manifold learning algorithms. Different values can result in significantly different results
early_exaggeration=10.0
learning_rate=477
n_iter=10000


tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = 44)
#si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
X_embedded = tsne.fit_transform(X)

plt.figure(learning_rate), plt.clf()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", len(set(y))))
plt.title('t-SNE Embedding')

cbar = plt.colorbar(scatter, ticks = np.linspace(1, len(temas), len(temas)), label='Classes')
temas_dict = {i: tema for i, tema in enumerate(temas)}
# Asignar nombres a los ticks del colorbar
cbar.set_ticklabels([temas_dict[i] for i in range(len(temas))])
# Agregar cuadro de texto con las variables
textstr = '\n'.join((
    f'n_components={n_components}',
    f'perplexity={perplexity}',
    f'early_exaggeration={early_exaggeration}',
    f'learning_rate={learning_rate}',
    f'n_iter={n_iter}'
))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
plt.show()

path_imagenes = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/TSNE'

plt.savefig(path_imagenes + f'/perplexity8_early10_learningrate{learning_rate}.png', transparent = False)
#%%
n_components=3
perplexity=8.0 #the number of nearest neighbors that is used in other manifold learning algorithms. Different values can result in significantly different results
early_exaggeration=10.0
learning_rate=240
n_iter=10000


tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=5000, random_state = 44)
#si quiero cosas reproducibles tengo q poner un núm en random_state: Pass an int for reproducible results
X_embedded = tsne.fit_transform(X)

plt.figure(learning_rate), plt.clf()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("viridis", len(set(y))))
plt.title('t-SNE Embedding')

cbar = plt.colorbar(scatter, ticks = np.linspace(1, len(temas), len(temas)), label='Classes')
temas_dict = {i: tema for i, tema in enumerate(temas)}
# Asignar nombres a los ticks del colorbar
cbar.set_ticklabels([temas_dict[i] for i in range(len(temas))])
# Agregar cuadro de texto con las variables
textstr = '\n'.join((
    f'n_components={n_components}',
    f'perplexity={perplexity}',
    f'early_exaggeration={early_exaggeration}',
    f'learning_rate={learning_rate}',
    f'n_iter={n_iter}'
))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
plt.show()

path_imagenes = f'C:/Users/Usuario/Desktop/Cori/Tesis/{entrevista}_entrevista/TSNE'

plt.savefig(path_imagenes + f'/perplexity8_early10_learningrate{learning_rate}.png', transparent = False)
#%%
# Graficar los datos reducidos a 2 dimensiones
plt.figure(12), plt.clf()
plt.scatter(X_embedded[0:30, 0], X_embedded[0:30, 1], c='green', label = "camp")
plt.scatter(X_embedded[120:, 0], X_embedded[120:, 1], c='blue', label = "filler")
plt.colorbar(label='Classes')
plt.title('t-SNE Embedding')
plt.legend()
plt.show()
'''
de la figura 4 a la 10 está hecho con n = 2

de 4 a 13 usé tsne perplexity=30.0, early_exaggeration=12.0, learning_rate=500, n_iter=10000,
'''