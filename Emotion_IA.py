#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from IPython.display import display
#from tensorflow.python.keras import * -> Para las versiones de TF
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
#from keras import optimizers -> para las nuevas versiones de TF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow


# In[2]:


# Cargar los puntos faciales
keyfacial_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Emotion+AI+Dataset/Emotion AI Dataset/data.csv')


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


keyfacial_df.head(10)


# In[5]:


# Obtener la informacion relevante del dataset

keyfacial_df.info()


# In[6]:


# Comprobamos si hay valores nulos en el dataset
keyfacial_df.isnull().sum()


# In[7]:


keyfacial_df['Image'].shape


# In[8]:


# Dado que los valores para la imagen se dan como cadenas separadas por espacios
# separamos los valores usando " " como separador
# luego convertimos esto a una matriz sanumerica usando np.fromstring y la
# convertimos en una matriz unidimencional 1D obtenida de una matriz 2D de forma
# (96 * 96)
keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x: np.fromstring (x,dtype = int, sep = ' ').reshape(96, 96))



# In[9]:


# Obtain the shape of the imege
keyfacial_df['Image'][0].shape


# MINI RETO #1
#   *   Obtenga los valores promedios minimo y maximo para 'righ_eye_center_x'

# In[10]:


keyfacial_df.describe()


# ## **Tarea 3: Visualizacion de imagenes**
# 

# In[11]:


#Representamos una imagen aleatoria del conjunto de datos junto con puntos clave faciales
#Los datos de la imagen se obtienen de df ['image'] y se representan usando plt.imshow
#15 coordenadas x e y para la imagen correspondiente
#Dado que las coordenadas x estan en columnas pares como 0,2,4, ... y las coordenadas y estan en columnas impares como 1,3,5, ...
#Accedemos a su valor usando el comando .loc, que obtiene los valores de las coordenadas de la imagen en la funcion de la columna a la que se refiere.

i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
for j in range(1, 31, 2):
  plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')


# In[12]:


# Veamos mas imagenes en formato matricial
fig = plt.figure(figsize=(20, 20))

for i in range(16):
  ax = fig.add_subplot(4, 4, i + 1)
  image = plt.imshow(keyfacial_df['Image'][i],cmap = 'gray')
  for j in range(1, 31, 2):
    plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')


# # **Mini Reto 2**

# * Realizar una verificacion adicional en los datos visualizando aleatoriamente 64 nuevas imagenes junto con sus puntos claves correspondientes

# In[13]:


import random
fig = plt.figure(figsize=(20, 20))
for i in range(64):
  k = random.randint(1, len(keyfacial_df))
  ax = fig.add_subplot(8, 8, i + 1)
  image = plt.imshow(keyfacial_df['Image'][k], cmap = 'gray')
  for j in range(1,31,2):
    plt.plot(keyfacial_df.loc[k][j-1], keyfacial_df.loc[k][j], 'rx')


# Tarea

# In[14]:


# Creamos una copia del dataframe
import copy
keyfacial_df_copy = copy.copy(keyfacial_df)


# In[15]:


# Obtenemos las columnas del dataframe

columns = keyfacial_df_copy.columns[:-1]
columns


# In[16]:


# Horizontal Flip - Damos la vuelta a las imagenes entorno al eje y
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 1))

# dado que estamos volteando horizontalmente, los valores de la coordenada y serian los mismo
# solo cambiaria los valores de la coordenada x, todo lo tenemos que hacer
# es restar nuestros valores iniciales a la coordenada x del ancho de la imagen (96)
for i in range(len(columns)):
  if i % 2 == 0:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))


# In[17]:


# Mostramos la imagen original
plt.imshow(keyfacial_df['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
  plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')


# In[18]:


# Mostramos la imagen girada horizontalmente
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
  plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# In[19]:


# conectamos el dataset original con le dataframe aumentado
augmented_df = np.concatenate([keyfacial_df, keyfacial_df_copy])


# In[20]:


augmented_df.shape


# In[21]:


#Aumentar aleatoriamente el brillo de las imagenes
#Multiplicamos los valores de los pixeles por los valores aleatorios entre
# 1,5 y 2 para aumentar el brillo de la iamgen
# Recortamos el valor entre 0 y 255

import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda  x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape


# In[22]:


# Mostramos la imagen con el brillo aumentado
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
  plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# MINI RETO #3
# 
# 
# 
# *   Aumenta las imagenes volteandolas verticalmente (Sugerencia: voltea a lo largo del eje x y ten en cuenta que si lo hacemos a lo largo del eje x, las coordenadas x no cambiaran)
# 
# 

# In[23]:


keyfacial_df_copy =  copy.copy(keyfacial_df)


# In[24]:


keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 0))

for i in range(len(columns)):
  if i % 2 == 1:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))


# MINI RETO #4
# 
# 
# *   haz una comprobacion adicional y visualiza las imagenes
# 
# 

# In[25]:


# Mostramos la imagen volteada
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
  plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# In[26]:


# Obtenemos el valor de las imagenes que estan presentes en la columna 31 (dado al indice comienza desde 0, nos referimos a la columna 31 por 30 en Python)
img = augmented_df[:, 30]
# Normalizamos las imagenes
img = img / 255.0

#Creamos un array vacio de tamaño (x, 96, 96, 1) para suministrar le modelo
X = np.empty((len(img), 96, 96, 1))

#Iteramos sobre la lista de imagenes y añadimos las mismas al array vacio tras expoandir su dimencion
for i in range(len(img)):
  X[i] = np.expand_dims(img[i], axis = 2)

#Convertimos el tipo de array a float32
X = np.asarray(X).astype (np.float32)
X.shape


# Tarea #5: Normalizaciòn de los datos y preparaciòn para el entrenamiento.
# 

# In[27]:


# Obtenemos el valor de las coordenadas x & y que se utilizaran como target
y = augmented_df[:, :30]
y = np.asarray(y).astype(np.float32)
y.shape


# In[28]:


# Dividimos los datos en entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Mini Reto#5
# * Intenta usar un valor diferente para test_split y verificar que la division es correcta

# In[29]:


X_train.shape


# In[30]:


X_test.shape


# Tarea #6: Entender la teoria e intuicion detras de las redes neuronales

# Las funciones de activación son esenciales en redes neuronales porque introducen no linealidad, permitiendo que el modelo aprenda relaciones complejas. Aquí hay tres funciones de activación comunes:
# 
# 1. ReLU (Rectified Linear Unit)
# Fórmula:
# 𝑓
# (
# 𝑥
# )
# =
# max
# ⁡
# (
# 0
# ,
# 𝑥
# )
# f(x)=max(0,x)
# Ventajas:
# Simple de calcular.
# Ayuda a resolver el problema de desvanecimiento del gradiente.
# Convergencia más rápida en redes profundas.
# Desventaja:
# Puede causar "neuronas muertas" (valores que siempre son cero si los pesos iniciales o actualizados hacen que el valor de entrada sea negativo).
# 

# 2. Sigmoide
# Fórmula:
# 𝑓
# (
# 𝑥
# )
# =
# 1
# /1
# +
# 𝑒
# −
# 𝑥
# f(x)=
# 1+e
# −x
#  /1
# ​
# 
# Ventajas:
# Salidas en el rango
# [
# 0
# ,
# 1
# ]
# [0,1], útil para probabilidades.
# Desventajas:
# Gradiente se desvanece en entradas extremas.
# No centrada en cero, lo que puede ralentizar el entrenamiento.

# 3. Tanh (Tangente hiperbólica)
# Fórmula:
# 𝑓
# (
# 𝑥
# )
# =
# 𝑒
# 𝑥
# −
# 𝑒
# −
# 𝑥
# 𝑒
# 𝑥
# +
# 𝑒
# −
# 𝑥
# f(x)=
# e
# x
#  +e
# −x
# 
# e
# x
#  −e
# −x
# 
# ​
# 
# Ventajas:
# Salidas en el rango
# [
# −
# 1
# ,
# 1
# ]
# [−1,1], mejor centrada en cero que sigmoide.
# Desventajas:
# También sufre el problema de desvanecimiento del gradiente.

# Tipo preferido para capas ocultas:
# La ReLU es generalmente la opción preferida para las capas ocultas debido a su simplicidad y eficiencia en redes profundas. En ciertos casos, variantes como Leaky ReLU o ELU se utilizan para abordar el problema de las neuronas muertas. Sin embargo, la elección también puede depender del tipo de datos y la arquitectura del modelo.

# 

# In[34]:


# ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tangente hiperbólica (tanh)
def tanh(x):
    return np.tanh(x)


# In[35]:


# Simula un dataset
import pandas as pd

# Crear un DataFrame
data = {'valores': np.linspace(-10, 10, 100)}
df = pd.DataFrame(data)


# In[36]:


# Aplicar ReLU
df['relu'] = relu(df['valores'])

# Aplicar sigmoide
df['sigmoid'] = sigmoid(df['valores'])

# Aplicar tangente hiperbólica
df['tanh'] = tanh(df['valores'])


# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir las funciones de activación
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Crear el DataFrame de ejemplo
data = {'valores': np.linspace(-10, 10, 200)}
df = pd.DataFrame(data)
df['relu'] = relu(df['valores'])
df['sigmoid'] = sigmoid(df['valores'])
df['tanh'] = tanh(df['valores'])

# Crear la figura con tamaño mejorado
plt.figure(figsize=(14, 8))

# Graficar las funciones de activación con colores más atractivos y mejor contraste
plt.plot(df['valores'], df['valores'], label='Identidad (y=x)', linestyle='--', color='gray', linewidth=2)  # Línea de referencia
plt.plot(df['valores'], df['relu'], label='ReLU', color='#1f77b4', linewidth=2)  # Azul
plt.plot(df['valores'], df['sigmoid'], label='Sigmoide', color='#ff7f0e', linewidth=2)  # Naranja
plt.plot(df['valores'], df['tanh'], label='Tanh', color='#2ca02c', linewidth=2)  # Verde

# Mejorar la visibilidad de los ejes
plt.axhline(0, color='black', linewidth=1, alpha=0.7, linestyle='--')  # Línea horizontal en y=0
plt.axvline(0, color='black', linewidth=1, alpha=0.7, linestyle='--')  # Línea vertical en x=0

# Títulos y etiquetas
plt.title('Comparación de Funciones de Activación', fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Entrada', fontsize=14)
plt.ylabel('Salida', fontsize=14)

# Leyenda con estilo
plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True, framealpha=0.7)

# Mejorar el grid
plt.grid(visible=True, alpha=0.4, linestyle='--')

# Ajustar los márgenes para mejor visualización
plt.tight_layout()

# Mostrar la gráfica
plt.show()
