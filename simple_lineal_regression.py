# regresion lineal simple

# importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importamos el train.models_selection, que es la libreria que nos a dividir el dataset
from sklearn.model_selection import train_test_split

# iportamos la libreria encargada de realizar la regresion
from sklearn.linear_model import LinearRegression
# importamos el dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# dividimos el dataset en conjunto de entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0)

# creamos el modelo de regresion simple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(x_train, y_train)

# predecir el conjunto de test
y_prediction = regression.predict(x_test)
print(y_test, y_prediction)


'''# visualizar los resultados de entrenamineto

# funcion para pintar una nube de puntos
plt.scatter(x_train, y_train, color='red')

# para representar una recta de regresion
plt.plot(x_train, regression.predict(x_train), color='blue')

# podemos agregarle titulo a nuestra grafica
plt.title('Sueldo vs Anios Experiencia (Conjunto de entrenamiento)')

# agregar etiquetas para los ejes
plt.xlabel('Anios de experiencia')
plt.ylabel('Sueldo en($)')
plt.show()
'''
# ----------------------------------------------------------------------
# visualizar los resultados de entrenamineto

# funcion para pintar una nube de puntos
plt.scatter(x_test, y_test, color='red')

# para representar una recta de regresion
plt.plot(x_train, regression.predict(x_train), color='blue')

# podemos agregarle titulo a nuestra grafica
plt.title('Sueldo vs Anios Experiencia (Conjunto de testing)')

# agregar etiquetas para los ejes
plt.xlabel('Anios de experiencia')
plt.ylabel('Sueldo en($)')
plt.show()
