# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:50:07 2021

@author: Julio C. Torreblanca
"""
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importamos los datos de un archivo txt
data = pd.read_csv(r'C:\Users\cesar.avila\Documents\IPN\Semestre 7\Física Numérica\Tareas\Tarea6\DatosLagrange.txt',header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
E = data.iloc[:,0]
f = data.iloc[:,1]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
x = np.float64(E)
y = np.float64(f)

#Creamos el CubicSpline
cs = CubicSpline(x, y)

#Creamos un arrgelo que contendrá los puntos a ser evaluados  en el eje x
eje_x = np.arange(-2.0, 205, 5)
eje_y = cs(eje_x)

fig, ax = plt.subplots()
ax.plot(x,y, 'og', label = 'Datos experimentales')
ax.plot(eje_x, eje_y,  label = 'Spline Cúbico')
ax.legend(loc='upper right')
plt.show() 