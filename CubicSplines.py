# -*- coding: utf-8 -*-
"""
Ejercicio 2: Splines Cúbicos
"""
from scipy.interpolate import CubicSpline, sproot, splrep
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
ax.grid()
plt.show() 


####Obtengamos E_r y Gamma
i = 1 #Inicializamos un contador
E_r = 0. #Inicializamos la energía de resonancia

#Esta parte evalúa de 1 en 1 nuestro ajuste para determinar el máximo y así obtener E_r
while i != 200:
    Antes = cs.__call__(i-1)
    Desp = cs.__call__(i)
    if Desp > Antes:
        E_r = i
    i += 1    

#Maximo dividido por la mitad    
mitad_maximo = cs.__call__(E_r)/2.

#Creamos el spline recorrido como tipo splrep para poder sacar sus raices
cs_recorrido = splrep(x, y- mitad_maximo)
#sacamos las raices
r1, r2 = sproot(cs_recorrido)

#Valor de Gamma
gamma = r2 - r1

#Errores porcentuales
errorE_r = 100.*abs(78 - E_r)/78.
errorGamma = 100.*abs(55-gamma)/55.

print('-'*50)
print(f'Energía de resonancia: {E_r} MeV| Con un error de: {errorE_r} %')
print(f'Gamma: {gamma} MeV| Con un error de: {errorGamma} %')


