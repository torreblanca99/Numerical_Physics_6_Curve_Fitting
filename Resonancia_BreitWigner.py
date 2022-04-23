# -*- coding: utf-8 -*-
"""
Resonancia de Breit-Wigner
"""

import pandas as pd
import sympy as sp
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

def gx(i:int , values: list): 
    """
    Esta función calcula el valor de g(x_i) usando el cálculo simbólico
    de sumpy
    
    Parámetros:
        i = número de índice de x_i, i =0,1,2,...n-1 con n = no. de datos
        x = lista que contiene los datos experimentales para obtener g(x_i)
    Salida:
        Regresa la ecuación g(x_i)
    """
    
    g, a1, a2, a3 = sp.symbols('g, a1, a2, a3') #Creamos las variables para el cálculo simbólico
    
    #Inicializamos las variables
    g = 0

    #Esta parte calculo g(x_i) usando la fórmula
    
    g = a1/((values[i] - a2)**2 + a3)
    g = sp.expand(g)
    return g

def f_vector(f1,f2,f3, x:float, y:float, z:float):
    """Esta función calcula el f vector en la solución de Newton-Rapghson y
    evalúa a las funciones f1, f2, f3 en los puntos x,y,z. 
    Regresa una matriz."""
    
    a1, a2, a3 = sp.symbols('a1, a2, a3')

    valor1 = f1.subs([(a1,x), (a2,y), (a3,z)])
    valor2 = f2.subs([(a1,x), (a2,y), (a3,z)])
    valor3 = f3.subs([(a1,x), (a2,y), (a3,z)])
    
    return np.array([[valor1], [valor2], [valor3]])
    
def jacobiano(f1,f2,f3, x:float, y:float, z:float):
    """Esta función calcula la matriz Jacobiana en la solución de Newton-Rapghson 
    y evalúa a las derivadas parciales de f1, f2, f3 en los puntos x,y,z. 
    Regresa una matriz."""
    
    df, a1, a2, a3 = sp.symbols('df, a1, a2, a3')
    
    variables = [a1, a2, a3]
    funciones = [f1, f2, f3]
    
    #Saca la derivada y evalua los valores de a1,a2,a3=x,y,z
    J =np.array([[np.float64(sp.diff(funciones[i], variables[j]).subs([(a1,x), (a2,y), (a3,z)])) 
                 for j in range(len(variables))] for i in range(len(variables))])
    
    return J
    

# Importamos los datos de un archivo txt
data = pd.read_csv(r'C:\Users\cesar.avila\Documents\IPN\Semestre 7\Física Numérica\Tareas\Tarea6\DatosLagrange.txt'
                   ,header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
E = data.iloc[:,0]
f = data.iloc[:,1]
sigma = data.iloc[:,2]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
data_x = np.float64(E)
data_y = np.float64(f)
sigma_y = np.float64(sigma)

#Guardamos el número de datos 
n = len(data_x)

# Definimos variables para la obtendión g(x) y los inicializamos
g, a1, a2, a3 = sp.symbols('g, a1, a2, a3')
g = 0.

# Definimos variables para la obtendión f_1, f2, f3 y los inicializamos
f1, f2, f3, aux = sp.symbols('f1, f2, f3, aux')
f1 = 0.
f2 = 0.
f3 = 0.



for i in range(n):
    #Cálculo de g(x_i)
    g = gx(i,data_x)
    #Cálculo de  f1
    aux = (data_y[i] - g) / (((data_x[i] - a2)**2 + a3) * sigma_y[i]**2)
    aux = sp.expand(aux)
    f1 += aux
    
    #Cálculo de  f2
    aux = (data_y[i] - g) * (data_x[i] - a2) / (((data_x[i] - a2)**2 + a3)**2 * sigma_y[i]**2)
    aux = sp.expand(aux)
    f2 += aux
    
    #Cálculo de  f3
    aux = (data_y[i] - g) / (((data_x[i] - a2)**2 + a3)**2 * sigma_y[i]**2)
    aux = sp.expand(aux)
    f3 += aux

#Semillas de a1,a2,a3    
x = 1.
y = 85.
z = 756.25

# Longitud a1+Delta_a1,a2+Delta_a2, a3+Delta_a3

a = [0., 0., 0.]
a_new = [x, y, z]

# Definimos el error para detener la iteración de Newton-Raphson
error = 10**(-5)
i=1

condicion = abs(a_new[0] - a[0])

#Iteraciones Newton-Raphson:
while  condicion > error:
    print(f'No. iteración: {i}')
    print(f'Condición: {condicion}')
    A = f_vector(f1, f2, f3, a_new[0], a_new[1], a_new[2]) 
    B = jacobiano(f1, f2, f3, a_new[0], a_new[1], a_new[2])
    inv_B = np.linalg.inv(B)
      
    Delta_a = -inv_B.dot(A)
    
    #Guardamos los viejos valores
    a[0] = a_new[0]
    a[1] = a_new[1]
    a[2] = a_new[2]
    
    #Asignamos los nuevos valores
    a_new[0] += Delta_a[0][0] 
    a_new[1] += Delta_a[1][0]
    a_new[2] += Delta_a[2][0]
    condicion = abs(a_new[0] - a[0])
    i+=1

print('-'*50)    
print(f'Valores:\na_1 = {a_new[0]}\na_2 = {a_new[1]} \na_3 = {a_new[2]}')