# -*- coding: utf-8 -*-
"""
Ejercicio 1: Interpolación de Lagrange

"""
import pandas as pd
import sympy as sp
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


def Lx(i:int , values: list): 
    """
    Esta función calcula el coeficiente de gragange L_i(x) de forma simbólica 
    para la imterpolación de Lagrange.
    
    Parámetros:
        i = número de índice de L_i(x) po obtener
        x = lista que contiene los datos experimentales para obtener L_i(x)
    Salida:
        Regresa un polinomio de grado n-1, con n el número de datos. Este 
        polinomio será el término L_i(x) en la interpolación de Lagrange.
    """
    
    n = len(values)# Asigna el número de datos
    L, x , y = sp.symbols('L, x, y') #Creamos las variables para el cálculo simbólico
    
    #Inicializamos las variables
    y = 1
    L = 1

    #Este ciclo calcula L_i(x) haciendo los productos en la fórmula
    for j in range(n):
        if j != i-1:
            y = (x - values[j])/float((values[i-1] - values[j]))
            L *= y
    #Expandimos el resultado haciendo todos los productos
    L = sp.expand(L)
    return L
    
# Importamos los datos de un archivo txt
data = pd.read_csv(r'C:\Users\cesar.avila\Documents\IPN\Semestre 7\Física Numérica\Tareas\Tarea6\DatosLagrange.txt'
                   ,header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
E = data.iloc[:,0]
f = data.iloc[:,1]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
data_x = np.float64(E)
data_y = np.float64(f)

#Número de datos
n = len(data_x)

# Definimos variables para la obtendión del polinomio y los inicializamos
g, x = sp.symbols('g, x')
g = 0.

#Calculamos el polinomio completo de Lagrange de grado n-1
for i in range(n):
    g += data_y[i]*Lx(i+1,data_x)

#Convertimos el g a un objeto tipo polinomio y guardamos sus coeficientes en una lista
p = sp.Poly(g)
coef = p.rep.rep

#Creamos el polinomio con numpy para un mejor manejo
pol = poly.Polynomial(coef[::-1])

#Creamos un arrgelo que contendrá los puntos a ser evaluados  en el eje x
valores_E = np.arange(-2.0, 205, 5)

#Graficamos el polinomio y los ptos experimentales
fig = plt.figure(figsize = (10,8))
plt.plot(valores_E, pol(valores_E), 'b', data_x, data_y, 'ro')
plt.title('Interpolación de Lagrange')
plt.grid()
plt.xlabel('$E_i$  [MeV]')
plt.ylabel('$f(E_i)$  [MeV]')
#plt.show()

#Obtengamos el máximo del polinomio de Lagrange con el criterio de la primer derivada
df = sp.diff(g,x) #1ra derivada

critic_points = list(sp.solveset(sp.Eq(df,0))) #Obtenemos las raices de la 1ra derivada

print('Las raíces de la 1ra derivada son:\n',critic_points)#Imprimimos las raíces
print('-'*50)

#obtengamos la energía de resonancia y gamma
E_r = critic_points[2]
mitad_max = sp.sympify(g).subs(x,E_r) /2. #Obtenemos el máximo por la mitad

#Raices de la ecuación g(x) = E_r
candidatos = list(sp.solveset(sp.Eq(g,mitad_max)))
print('Soluciones a la ecuación g(x) = E_r\n',candidatos)

#Valor de Gamma
gamma = candidatos[3]- candidatos[2]

#Errores porcentuales
errorE_r = 100.*abs(78 - E_r)/78.
errorGamma = 100.*abs(55-gamma)/55.

print('-'*50)
print(f'Energía de resonancia: {E_r} | Con un error de: {errorE_r} %')
print(f'Gamma: {gamma} | Con un error de: {errorGamma} %')

