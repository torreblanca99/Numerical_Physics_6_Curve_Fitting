# -*- coding: utf-8 -*-
"""
Ejercicio 4: Fuente de Voltaje
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
    
    g, a1, a2 = sp.symbols('g, a1, a2') #Creamos las variables para el cálculo simbólico
    
    #Inicializamos las variables
    g = 0.

    #Esta parte calculo g(x_i) usando la fórmula
    
    g = a1*sp.exp(-a2*values[i])
    g = sp.expand(g)
    return g

def f_vector(f1,f2, x:float, y:float):
    """Esta función calcula el f vector en la solución de Newton-Rapghson y
    evalúa a las funciones f1, f2 en los puntos x,y. 
    Regresa una matriz."""
    
    a1, a2 = sp.symbols('a1, a2')

    valor1 = f1.subs([(a1,x), (a2,y)])
    valor2 = f2.subs([(a1,x), (a2,y)])

    
    return np.array([[valor1], [valor2]])
    
def jacobiano(f1,f2, x:float, y:float):
    """Esta función calcula la matriz Jacobiana en la solución de Newton-Rapghson 
    y evalúa a las derivadas parciales de f1, f2, f3 en los puntos x,y,z. 
    Regresa una matriz."""
    
    df, a1, a2 = sp.symbols('df, a1, a2')
    
    variables = [a1, a2]
    funciones = [f1, f2]
    
    #Saca la derivada y evalua los valores de a1,a2 = x,y
    J = np.array([[np.float64(sp.diff(funciones[i], variables[j]).subs([(a1,x), (a2,y)])) 
                 for j in range(len(variables))] for i in range(len(variables))])
    
    return J
    
# Importamos los datos de un archivo txt
data = pd.read_csv(r'C:\Users\cesar.avila\Documents\IPN\Semestre 7\Física Numérica\Tareas\Tarea6\DatoVoltage.txt'
                   ,header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
t = data.iloc[:,0]
V = data.iloc[:,1]
sigma = data.iloc[:,2]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
data_x = np.float64(t)
data_y = np.float64(V)
sigma_y = np.float64(sigma)


#Guardamos el número de datos 
n = len(data_x)

# Definimos variables para la obtendión g(x) y los inicializamos
g, a1, a2 = sp.symbols('g, a1, a2')
g = 0.

# Definimos variables para la obtendión f_1, f2, f3 y los inicializamos
f1, f2, aux = sp.symbols('f1, f2, aux')
f1 = 0.
f2 = 0.

for i in range(n):
    #Cálculo de g(x_i)
    g = gx(i,data_x)
    #Cálculo de  f1
    aux = (data_y[i] - g) * sp.exp(-a2*data_x[i]) / sigma_y[i]**2
    aux = sp.expand(aux)
    f1 += aux
    
    #Cálculo de  f2
    aux = (data_y[i] - g) * data_x[i] * sp.exp(-a2*data_x[i]) / sigma_y[i]**2
    aux = sp.expand(aux)
    f2 += aux
 
#Semillas de a1,a2,  
x = 1.
y = 0.00001

# Longitud a1+Delta_a1,a2+Delta_a2, a3+Delta_a3

a = [0., 0.]
a_new = [x, y]

# Definimos el error para detener la iteración de Newton-Raphson
error = 10**(-20)
i=1

condicion1 = abs(a_new[0] - a[0])

#Iteraciones Newton-Raphson:
while  (condicion1 > error):
    A = f_vector(f1, f2, a_new[0], a_new[1]) 
    B = jacobiano(f1, f2, a_new[0], a_new[1])
    inv_B = np.linalg.inv(B)
      
    Delta_a = -inv_B.dot(A)
    
    #Guardamos los viejos valores
    a[0] = a_new[0]
    a[1] = a_new[1]
    
    #Asignamos los nuevos valores
    a_new[0] += np.float64(Delta_a[0][0])
    a_new[1] += np.float64(Delta_a[1][0])
    
    condicion1 = abs(a_new[0] - a[0])
    print(f'No. iteración: {i}')
    print(f'Condición: {condicion1}')
    i+=1

print('-'*50)    
print(f'Valores:\na_1 = {a_new[0]}\na_2 = {a_new[1]}')


# Cálculo de chi^2 con los valores de a_1 = a_new[0] y a_2 = a_new[1]

chi = sp.symbols('chi') # Creamos las variables para el cálculo simbólico
    
#Inicializamos las variables
chi = 0.

for i in range(len(data_x)):
    aux = ((data_y[i] - a_new[0]*np.exp(- a_new[1] *data_x[i])) / sigma_y[i])**2 
    chi += aux
    
print('-'*50)
print(f'Valor de chi^2: {chi}')


# Hagamos una gráfica semilog para obtener ajuste por mínimos cuadrados a una recta

logaritmos = np.log(data_y)
incertidumbre_ln = sigma_y/data_y

# Graficquemos y obtengamos el ajuste 
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(data_x, logaritmos, yerr=incertidumbre_ln) # Grafica la incertidumbre en cada punto
ax.set_ylabel('$ln V(t))$ [Volts]')
ax.set_xlabel('$t$ [ns]')
ax.set_title('Grafico semilog decaimiento del voltaje')
ax.grid()
plt.show()

# Obtengamos el ajuste de recta por mínimos cuadrados
S = 0.
Sx = 0.
Sy = 0.
Sxx = 0.
Sxy = 0.

for i in range(len(logaritmos)):
    S += 1/incertidumbre_ln[i]**2
    Sx += data_x[i]/incertidumbre_ln[i]**2
    Sy += logaritmos[i]/incertidumbre_ln[i]**2
    Sxx += data_x[i]**2 / incertidumbre_ln[i]**2
    Sxy += data_x[i]*logaritmos[i] / incertidumbre_ln[i]**2

# Solución al sistema de ecuaciones  
matriz1 = np.array([[S, Sx],[Sx, Sxx]])
matriz2 = np.array([[Sy], [Sxy]])

inversa = np.linalg.inv(matriz1)
resultado = inversa.dot(matriz2)
# b = ordenada al origen y m = pendiente
b, m = float(resultado[0]), float(resultado[1])

print('-'*50)
print(f'Los valores de Gamma y V_0 por este ajuste son:')
print(f'Gamma: {-m}')
print(f'V_0: {np.exp(b)}')
