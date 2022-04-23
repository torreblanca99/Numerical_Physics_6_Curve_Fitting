# -*- coding: utf-8 -*-
"""
Ejercicio 5. Ajustando el espectro de un cuerpo negro}
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Leemos el archivo
data = pd.read_csv(r'C:\Users\cesar.avila\Documents\IPN\Semestre 7\Física Numérica\Tareas\Tarea6\COBE.txt'
                   ,header=0,delim_whitespace = True)

# Extraccion de datos 
Nu = data.iloc[:,0]
I = data.iloc[:,1]
SigmaI = data.iloc[:,2]

# Manejo de los datos a flotantes y reacomodo de unidades
nu_float = np.float64(Nu)
I_float = np.float64(I)
SigmaI_float = np.float64(SigmaI)/1000 

fig, ax = plt.subplots()
ax.errorbar(nu_float, I_float, yerr=SigmaI_float) # Grafica la incertidumbre en cada punto
ax.set_xlabel('$\\nu$ [Hz]')
ax.set_ylabel('$I(\\nu, T)$ [MJy/sr]')
ax.set_title('Experimento COBE')
ax.grid()
plt.show()


# Valores de las contantes (Usaremos unidades del SI)
h = 6.6261E-34 # Cte de Planck en J*s 
c = 2.9979E8 # Vel. luz en m/s
k =1.3806E-23 # Cte de Boltzmann en J/Kelvin

#Transformamos los valores de la irradiancia y la frecuencia a unidades SI
nu_SI = nu_float*2.9979E10
I_SI = I_float*1E-20
SigmaI_SI = Error_float*1E-20

#Guadamos los valores para el ajuste de recta en tres listas distintas
logaritmos = []
incertidumbre_ln = []
eje_x = np.float64(h*nu_SI/k)

for i in range(len(nu_SI)):
    valor1 = np.log((2*h*nu_SI[i]**3)/(c**2 * I_SI[i]) +1 )
    valor2 = (2*h*nu_SI[i]**3 / c**2 * I_SI[i]**2) *SigmaI_SI[i] / ((2*h*nu_SI[i]**3)/(c**2 * I_SI[i]) +1)
    logaritmos.append(np.float64(valor1))
    incertidumbre_ln.append(np.float64(valor2))


fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(eje_x, logaritmos, yerr=incertidumbre_ln) # Grafica la incertidumbre en cada punto
ax.set_xlabel('$\dfrac{h \\nu}{k}$ [K]')
ax.set_ylabel('$ln (\dfrac{2 h}{c^2} \dfrac{\\nu^3}{I} + 1)$ [sr]')
ax.set_title('Experimento COBE: Grafico logaritmico')
ax.grid()
plt.show()

# Obtengamos el ajuste por mínimos cuadrados
S = 0.
Sx = 0.
Sy = 0.
Sxx = 0.
Sxy = 0.

for i in range(len(logaritmos)):
    S += 1/incertidumbre_ln[i]**2
    Sx += eje_x[i]/incertidumbre_ln[i]**2
    Sy += logaritmos[i]/incertidumbre_ln[i]**2
    Sxx += eje_x[i]**2 / incertidumbre_ln[i]**2
    Sxy += eje_x[i]*logaritmos[i] / incertidumbre_ln[i]**2

# Solución al sistema de ecuaciones  
matriz1 = np.array([[S, Sx],[Sx, Sxx]])
matriz2 = np.array([[Sy], [Sxy]])

inversa = np.linalg.inv(matriz1)
resultado = inversa.dot(matriz2)

a1, a2 = float(resultado[0]), float(resultado[1])

print(f'La temperatura es: {1/a2} K')