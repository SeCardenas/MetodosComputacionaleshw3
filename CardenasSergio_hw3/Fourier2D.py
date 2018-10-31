import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from matplotlib.colors import LogNorm

#Importar imagen y aplicar transformada 2D
image = plt.imread('arbol.png').astype(float)
imfft = fft2(image, axes=(0,1))

plt.figure()
plt.imshow(np.abs(imfft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier de la imagen')
plt.savefig('CardenasSergio_FT2D.pdf')
plt.close()

#Al ver la grafica de la transformada, se pueden observar pequenos puntos rojos en esta.
#Dado que el ruido es periodico, los coeficientes de estos deben estar concentrados en pequenos rangos de la transformada.
#Esto quiere decir que lo mas probable es que los puntos rojos vistos en la grafica sean el ruido.

#Se define la fraccion de coeficientes que se van a mantener, es decir, que no se van a considerar al aplicar el filtro (se puede variar).
fraction = 0.1
#filas (r) y columnas (c)
r,c = imfft.shape
#Copia de la transformada
imfft2 = np.copy(imfft)
#Se eliminan todos los coeficientes fuera del rango que tengan valores muy altos (se puede variar).
maxvalue = 1000
coef2rm = imfft2[int(r*fraction):int(r*(1-fraction))] #Coeficientes candidatos a ser eliminados.
imfft2[int(r*fraction):int(r*(1-fraction))] = coef2rm*(abs(coef2rm)<maxvalue) #Se eliminan todos los que sean mayores a cierto valor.
#Se repite el mismo procedimiento en la otra direccion.
coef2rm = imfft2[:, int(c*fraction):int(c*(1-fraction))]
imfft2[:, int(c*fraction):int(c*(1-fraction))] = coef2rm*(abs(coef2rm)<maxvalue)
