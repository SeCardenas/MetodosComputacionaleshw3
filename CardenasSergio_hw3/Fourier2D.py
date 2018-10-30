import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from matplotlib.colors import LogNorm

#Importar imagen y aplicar transformada 2D
image = plt.imread('arbol.png').astype(float)
imfft = fft2(image, axes=(0,1))

plt.figure()
plt.imshow(np.abs(imfft))
plt.colorbar()
plt.title('Transformada de Fourier de la imagen')
plt.savefig('CardenasSergio_FT2D.pdf')
plt.close()
