import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, ifft
import scipy.interpolate as interp

def dft(signal):
	N = len(signal)
	dft = np.zeros(N,dtype=complex)
	Z = np.exp(-2j*np.pi/N)
	for k in range(N):
		ak = 0
		for n in range(N):
			ak += (Z**(n*k))*signal[n]
		dft[k] = ak
	return dft

signal_data = np.loadtxt('signal.dat', delimiter=',')
incomplete_data = np.loadtxt('incompletos.dat', delimiter=',')

signalx = signal_data[:,0]
signaly = signal_data[:,1]
incompletex = incomplete_data[:,0]
incompletey = incomplete_data[:,1]

plt.figure()
plt.plot(signalx, signaly)
plt.title('Signal')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.savefig('CardenasSergio_signal.pdf')
plt.close()

dt = signalx[1] - signalx[0]
freqs = fftfreq(len(signaly), dt)
transform = dft(signaly)
#print 1/(dt*len(signaly))
plt.figure()
plt.plot(freqs, abs(transform))
plt.title('Transformada de Fourier de Signal')
plt.xlabel('f')
plt.ylabel('|Y(f)|')
plt.savefig('CardenasSergio_TF.pdf')
plt.close()

#140, 210, 385 Hz
print 'Las frecuencias principales son:'
for i in freqs[abs(transform)>300]:
	if i>0:
		print i, 'Hz'

#filtro
fc = 1000
dft_filtered = transform*(abs(freqs)<=fc)
signal_filtered = ifft(dft_filtered)
plt.figure()
plt.plot(signalx, signal_filtered)
plt.title('Signal filtrada')
plt.xlabel('t')
plt.ylabel('y_filtered(t)')
plt.savefig('CardenasSergio_filtrada.pdf')
plt.close()

#Transformada de datos incompletos
print 'Al observar los datos del archivo incompletos.dat, se nota que cada muestra no esta separada por el mismo intervalo de tiempo. Para que la Transformada de Fourier Discreta funcione, es necesario que los datos esten equidistantes en tiempo.'

#interpolacion
fcuadratic = interp.interp1d(incompletex, incompletey,kind = "quadratic")
fcubic = interp.interp1d(incompletex, incompletey,kind = "cubic")

interpx = np.linspace(incompletex[0], incompletex[-1], 512)
ycuadratic = fcuadratic(interpx)
ycubic = fcubic(interpx)

dftcuadratic = dft(ycuadratic)
dftcubic = dft(ycubic)
dt2 = interpx[1]-interpx[0]
interpfreq = fftfreq(len(interpx), dt2)

#graficas de dft
plt.figure()
plt.subplot('311')
plt.plot(freqs, abs(transform))
plt.title('Transformada signal original')
plt.xlabel('f')
plt.ylabel('|Y(f)|')
plt.subplot('312')
plt.plot(interpfreq, abs(dftcuadratic))
plt.title('Transformada interpolacion cuadratica')
plt.xlabel('f')
plt.ylabel('|Y(f)|')
plt.subplot('313')
plt.plot(interpfreq, abs(dftcubic))
plt.title('Transformada interpolacion cubica')
plt.xlabel('f')
plt.ylabel('|Y(f)|')
plt.savefig('CardenasSergio_TF_interpola.pdf')
plt.close()

#Diferencias
print 'Una diferencia que se encontro es que la frecuencia de 385 Hz pierde aplitud en ambas interpolaciones. Tambien se observa que en las dfts de las interpolaciones, las frecuencias bajas son mas cercanas a cero (esto tiene sentido, pues las funciones cuadraticas y cubicas son suaves). Una ultima diferencia es que el ruido de "bajas frecuencias" (entre 400Hz y 2000 Hz) aumento su amplitud con respecto a la senal original.'

#filtros
fc2 = 500
signalfiltered2 = transform*(abs(freqs)<=fc2)
cuadraticfiltered = dftcuadratic*(abs(interpfreq)<=fc)
cuadraticfiltered2 = dftcuadratic*(abs(interpfreq)<=fc2)
cubicfiltered = dftcubic*(abs(interpfreq)<=fc)
cubicfiltered2 = dftcubic*(abs(interpfreq)<=fc2)

#Graficas
plt.figure()
plt.subplot('211')
plt.plot(signalx, signal_filtered, label='original filtrada')
plt.plot(interpx, ifft(cuadraticfiltered), label='cuadratica filtrada')
plt.plot(interpx, ifft(cubicfiltered), label='cubica filtrada')
plt.legend(loc=0, fontsize='small')
plt.title('Filtro en 1000 Hz')
plt.ylim(-6, 8)
plt.subplot('212')
plt.plot(signalx, ifft(signalfiltered2), label='original filtrada')
plt.plot(interpx, ifft(cuadraticfiltered2), label='cuadratica filtrada')
plt.plot(interpx, ifft(cubicfiltered2), label='cubica filtrada')
plt.legend(loc=0, fontsize='small')
plt.title('Filtro en 500 Hz')
plt.ylim(-6, 8)
plt.savefig('CardenasSergio_2Filtros.pdf')
plt.close()
