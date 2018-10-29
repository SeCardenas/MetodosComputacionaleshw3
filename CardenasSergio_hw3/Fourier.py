import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq

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

dt = signalx[1] - signalx[0]
plt.figure()
plt.plot(fftfreq(len(signaly), dt), abs(dft(signaly)))
plt.savefig('CardenasSergio_TF.pdf')
