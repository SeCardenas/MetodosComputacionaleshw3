import numpy as np
import matplotlib.pyplot as plt

#Covarianza de dos conjuntos de datos
def cov(x1, x2):
	return np.sum((x1-np.mean(x1))*(x2-np.mean(x2)))/(len(x1)-1)

#Matriz de covarianza
def covMatrix(m):
	n = len(m)
	covm = np.zeros([n,n])
	for i in range(n):
		for j in range(n):
			covm[i,j] = cov(m[i], m[j])
	return covm

#Leer datos
data = np.array(np.genfromtxt('WDBC.dat', delimiter=', ', dtype=None))
data = np.array([s.split(',') for s in data]).T[1:] #Se transpone para que cada fila sea una variable, se eliminan los ids
diagnoses = data[0]
data = data[1:].astype(float)

#Normalizar y centrar
for i in range(len(data)):
	data[i] = (data[i]-np.mean(data[i]))/np.std(data[i])

#Matriz de covarianza de los datos normalizados
covm = covMatrix(data)
print 'Matriz de covarianza:'
print covm

eigvals, eigvecs = np.linalg.eig(covm)
print 'a continuacion se listan todos los autovalores y autovectores'
for i in range(len(eigvals)):
	print 'El autovector\n', eigvecs[:,i], '\ntiene como autovalor', eigvals[i], '\n'

#Parametros principales
print 'Las componentes principales son las siguientes:'
print eigvecs[0]
print eigvecs[1]

print 'De acuerdo con esto, uno de los parametros mas importates es el numero 17, que es el que tiene el mayor valor en la primera componente principal. Del segundo autovector, el parametro con mas importancia en esta componente es cuarto.'

#Proyecciones
Proy1B = np.dot(eigvecs[0],data[:,diagnoses=='B'])
Proy2B = np.dot(eigvecs[1],data[:,diagnoses=='B'])
Proy1M = np.dot(eigvecs[0],data[:,diagnoses=='M'])
Proy2M = np.dot(eigvecs[1],data[:,diagnoses=='M'])

#Grafica
plt.figure()
plt.scatter(Proy1B, Proy2B, c='b', alpha=0.6, label='Benignos')
plt.scatter(Proy1M, Proy2M, c='r', alpha=0.6, label='Malignos')
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('CardenasSergio_PCA.pdf')
plt.close()
