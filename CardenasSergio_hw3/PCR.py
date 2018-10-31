import numpy as np

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