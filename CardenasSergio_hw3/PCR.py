import numpy as np

#Leer datos
data = np.array(np.genfromtxt('WDBC.dat', delimiter=', ', dtype=None))
data = np.array([s.split(',') for s in data]).T[1:] #Se transpone para que cada fila sea una variable, se eliminan los ids
diagnoses = data[0]
data = data[1:].astype(float)
