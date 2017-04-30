import numpy as np
import scipy.io as sio

# Carga los datos
data = sio.loadmat('dataAcs')

# Se definen los datos por palabra
dataCalor = data['dataCalor']
dataCarino = data['dataCarino']
dataSushi = data['dataSushi']

# Se definen las etiquetas por palabra
labelCalor = data['labelCalor'][:,1:2]
labelCarino = data['labelCarino'][:,1:2]
labelSushi = data['labelSushi'][:,1:2]

# Se intercambian los ejes para 
# (canales, muestras, ventanas) = (10,900,206)
dataCalor = np.swapaxes(dataCalor,2,0)
dataCalor = np.swapaxes(dataCalor,1,2)

dataCarino = np.swapaxes(dataCarino,2,0)
dataCarino = np.swapaxes(dataCarino,1,2)

dataSushi = np.swapaxes(dataSushi,2,0)
dataSushi = np.swapaxes(dataSushi,1,2)

# Se arreglan las etiquetas a 0 y 1
np.place(labelCalor, labelCalor<0, 0)
np.place(labelCarino, labelCarino<0, 0)
np.place(labelSushi, labelSushi<0, 0)

# Se forma un solo conjunto de datos por palabra (9000, 206)

tmpCalor = []
tmpLabelCalor = []

for channel in dataCalor:
	tmpCalor.extend(channel)
	tmpLabelCalor.extend(labelCalor)

tmpCalor = np.asarray(tmpCalor)
tmpLabelCalor = np.asarray(tmpLabelCalor)
#tmpLabelCalor = np.squeeze(tmpLabelCalor)

tmpCarino = []
tmpLabelCarino = []

for channel in dataCarino:
	tmpCarino.extend(channel)
	tmpLabelCarino.extend(labelCarino)

tmpCarino = np.asarray(tmpCarino)
tmpLabelCarino = np.asarray(tmpLabelCarino)
#tmpLabelCarino = np.squeeze(tmpLabelCarino)

tmpSushi = []
tmpLabelSushi = []

for channel in dataSushi:
	tmpSushi.extend(channel)
	tmpLabelSushi.extend(labelSushi)

tmpSushi = np.asarray(tmpSushi)
tmpLabelSushi = np.asarray(tmpLabelSushi)
#tmpLabelSushi = np.squeeze(tmpLabelSushi)

tmp_acs = np.concatenate(
	(tmpCalor,tmpCarino,tmpSushi))

tmp_label = np.concatenate(
	(tmpLabelCalor,tmpLabelCarino,tmpLabelSushi))

data_acs = np.concatenate((tmp_acs, tmp_label),
	axis=1)

np.save('data_ACS', data_acs)