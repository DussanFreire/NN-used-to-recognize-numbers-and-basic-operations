from RedNeuronal import RedNeuronal
import numpy
import h5py
from skimage import feature

# instanciamos nuestra red
r = RedNeuronal()

# configuramos topologia de la red, capas y unidades de activacion
r.capa1 = 36
r.capa2 = 64
r.capa3 = 38
r.capa4 = 12

# cargamos archivo ya generado en "obtener_dataset_con_signos.py"
arch = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")
X_train = arch["X_train"][:]
y_train = arch["y_train"][:]
X_test = arch["X_test"][:]
y_test = arch["y_test"][:]

# obtenemos descriptores
lista_hog = []
for x in X_train:
    descriptor = feature.hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
    lista_hog.append(descriptor)

descriptores = numpy.array(lista_hog)

# se inicializan parametros de la red
r.inicializar_parametros()
r.fit(descriptores, y_train)

# entrenamiento de la red
r.entrenar("new_theta_hog")

# obtenemos los descriptores para los tests
lista_hog = []
for x in X_test:
    descriptor = feature.hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
    lista_hog.append(descriptor)

descriptores = numpy.array(lista_hog)
print(descriptores.shape, y_test.shape)
# probar test y mostrar resultados
r.obtener_presicion(descriptores, y_test)
r.obtener_matriz_confusion_por_valor(descriptores, y_test)