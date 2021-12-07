import numpy
from RedNeuronal import RedNeuronal
import h5py
from skimage import feature

# direccion alvaro
# data = h5py.File(r"C:\Users\Lenovo\Downloads\modelado\practica_3\digitos.h5", "r")
# direccion dussan
data = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")

X_train = data["X_train"][:]
y_train = data["y_train"][:]
X_test = data["X_test"][:]
y_test = data["y_test"][:]

# inicializar red
r= RedNeuronal()
r.capa1 = 36
r.capa2 = 64
r.capa3 = 32
r.capa4 = 12
r.cargar("thetas_hog.h5")
lista_hog = []
# obetener descriptor
for x in X_test:
    descriptor = feature.hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
    lista_hog.append(descriptor)
descriptores = numpy.array(lista_hog)
# mostrar rendimiento en consola
r.obtener_presicion(descriptores, y_test)
r.obtener_matriz_confusion_por_valor(descriptores, y_test)