from RedNeuronal import RedNeuronal
import h5py

# direccion alvaro
# data = h5py.File(r"C:\Users\Lenovo\Downloads\modelado\practica_3\digitos.h5", "r")
# direccion dussan
data = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")


X_train = data["X_train"][:]
y_train = data["y_train"][:]
X_test = data["X_test"][:]
y_test = data["y_test"][:]
r= RedNeuronal()

#Por cada valor
r.capa1 = 784
r.capa2 = 256
r.capa3 = 64
r.capa4 = 12
r.inicializar_parametros()
r.fit(X_train, y_train)
r.cargar("theta_digitos.h5")

# mostrar rrendimiento en consola
r.obtener_presicion(X_test, y_test)
r.obtener_matriz_confusion_por_valor(X_test, y_test)