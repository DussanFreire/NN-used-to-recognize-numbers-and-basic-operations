from RedNeuronal import RedNeuronal
import h5py

# instanciamos nuestra red
r = RedNeuronal()

# configuramos topologia de la red, capas y unidades de activacion
r.capa1 = 784
r.capa2 = 256
r.capa3 = 64
r.capa4 = 12

# cargamos archivo ya generado en "obtener_dataset_con_signos.py"
arch = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")
X_train = arch["X_train"][:]
y_train = arch["y_train"][:]
X_test = arch["X_test"][:]
y_test = arch["y_test"][:]

# Inicializamos
r.inicializar_parametros()
r.fit(X_train, y_train)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# entrenamiento de la red
r.entrenar("new_theta.h5")

# probar test y mostrar resultados
r.obtener_presicion(X_test, y_test)
r.obtener_matriz_confusion_por_valor(X_test, y_test)

