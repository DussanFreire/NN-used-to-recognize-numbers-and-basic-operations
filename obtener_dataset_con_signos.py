import h5py
from image_tool_box import ImageToolBox

# cargamos el conjunto entrenamiento
# Dussan
train_data = h5py.File(r"C:\Users\Dussan\Desktop\digitos.h5", "r")
test_data = h5py.File(r"C:\Users\Dussan\Desktop\digitos_test.h5", "r")

# Alvaro
# train_data = h5py.File(r"C:\Users\Lenovo\Downloads\modelado\practica_3\digitos.h5", "r")
# test_data = h5py.File(r"C:\Users\Lenovo\Downloads\modelado\practica_3\digitos_test.h5", "r")

# Transformando las dimensiones:
X_train = train_data["X"][:].reshape(60000, -1)
y_train = train_data["y"][:]

X_test = test_data["X"][:].reshape(10000, -1)
y_test = test_data["y"][:]


# Preparar datos, con los signos "+" y "-"
# 10 = "+" y 11 = "-"
# Agregaron los signos negativos
X_train, y_train, X_test, y_test = ImageToolBox.agregar_signo_al_data("signos negativos utilizables", 11, X_train, y_train, X_test, y_test)

# Agregando los signos positivos
X_train, y_train, X_test, y_test = ImageToolBox.agregar_signo_al_data("signos positivos utilizables", 10, X_train, y_train, X_test, y_test)

# guardar
arch = h5py.File("digitos_con_signos.h5", "w")
arch.create_dataset("X_train", data=X_train)
arch.create_dataset("y_train", data=y_train)
arch.create_dataset("X_test", data=X_test)
arch.create_dataset("y_test", data=y_test)
