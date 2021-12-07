import cv2
from tensorflow.keras.models import load_model
from image_tool_box import ImageToolBox
import h5py


def obtener_presicion(X_test, y_test, modelo):
    cant_elementos, _ = X_test.shape
    aciertos = 0
    fallas = 0

    for i in range(0, cant_elementos):
        num_real = y_test[i]
        prediction = modelo.predict(X_test[i][:].reshape(1, -1)).argmax()
        if num_real == prediction:
            aciertos += 1
            continue
        else:
            fallas += 1
            continue

    print("Entrenamiento de la red, se genero el modelo: ")
    print("Aciertos: ", aciertos, "; Porcentaje: ", (aciertos / (aciertos + fallas)) * 100)
    print("Fallos: ", fallas, "; Porcentaje: ", (fallas / (aciertos + fallas)) * 100)


def obtener_matriz_confusion_por_valor(X_test, y_test, modelo):
    cant_elementos = y_test.shape[0]
    for num in range(0, 12):
        verdadero_pos = 0
        falso_pos = 0
        verdadero_neg = 0
        falso_neg = 0
        for i in range(0, cant_elementos):
            num_real = y_test[i]
            num_pred = modelo.predict(X_test[i][:].reshape(1, -1)).argmax()
            if num_real == num:
                if num_real == num_pred:
                    verdadero_pos += 1
                else:
                    falso_neg += 1
            if num_real != num:
                if num_pred == num:
                    falso_pos += 1
                else:
                    verdadero_neg += 1
        caracter = num
        if num == 10:
            caracter = "+"
        if num == 11:
            caracter = "-"
        presicion = verdadero_pos/(verdadero_pos + falso_pos)
        exhaustividad = verdadero_pos/(verdadero_pos + falso_neg)
        f_uno = 2*((presicion*exhaustividad)/(presicion+exhaustividad))
        print(f"CARACTER: {caracter}")
        print(f"- Falsos positivos: {falso_pos}")
        print(f"- Falsos negativos: {falso_neg}")
        print(f"- Verdaderos positivos: {verdadero_pos}")
        print(f"- Verdaderos negativos: {verdadero_neg}")
        print(f"- Presicion: {presicion}")
        print(f"- Exhaustividad: {exhaustividad}")
        print(f"- F1: {f_uno}")

data = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")


X_train = data["X_train"][:]
y_train = data["y_train"][:]
X_test = data["X_test"][:]
y_test = data["y_test"][:]

#Por cada valor
mi_modelo = load_model("../Thethas Almacenados/mejores thetas tensor/thetas_tensor_epocas_20.h5")

# mostrar rrendimiento en consola

obtener_matriz_confusion_por_valor(X_test, y_test,mi_modelo)
obtener_presicion(X_test, y_test,mi_modelo)
