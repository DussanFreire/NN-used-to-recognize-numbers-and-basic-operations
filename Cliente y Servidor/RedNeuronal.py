import h5py
import numpy
from scipy.optimize import minimize


class RedNeuronal:
    def __init__(self):
        self.X = None
        self.y = None
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None
        self.lambda_ = 1
        self.capa1 = None
        self.capa2 = None
        self.capa3 = None
        self.capa4 = None

        # cargar variables caracteristicas y objetivo
    def fit(self, x, y):
        self.X = x
        self.y = y

    # inicializacion de parametros
    def inicializar_parametros(self, epsilon=0.12):
        self.theta1 = numpy.random.rand(self.capa2, (self.capa1 + 1)) * 2 * epsilon - epsilon
        self.theta2 = numpy.random.rand(self.capa3, (self.capa2 + 1)) * 2 * epsilon - epsilon
        self.theta3 = numpy.random.rand(self.capa4, (self.capa3 + 1)) * 2 * epsilon - epsilon

    # funcion de activacion : sigmoide
    @staticmethod
    def sigmoide(z):
        return 1 / (1 + numpy.exp(-z))

    # derivada de la funcion sigmoide
    def derivada_sigmoide(self, z):
        return self.sigmoide(z) * (1 - self.sigmoide(z))

    # funcion costo: entropia cruzada, gradiente: back propagation
    def funcion_costo_gradiente(self, t):
        # re armado de los parametros segun topologia de la red
        t1 = numpy.reshape(t[0:self.capa2 * (self.capa1 + 1)], (self.capa2, (self.capa1 + 1)))
        t2 = numpy.reshape(t[self.capa2 * (self.capa1 + 1):self.capa2 * (self.capa1 + 1)+(self.capa3 * (self.capa2 + 1))], (self.capa3, (self.capa2 + 1)))
        t3 = numpy.reshape(t[self.capa2 * (self.capa1 + 1)+(self.capa3 * (self.capa2 + 1)):], (self.capa4, (self.capa3 + 1)))
        #numpy.reshape(t[self.capa2 * (self.capa1 + 1):], (self.capa3, (self.capa2 + 1)))

        # dimension de X
        m, n = self.X.shape

        # Computar h: front propagation
        a1 = numpy.concatenate([numpy.ones((m, 1)), self.X], axis=1)
        a2 = self.sigmoide(a1.dot(t1.T))
        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoide(a2.dot(t2.T))
        a3 = numpy.concatenate([numpy.ones((a3.shape[0], 1)), a3], axis=1)
        h = self.sigmoide(a3.dot(t3.T))

        # vectorizar y
        y_vec = numpy.eye(self.capa4)[self.y.reshape(-1)]

        # computo del parametro de regularizacion, para contrarestar el sobre ajuste
        param_reg = (self.lambda_ / (2 * m)) * (numpy.sum(numpy.square(t1[:, 1:])) +
                                                numpy.sum(numpy.square(t2[:, 1:])) +
                                                numpy.sum(numpy.square(t3[:, 1:])))

        # computo del costo: entropia cruzada
        j = - 1 / m * numpy.sum(numpy.log(h) * y_vec + numpy.log(1 - h) * (1 - y_vec)) + param_reg

        # computar el gradiente: back propagation
        # error en la ultima capa
        delta4 = h - y_vec

        # error en la penultima capa
        delta3 = delta4.dot(t3)[:, 1:] * self.derivada_sigmoide(a2.dot(t2.T))
        # error en la antepenultima capa
        delta2 = delta3.dot(t2)[:, 1:] * self.derivada_sigmoide(a1.dot(t1.T))

        # computo errores en las capas acumulado
        delta_acum_1 = delta2.T.dot(a1)
        delta_acum_2 = delta3.T.dot(a2)
        delta_acum_3 = delta4.T.dot(a3)
        # computo del gradiente
        grad1 = 1 / m * delta_acum_1
        grad2 = 1 / m * delta_acum_2
        grad3 = 1 / m * delta_acum_3

        #  penalización con el parametro de regularizacion
        grad1[:, 1:] = grad1[:, 1:] + (self.lambda_ / m) * t1[:, 1:]
        grad2[:, 1:] = grad2[:, 1:] + (self.lambda_ / m) * t2[:, 1:]
        grad3[:, 1:] = grad3[:, 1:] + (self.lambda_ / m) * t3[:, 1:]
        # concatenar gradientes
        grad = numpy.concatenate([grad1.flatten(), grad2.flatten(),grad3.flatten()])
        return j, grad

    # entrenamiento de la red, generacion de l modelo
    def entrenar(self, destino):
        # j_grad como funcion de alto orden
        j_grad = lambda p: self.funcion_costo_gradiente(p)

        # inicializar parametros
        theta_inical = numpy.concatenate([self.theta1.flatten(), self.theta2.flatten(), self.theta3.flatten()])

        # maximo de iteraciones
        opciones = {'maxiter': 800}

        # computamos parametros optimos, minimizacion de la funcion costo
        res = minimize(j_grad, theta_inical, jac=True, method="TNC", options=opciones)
        theta_optimo = res.x

        # armamos el modelo segun topologia de la red
        self.theta1 = numpy.reshape(theta_optimo[0:self.capa2 * (self.capa1 + 1)], (self.capa2, (self.capa1 + 1)))
        self.theta2 = numpy.reshape(theta_optimo[self.capa2 * (self.capa1 + 1):self.capa2 * (self.capa1 + 1) + (self.capa3 * (self.capa2 + 1))], (self.capa3, (self.capa2 + 1)))
        self.theta3 = numpy.reshape(theta_optimo[self.capa2 * (self.capa1 + 1)+(self.capa3 * (self.capa2 + 1)):], (self.capa4, (self.capa3 + 1)))

        # guardamos el modelo
        arch = h5py.File(destino, "w")
        arch.create_dataset("Theta1", data=self.theta1)
        arch.create_dataset("Theta2", data=self.theta2)
        arch.create_dataset("Theta3", data=self.theta3)

    # reconocimiento automatico: front propagation
    def predecir(self, imagen):

        a1 = numpy.concatenate([numpy.ones((1, 1)), imagen], axis=1)

        a2 = self.sigmoide(a1.dot(self.theta1.T))

        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoide(a2.dot(self.theta2.T))

        a3 = numpy.concatenate([numpy.ones((a3.shape[0], 1)), a3], axis=1)
        a4 = self.sigmoide(a3.dot(self.theta3.T)).T

        return a4.argmax(), a4[a4.argmax()]

    # cargar el modelo entrenado
    def cargar(self, archivo):
        arch = h5py.File(archivo, "r")
        self.theta1 = arch["Theta1"][:]
        self.theta2 = arch["Theta2"][:]
        self.theta3 = arch["Theta3"][:]

    def obtener_presicion(self, X_test, y_test):
        cant_elementos, _ = X_test.shape
        aciertos = 0
        fallas = 0

        for i in range(0, cant_elementos):
            num_real = y_test[i]
            prediction, _ = self.predecir(X_test[i][:].reshape(1, -1))
            if num_real == prediction:
                aciertos += 1
                continue
            else:
                fallas += 1
                continue

        print("Entrenamiento de la red, se genero el modelo: ")
        print("Capa 1: ", self.theta1.shape)
        print("Capa 2: ", self.theta2.shape)
        print("Capa 3: ", self.theta3.shape)
        print("Aciertos: ", aciertos, "; Porcentaje: ", (aciertos / (aciertos + fallas)) * 100)
        print("Fallos: ", fallas, "; Porcentaje: ", (fallas / (aciertos + fallas)) * 100)

    def obtener_matriz_confusion_por_valor(self, X_test, y_test):
        cant_elementos = y_test.shape[0]
        lista_predicciones = []
        for num in range(0, 12):
            verdadero_pos = 0
            falso_pos = 0
            verdadero_neg = 0
            falso_neg = 0
            for i in range(0, cant_elementos):
                num_real = y_test[i]
                num_pred, _ = self.predecir(X_test[i][:].reshape(1, -1))
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
            print(f"- Presicion: {presicion}")
            print(f"- Exhaustividad: {exhaustividad}")
            print(f"- F1: {f_uno}")
            print(f"- Verdaderos positivos: {verdadero_pos}")
            print(f"- Verdaderos negativos: {verdadero_neg}")
            print(f"- Falsos positivos: {falso_pos}")
            print(f"- Falsos negativos: {falso_neg}")





