import cv2
import matplotlib.pyplot as pl
import os
import numpy


class ImageToolBox:
    @staticmethod
    def obtener_imagen_en_blanco_y_negro(imagen):
        # convertimos a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # filtro gausseano para mejorar la deteccion del contorno
        imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

        # convertir a blanco y negro
        ret, imagen_bn = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)
        return  imagen_bn

    @staticmethod
    def obtener_simbolo(num):
        if num == 10:
            return "+", True
        if num == 11:
            return "-", True
        return str(num), False

    @staticmethod
    def marcar_resultado(numeros, operadores, imagen):
        for operador in operadores:
            pos_en_y_op = operador[1][1]
            pos_en_x_op = operador[1][0]
            simbolo = operador[0]
            num_izq = None
            num_der = None
            for num in numeros:
                if num[1][1] < pos_en_y_op < num[1][1] + num[1][3]:
                    if (num_izq is None or num_izq[1][0] < num[1][0]) and pos_en_x_op > num[1][0]:
                        num_izq = num
                        continue
                    if (num_der is None or num_der[1][0] > num[1][0]) and pos_en_x_op < num[1][0]:
                        num_der = num
                        continue
            if num_izq is not None and num_der is not None:
                resultado = 0
                if simbolo == '+':
                    resultado = int(num_izq[0]) + int(num_der[0])
                else:
                    resultado = int(num_izq[0]) - int(num_der[0])
                cv2.putText(imagen, str(resultado), (operador[1][0], operador[1][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    @staticmethod
    def obtener_grupos(imagen_bn):
        grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return grupos

    @staticmethod
    def dibujar_ventana_a_cada_candidato(imagen, cord):
        cv2.rectangle(imagen, (cord[0], cord[1]), (cord[0] + cord[2], cord[1] + cord[3]), (255, 0, 0), 2)

    @staticmethod
    def obtener_espacio_adicional_de_una_img(cord):
        if cord[3] < cord[2] * 0.5:
            espacio_vertical = int(cord[3] * 10)
            espacio_horizonal = int(cord[3] * 5)
        else:
            espacio_vertical = int(cord[3] * 1.6)
            espacio_horizonal = int(cord[3] * 1.6)
        p1 = int((cord[1] + cord[3] // 2) - espacio_vertical // 2)
        p2 = int((cord[0] + cord[2] // 2) - espacio_horizonal // 2)
        return espacio_vertical, espacio_horizonal, p1, p2

    @staticmethod
    def mostrar_digito(digito):
        pl.imshow(digito)
        pl.show()

    @staticmethod
    def colocar_operador_en_img(simbolo, cord, imagen):
        cv2.putText(imagen, simbolo, (cord[0], cord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (247, 255, 0))

    @staticmethod
    def colocar_num_en_img(simbolo, cord, imagen):
        cv2.putText(imagen, simbolo, (cord[0], cord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))


    @staticmethod
    def mostrar_informacion_de_prediccion(prediccion, cord):
        print("Se reconoce el digito ", prediccion[0], " con el ", prediccion[1] * 100, "% de certeza")
        print("- Distancia en y", cord[3], "Distancia en x", cord[2])
        print("- Posicion en y", cord[1], "Posicion en x", cord[0])

    @staticmethod
    def agregar_signo_al_data(folder_path, valor_output, x_tr, y_tr, x_te, y_te, cantidad=750):
        # folder_path = "signos positivos utilizables"
        imagenes = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        indice = 0

        while indice < cantidad:
            image_path = imagenes[indice]

            # sin transpuesta, no se necesita transpuesta para mostrar, los mas probable es que se necesite la transpuesta para cargar
            imagen = cv2.imread(image_path)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(-1)
            # imagen = imagen.reshape(-1)

            # concatenar una foto al data y concatenar el valor al output
            if indice < cantidad * 0.7:
                x_tr = numpy.concatenate((x_tr, numpy.array([imagen])), axis=0)
                y_tr = numpy.append(y_tr, valor_output)
            else:
                x_te = numpy.concatenate((x_te, numpy.array([imagen])), axis=0)
                y_te = numpy.append(y_te, valor_output)

            indice += 1
        return x_tr, y_tr, x_te, y_te
