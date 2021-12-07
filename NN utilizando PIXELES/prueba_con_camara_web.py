import cv2
from RedNeuronal import RedNeuronal
from image_tool_box import ImageToolBox

# capturar la camara
camara = cv2.VideoCapture(1)
# instanciar la red neuronal
r = RedNeuronal()
# cargamos el modelo entrenado
r.cargar("theta_digitos.h5")

# mostrar la ventana de la captura de la camara
while True:
    # capturamos en imagen el frame de la camara
    _, imagen = camara.read()

    # convertir a blanco y negro
    imagen_bn = ImageToolBox.obtener_imagen_en_blanco_y_negro(imagen)

    # detectamos grupo e identificamos contornos
    grupos = ImageToolBox.obtener_grupos(imagen_bn)

    # enventanar candidatos
    ventanas = [cv2.boundingRect(g) for g in grupos]

    # incializar registro de operadores y numeros
    lista_de_operadores = []
    lista_de_numeros = []

    # enventanamos los candidatos
    for cord in ventanas:
        # dibujar ventana a cada candidato
        ImageToolBox.dibujar_ventana_a_cada_candidato(imagen, cord)

        # obtener un espacio adicional para ajustar el area de un candidato de manera correcta
        espacio_vertical, espacio_horizonal, p1, p2 = ImageToolBox.obtener_espacio_adicional_de_una_img(cord)

        # ajustamos el area de cada candidato
        digito = imagen_bn[p1:p1 + espacio_vertical, p2:p2 + espacio_horizonal]

        if p2 > 0 and p1 > 0 and espacio_vertical > 40 and espacio_horizonal > 40:

            # escalamos el candidato a una imagen 28x28
            digito = cv2.resize(digito, dsize=(28, 28), interpolation=cv2.INTER_AREA)

            # aplanamos al candidato
            aux = digito.flatten().reshape(1, -1)

            # pasamos el candidato aplanado por nuestro modelo
            prediccion = r.predecir(aux)
            simbolo, es_una_operacion = ImageToolBox.obtener_simbolo(prediccion[0])

            # colocamos el digito reconocido en la imagen
            if es_una_operacion:
                ImageToolBox.colocar_operador_en_img(simbolo, cord, imagen)
                lista_de_operadores.append((simbolo, cord))
            else:
                ImageToolBox.colocar_num_en_img(simbolo, cord, imagen)
                lista_de_numeros.append((simbolo, cord))

        # marcamos los resultados
        ImageToolBox.marcar_resultado(lista_de_numeros, lista_de_operadores, imagen)

        #mostramos la imagen
        cv2.imshow('Self', imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
