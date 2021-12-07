import cv2
from RedNeuronal import RedNeuronal
from skimage.feature import hog
from image_tool_box import ImageToolBox

r = RedNeuronal()

# cargamos el modelo entrenado
r.cargar("thetas_hog.h5")

# cargamos la imagen: nuestro modelo debe reconocer los digitos
imagen = cv2.imread("../imagenes_de_prueba/nivel_5.jpg")

# convertir a blanco y negro
imagen_bn = ImageToolBox.obtener_imagen_en_blanco_y_negro(imagen)

# detectamos grupo e identificamos contornos
grupos = ImageToolBox.obtener_grupos(imagen_bn)

# enventanar candidatos
ventanas = [cv2.boundingRect(g) for g in grupos]

# incializar registro de operadores y numeros
lista_de_operadores = []
lista_de_numeros = []


# cada tupla de ventanas representa un candidato a digito
for cord in ventanas:
    # dibujar ventana a cada candidato
    ImageToolBox.dibujar_ventana_a_cada_candidato(imagen, cord)

    # obtener un espacio adicional para ajustar el area de un candidato de manera correcta
    espacio_vertical, espacio_horizonal, p1, p2 = ImageToolBox.obtener_espacio_adicional_de_una_img(cord)

    # ajustamos el area de cada candidato
    digito = imagen_bn[p1:p1 + espacio_vertical, p2:p2 + espacio_horizonal]

    if p2 > 0:
        # escalamos el candidato a una imagen 20x20
        digito = cv2.resize(digito, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        digitos = cv2.dilate(digito, (3, 3,))

        # Obtener descriptor
        descriptor = hog(digitos, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))

        # aplanamos el candidato
        aux = descriptor.flatten().reshape(1, -1)

        # Mostrar digito capturado, para debuggear
        # Image_tool_box.mostrar_digito(digito)

        # predecir
        prediccion = r.predecir(aux)

        # obtener simbolo y clasificar
        simbolo, es_una_operacion = ImageToolBox.obtener_simbolo(prediccion[0])

        # colocamos el digito reconocido en la imagen
        if es_una_operacion:
            ImageToolBox.colocar_operador_en_img(simbolo, cord, imagen)
            lista_de_operadores.append((simbolo, cord))
        else:
            ImageToolBox.colocar_num_en_img(simbolo, cord, imagen)
            lista_de_numeros.append((simbolo, cord))

        # mostrar informacion de prediccion
        ImageToolBox.mostrar_informacion_de_prediccion(prediccion, cord)

# marcamos los resultados
ImageToolBox.marcar_resultado(lista_de_numeros, lista_de_operadores, imagen)
# mostramos la imagen
cv2.imshow("Digitos", imagen)
cv2.waitKey()
