import cv2
from RedNeuronal import RedNeuronal

r = RedNeuronal()

# cargamos el modelo entrenado
r.cargar("NN utilizando PIXELES/theta_digitos.h5")

# cargamos la imagen: nuestro modelo debe reconocer los digitos
imagen = cv2.imread("Operadores/Fotos de operadores/fotos de signos negativos/i5.jpg")
# convertimos a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
# filtro gaussiano para mejorar la deteccion del contorno
imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
# convertir a blanco y negro
ret, imagen_bn = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)
# detectamos grupos identificando contornos
grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# enventanamos los candidatos
ventanas = [cv2.boundingRect(g) for g in grupos]

# cada tupla de ventanas representa un candidato a digito

signo_enventanado = None

for v in ventanas:
    # dibujar ventana a cada candidato

    # si el largo no es mucho, podriamos asumir que se trata de una resta
    cv2.rectangle(imagen, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), (255, 0, 0), 2)
    # ajustamos el area de cada candidato
    print(v[3],v[1])
    if v[3] - v[1] < 5:
        # v[3]<5
        espacio_vertical = int(v[3] * 10)
        espacio_horizonal = int(v[3] * 9)
    else:
        espacio_vertical = int(v[3] * 1.6)
        espacio_horizonal = int(v[3] * 1.6)
    p1 = int((v[1] + v[3] // 2) - espacio_vertical // 2)
    p2 = int((v[0] + v[2] // 2) - espacio_horizonal // 2)
    # capturamos cada candidato a digito
    digito = imagen_bn[p1:p1 + espacio_vertical, p2:p2 + espacio_horizonal]
    first = True
    if p2 > 0:

        # escalamos el candidato a una imagen 28x28
        digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
        digitos = cv2.dilate(digito, (3, 3, ))
        # aplanamos el candidato
        aux = digito.T.reshape(1, -1)
        if first:
            signo_enventanado = digito
            first = False

        cv2.putText(imagen, str(":v"), (v[0], v[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
cv2.imwrite("Operadores/Operadores en blanco y negro/signos negativos en blanco y negro/negativo_8.jpg", signo_enventanado)
# mostramos la imagen
cv2.imshow("Digitos", imagen)
cv2.waitKey()
