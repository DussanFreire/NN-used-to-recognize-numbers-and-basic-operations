from flask import Flask, request, Response
import numpy
import cv2
import jsonpickle
from RedNeuronal import RedNeuronal

# instanciar el servidor
app = Flask(__name__)
# instanciamos nuestra red neuronal
r = RedNeuronal()
# cargamos nuestro modelo
r.cargar("../NN utilizando PIXELES/theta_digitos.h5")
def obtener_simbolo(num):
	if num == 10:
		return "+", True
	if num == 11:
		return "-", True
	return str(num), False


def obtener_resultado(numeros, operadores):
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
			return (int(num_izq[0]), int(num_der[0]), simbolo, resultado)


@app.route('/api/test', methods=['POST'])
def test():
	re = request
	# convertir cadena a matriz de numpy
	nparr = numpy.fromstring(re.data, numpy.uint8)
	# decodificar la imagen
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# transformar a escala de grises
	imagen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# filtro gausseano para mejorar la deteccion del contorno
	imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
	# convertir a blanco y negro
	ret, imagen_bn = cv2.threshold(imagen, 90, 255, cv2.THRESH_BINARY_INV)
	# detectamos grupo e identificamos contornos
	grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	ventanas = [cv2.boundingRect(g) for g in grupos]
	lista_de_operadores= []
	lista_de_numeros = []
	for cord in ventanas:
		if cord[3] < cord[2] * 0.5:
			espacio_vertical = int(cord[3] * 10)
			espacio_horizonal = int(cord[3] * 5)
		else:
			espacio_vertical = int(cord[3] * 1.6)
			espacio_horizonal = int(cord[3] * 1.6)
		p1 = int((cord[1] + cord[3] // 2) - espacio_vertical // 2)
		p2 = int((cord[0] + cord[2] // 2) - espacio_horizonal // 2)

		digito = imagen_bn[p1:p1 + espacio_vertical, p2:p2 + espacio_horizonal]
		if p2 > 0:
			digito = cv2.resize(digito, dsize=(28, 28), interpolation=cv2.INTER_AREA)
			# aplanamos al candidato
			aux = digito.flatten().reshape(1, -1)
			# pasamos el candidato aplanado por nuestro modelo
			prediccion = r.predecir(aux)
			simbolo, es_una_operacion = obtener_simbolo(prediccion[0])
			# colocamos el digito reconocido en la imagen
			if es_una_operacion:
				lista_de_operadores.append((simbolo, cord))
			else:
				lista_de_numeros.append((simbolo, cord))
	# marcamos los resultados
	num_izq, num_der, operador,resultado = obtener_resultado(lista_de_numeros, lista_de_operadores)
	response = {
		'Num. de la izquierda': f"{num_izq}",
		'Num. de la derecha': f"{num_der}",
		'Operador': f"{operador}",
		'Resultado': '{}'.format(resultado)}
	# armamos json para enviar al cliente
	response_pickled = jsonpickle.encode(response)
	return Response(response=response_pickled, status=200, mimetype="application/json")


app.run()

