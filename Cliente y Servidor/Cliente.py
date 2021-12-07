import requests
import json
import cv2

# direccion a la que realizamos la solicitud
direccion = 'http://localhost:5000/'
# la ruta a la enviamos la solicitud
ruta = direccion + "/api/test"

# indicamos tipo de archivo a enviar
tipo_contenido = 'imagen/jpeg'
# indicamos la cabecera de la solicitud
cabezera = {'content-type': tipo_contenido}

# leemos la imagen que queremos enviar
imagen = cv2.imread('../imagenes_de_prueba/nivel_2.jpg')
# codificamos la imagen
_, imagen_codificada = cv2.imencode('.jpg', imagen)
# armamos solicitud, ruta solicitud, la imagen codificada, cabecera
response = requests.post(ruta, data=imagen_codificada.tobytes(), headers=cabezera)
# imprimimos la respuesta para ver el resultado
print(json.loads(response.text))

