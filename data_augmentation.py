import os
import random
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io


def rotar(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def ruido(image_array):
    return sk.util.random_noise(image_array)


def cambio_horizontal(image_array):
    return image_array[:, ::-1]


transformaciones = {
    'rotate': rotar,
    'noise': ruido,
    'horizontal_flip': cambio_horizontal
}

folder_path = 'Operadores/Operadores en blanco y negro/signos positivos en blanco y negro/'
folder_destiny_path = 'Operadores/Operadores Utilizables/signos positivos utilizables/'
num_trans = 750

imagenes = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_trans:
    image_path = random.choice(imagenes)
    image_to_transform = sk.io.imread(image_path)
    num_transformations_to_apply = random.randint(1, len(transformaciones))
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        key = random.choice(list(transformaciones))
        transformed_image = transformaciones[key](image_to_transform)
        num_transformations += 1
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_destiny_path, num_generated_files)
        io.imsave(new_file_path, transformed_image)
    num_generated_files += 1