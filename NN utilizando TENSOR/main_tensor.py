import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from numpy.random import seed


arch = h5py.File(r"C:\Users\Dussan\Desktop\digitos_con_signos.h5", "r")
X_train = arch["X_train"][:]
y_train = arch["y_train"][:]
X_test = arch["X_test"][:]
y_test = arch["y_test"][:]

seed(4)

modelo = Sequential([
    InputLayer(input_shape=(784, 1)),
    Reshape(target_shape=(28, 28, 1)),
    Dense(25, activation='sigmoid'),
    Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
    # max pooling es para usar softmax
    MaxPooling2D(pool_size=(2, 2)),
    # dropout es una capa extra que tiene una funcion como lambda
    Dropout(0.15),
    Flatten(),
    Dense(12)
])
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,),
    metrics=[
        "accuracy"
    ]
)

num_epochs = 20
modelo.fit(X_train, y_train, epochs=num_epochs)
modelo.save("thetas_tensor_epocas_20.h5")

