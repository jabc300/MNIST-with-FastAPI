import numpy as np
import tensorflow as tf
import matplotlib_inline as plt
from PIL import Image

def get_model():
    model = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(input_shape=(28,28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
    )
    model.compile(optimizer="Adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.load_weights("model_weights/cp.ckpt")
    return model

def get_image(image_file : np.array):
    img = image_file.convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    return im2arr

def predict(numbers_array : np.ndarray):
    new_model = get_model
    predict = new_model.predict(numbers_array)

    print(predict)
