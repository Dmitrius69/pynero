import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

def show_test():
    fig1 = plt.figure(1, )
    im_x_train = plt.imshow(x_train[0])

    fig2 = plt.figure(2, )
    im_x_test = plt.imshow(x_test[0])

    plt.show()
    return 1



mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#fig = plt.figure(1, figsize=(8, 8))
show_test()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

show_test()
