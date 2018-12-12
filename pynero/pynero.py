import numpy as np


import tensorflow as tf
import matplotlib.pyplot as plt

def show_test():
    fig1 = plt.figure(1, figsize=(10,10))
    for i in range(0, 9):
        plt.subplot(1,10,i+1)
        m_x_train = plt.imshow(x_train[i])
        plt.xlabel(y_train[i])
    fig12= plt.figure(2, figsize=(10,10))
    for i in range(0, 9):
        plt.subplot(1,10,i+1)
        m_x_train = plt.imshow(x_test[i])
        plt.xlabel(y_test[i])

    #fig2 = plt.figure(2, )
    #im_x_test = plt.imshow(x_test[0])

    plt.show()
    return 1



mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#fig = plt.figure(1, figsize=(8, 8))
show_test()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

show_test()

#создаем нашу нейронную тренировочную сеть
model = tf.keras.models.Sequential()

#Добавляем слои
#Первый слой делаем из наших изображений 28x28
#переводим из думерного массива в вектор длиной 28х28=784 элемента 
model.add(tf.keras.layers.Flatten())
#первый скрытый уровень
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
#
#
show_test()
#
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

print(y_train[0])

show_test()

