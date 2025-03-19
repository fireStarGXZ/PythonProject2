from keras.src.datasets import mnist
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten
from keras.src.optimizers import SGD
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical

import plot_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("X_train.shape:"+str(X_train.shape))
print("Y_train.shape:"+str(Y_train.shape))
print("X_test.shape:"+str(X_test.shape))
print("Y_test.shape:"+str(Y_test.shape))

print(Y_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()

X_train = X_train.reshape(60000,28,28,1)/255.0
X_test = X_test.reshape(10000,28,28,1)/255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()
model.add(Input(shape=(28,28,1)))
#filters代表卷积核数量,kernel_size代表卷积核尺寸,strides代表步长
model.add(Conv2D(6, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=4096)
loss, accuracy = model.evaluate(X_test, Y_test)
print("loss:", loss)
print("accuracy:", accuracy)