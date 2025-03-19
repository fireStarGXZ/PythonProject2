import dataset
import plot_utils

import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD
#从数据中获取随机豆豆
m=100
X,Y = dataset.get_beans(m)
plot_utils.show_scatter(X, Y)

model = Sequential()
model.add(Dense(1, 'sigmoid', input_dim=2))
#model.add(Dense(1, 'sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X, Y, epochs=1000, batch_size=10)
pres = model.predict(X)

plot_utils.show_scatter_surface(X, Y, model)











