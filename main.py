import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.datasets import fashion_mnist

# Загрузка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Нормализация данных
x_train = x_train / 255.0
x_test = x_test / 255.0

Image = 2

# Создание функции отображения изображения в отдельном окне
plt.figure(figsize=(6, 6))
plt.imshow(x_test[Image].reshape([28, 28]), cmap='gray')

# Создание нейросети
model = (keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),keras.layers.Dense(10, activation='softmax')]))

# Настройка параметров для нейросети
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Обучение Нейросети
model.fit(x_train, y_train, epochs=20)

# Оценка вероятности определения изображения
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nВероятность определения:', test_acc)

# Тестирование обученной нейросети
predictions = model.predict(x_test)
print("ответ нейросети: ", np.argmax(predictions[Image]))
print("правильный ответ: ", y_test[Image])

plt.show()