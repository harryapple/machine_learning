# -*- coding: utf-8 -*-
from keras.datasets import imdb

(train_data, train_labes), (test_data, test_labels) = imdb.load_data(num_words=10000)

#采用one-hot方法将每条评论放在1万个热词的数组中，后续考虑用embedding的方式替换
#np.zeros形成个ndarray的二维数组
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

#数据集和类别集都要转化相应的类型
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labes).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#数据拆分出来验证集
x_train_train = x_train[:10000]
x_train_val = x_train[10000:]

y_train_train = y_train[:10000]
y_train_val = y_train[10000:]

#3-3 神经网络模型
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
'''
#3-4 编译和
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
'''
#验证数据准确度
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train_val,
    y_train_val,
    epochs = 20,
    batch_size=512,
    validation_data = (x_train_train, y_train_train)
)

print(history.history)

#数据可视化
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#精度
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.title('Trainning and validation accuracy')

plt.xlabel('Epochs')
plt.ylabel('ACC')

plt.legend()
plt.show()