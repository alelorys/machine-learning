# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:07:37 2020

@author: Ola
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

 

data = pd.read_csv("data.csv")
#print(data.info())
#print(data.head())
#print(data.describe())

 

data.drop(["id","Unnamed: 32"], axis=1, inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

shapeX = x_data.shape
print(shapeX)
# train test split
from sklearn.model_selection import train_test_split

 

X_data, y_data, X_labels, y_labels = train_test_split(x_data,y,test_size = 0.3,random_state=1)

 

X_data = X_data.to_numpy()
y_data = y_data.to_numpy()


mean = X_data.mean(axis=0)
X_data -= mean
std = X_data.std(axis=0)
X_data /= std

y_data -= mean
y_data /= std

X_labels = np.asarray(X_labels).astype('float32')
y_labels = np.asarray(y_labels).astype('float32')
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, 
                       activation='relu', 
                       input_shape=(X_data.shape[1],)))
model.add(layers.Dense(16, 
                       activation='relu'))
model.add(layers.Dense(1, 
                       activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = X_data[:100]
partial_x_train = X_data[100:]

y_val= X_labels[:100]
partial_y_train = X_labels[100:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=8,
                    validation_data=(x_val, y_val))

model.evaluate(y_data, y_labels)

history_dict = history.history

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Parametr bo definiuje linię przerywaną w postaci niebieskich kropek.
plt.plot(epochs, loss, 'bo', label='Strata trenowania')
# Parametr b definiuje ciągłą niebieską linię.
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()

plt.clf()   # Czyszczenie rysunku.
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(X_data.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_data, X_labels, 
          epochs=2, 
          batch_size=512)
results = model.evaluate(y_data, y_labels)
print(results)