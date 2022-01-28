import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data( )

# Normalize the Arrays
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential( )  # 28*28
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                                 input_shape=(28, 28, 1)))  # 28*28*32

model.add(tf.keras.layers.Dropout(0.5))  # 28*28*32

model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # 14*14*32

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu))  # 14*14*64
model.add(tf.keras.layers.Dropout(0.5))  # 14*14*64
model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # 7*7*64
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu))  # 7*7*64
model.add(tf.keras.layers.Dropout(0.5))  # 7*7*64
model.add(tf.keras.layers.Flatten( ))  # 3136
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))  # 64
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 10

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

_ = model.fit(X_train, y_train, epochs=30, batch_size=512)
