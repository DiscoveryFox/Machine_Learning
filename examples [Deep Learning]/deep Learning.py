import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data( )

# Normalize the Arrays
X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential( )
model.add(tf.keras.layers.Flatten( ))
model.add(tf.keras.layers.Dense(1500, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # steps_per_execution=1
_ = model.fit(X_train, y_train, epochs=500, use_multiprocessing=True)  # steps_per_epoch=10000

# <editor-fold desc="print Number of Train/Test Images">
print('----------------')
print('Train Images: ', len(X_train))
print('Test Images: ', len(X_test))
print('----------------')
print( )
# </editor-fold>

print('Finding final training accuracy')
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print( )
print('Finding final test accuracy')
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print( )

print('----------------')
print('Accuracy Training: ', str(np.round(train_accuracy * 100, 2)), '%')
print( )
print('Accuracy Test: ', str(np.round(test_accuracy * 100, 2)), '%')
print('----------------')

# Predict Method

predictvalue = None
model.predict(predictvalue)
