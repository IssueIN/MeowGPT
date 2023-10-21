import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

x_train= np.load('x_train.npy')
x_test= np.load('x_test.npy')
y_train= np.load('y_train.npy')
y_test= np.load('y_test.npy')

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(20, 173, 1)),
  #  tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for emotions
])

#compile

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#fit
model.fit(x_train, y_train, epochs=20, validation_split=0.1)

#evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')