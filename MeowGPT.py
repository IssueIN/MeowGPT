import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#create x_train, y_train, x_test, y_test based on Mfccs_b, Mfccs_i, Mfccs_f
# x_train = np.array(all_mfccs_b)
# y_train = np.array(all_mfccs_i)
# x_test = np.array(all_mfccs_f)
# y_test = np.array(all_mfccs_f)
#randomly select 80% of the data for training and 20% for testing
x_train= np.load('x_train.npy')
x_test= np.load('x_test.npy')
y_train= np.load('y_train.npy')
y_test= np.load('y_test.npy')



#normalize data, not necessary but makes it easier for the model to learn
# x_train = tf.keras.utils.normalize(x_train, axis=1) #scales data between 0 and 1
# x_test = tf.keras.utils.normalize(x_test, axis=1) #scales data between 0 and 1

#build the model (feed-forward model)  #sequential neural network
model=tf.keras.models.Sequential()#this is a feed-forward model
model.add(tf.keras.layers.Flatten())#input layer, flattens the data in order to feed it into the neural network
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#hidden layer, 128 neurons, rectified linear unit activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#hidden layer, 128 neurons, rectified linear unit activation function
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))#output layer, 3 neurons, softmax activation function

#parameters for training the model
model.compile(optimizer='adam',#optimizer function, how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy',#loss function, degree of error
              metrics=['accuracy'])#what to track, accuracy is the most common metric

#train the model
model.fit(x_train, y_train, epochs=3) #epochs is the number of times the model sees the data
#save the model
model.save('epic_num_reader.model')

#calculate validation loss and validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print("these are the loss and the accuracy: ",val_loss, val_acc)

#make predictions