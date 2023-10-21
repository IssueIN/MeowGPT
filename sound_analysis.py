import tensorflow as tf
tf.__version__

mnist=tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize data, not necessary but makes it easier for the model to learn
x_train = tf.keras.utils.normalize(x_train, axis=1) #scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1) #scales data between 0 and 1

#build the model (feed-forward model)  #sequential neural network
model=tf.keras.models.Sequential()#this is a feed-forward model
model.add(tf.keras.layers.Flatten())#input layer, flattens the data in order to feed it into the neural network
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#hidden layer, 128 neurons, rectified linear unit activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#hidden layer, 128 neurons, rectified linear unit activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))#output layer, 10 neurons, softmax activation function

#parameters for training the model
model.compile(optimizer='adam',#optimizer function, how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy',#loss function, degree of error
              metrics=['accuracy'])#what to track, accuracy is the most common metric

#train the model
model.fit(x_train, y_train, epochs=3) #epochs is the number of times the model sees the data