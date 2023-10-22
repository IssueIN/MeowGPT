import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import numpy as np

# Load your data
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define the number of folds
n_splits = 5  # You can change this number as needed

# Create a KFold object
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
fold_accuracies = []

# Iterate over each fold
for train_index, val_index in kf.split(x_train):
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(112, (3, 3), activation='relu', input_shape=(20, 173, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(
                units=(96), 
                activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))   
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train_fold, y_train_fold, epochs=10, validation_data=(x_val_fold, y_val_fold))

    # Evaluate the model on the test set
    _, fold_accuracy = model.evaluate(x_test, y_test)
    fold_accuracies.append(fold_accuracy)

# Calculate and print the mean accuracy over all folds
mean_accuracy = np.mean(fold_accuracies)
max_accuracy = np.max(fold_accuracies)
print(f'Mean Test accuracy: {mean_accuracy}, Mean Test accuracy: {max_accuracy}')

# Save the model
model.save('meowmodel.h5')

