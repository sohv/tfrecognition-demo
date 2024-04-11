import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()

# Splitting training set into training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=5000, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

# store the number of units per layer in global variables
n_input = 784  # input layer (28X28 pixels)
n_hidden1 = 512  # hidden layer 1
n_hidden2 = 256  # hidden layer 2
n_hidden3 = 128  # hidden layer 3
n_output = 10  # output layer (0-9)

# define hyperparameters that remain constant throughout the training process
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5
# we will use dropout in final hidden layer to give each unit 50% chance of being eliminated at every training step.

# Setting up layers of the network
inputs = tf.keras.Input(shape=(n_input,))
layer_1 = tf.keras.layers.Dense(n_hidden1, activation='relu')(inputs)
layer_2 = tf.keras.layers.Dense(n_hidden2, activation='relu')(layer_1)
layer_3 = tf.keras.layers.Dense(n_hidden3, activation='relu')(layer_2)
layer_drop = tf.keras.layers.Dropout(rate=1 - dropout)(layer_3)
output_layer = tf.keras.layers.Dense(n_output, activation='softmax')(layer_drop)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=output_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train.reshape(-1, n_input), y_train,
                    batch_size=batch_size,
                    epochs=n_iterations,
                    validation_data=(X_val.reshape(-1, n_input), y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test.reshape(-1, n_input), y_test)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model.save('model.h5')



