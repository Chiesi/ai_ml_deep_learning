# Remember to run the following commands!
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install jupyterlab
# pip install ipywidgets
# jupyter labextension enable widgetsnbextension
# pip install kagglehub

# pip install joblib
# pip install tensorflow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# Load the MNIST fashion data - we're italians, after all!
train, test = tf.keras.datasets.fashion_mnist.load_data()

# Assign the variables to the dataset values, and batch them in buckets of 32
images, labels = train
labels = labels.astype(np.int32)
images = images/256
train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
train_ds = train_ds.batch(32)

# Inspectint a data sample is usually a good call
print ("label:" + str(labels[0]))
pixels = images[0]
plt.imshow(pixels, cmap='gray')
plt.show()

# Let's build our first deep learning network!
# It'll be a multi-layer perceptron with 2 hidden RELU layers (the first with 100
# neurons, the second with 50) and an output layer with 10 neurons
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# We'll optimize with Adam, a variant of gradient descent - the only difference
# being that it uses cross-entropy
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.
    SparseCategoricalAccuracy()]
)
model.fit(train_ds, epochs=10)

# Validate the model performance by comparing against the test dataset
images_test, labels_test = test
labels_test = labels_test.astype(np.int32)
images_test = images_test/256

test_ds = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
test_ds = train_ds.batch(32)
# We'll also shuffle the dataset to ensure ordering is not a factor
test_ds = train_ds.shuffle(30)
results = model.evaluate(test_ds)
print("Test loss and accuracy:", results)

# Bonus: 101 ways to lie with statistics!
predictions = model.predict(test[0])
predicted_labels = np.argmax(predictions, axis=1)
m = tf.keras.metrics.Accuracy()
m.update_state(predicted_labels, test[1])
m.result().numpy()

# Save the trained model
model.save("first_model.keras")
