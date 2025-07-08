import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Parameters
epochs = 10
batch_size = 32
input_shape = (32, 32, 3)
num_classes = 10

# Directories
os.makedirs('trained_models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess: convert to float and apply MobileNetV2 preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Dataset pipeline
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # Create dataset from tensors
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)) # Apply MobileNetV2 preprocessing
train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build model: MobileNetV2 base + classification head
base_model = MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
# Allow fine-tuning of all layers
base_model.trainable = True

model = Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('MobileNetV2 Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('MobileNetV2 Training and Validation Loss')
plt.savefig('figures/mobilenetv2_cifar10_history.png')

# Save trained model
model_path = 'trained_models/mobilenetv2_cifar10.h5'
print(f"Saving model to '{model_path}'...")
model.save(model_path)

# Confirm completion
print('MobileNetV2 training and saving complete.')
