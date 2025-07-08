import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CONFIG
BIT_FLIP_RANGE = [i * 0.001 for i in range(11)]
OUTPUT_DIR = "figures/mobilenetv2_0-1"
LOG_DIR = "figures/mobilenetv2_logs"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, 10)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Load MobileNetV2 with CIFAR-10 compatible input shape
model = MobileNetV2(weights=None, classes=10, input_shape=(32, 32, 3))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights("trained_models/mobilenetv2_cifar10.h5")  # Ensure this is trained/saved

# Helper functions
def get_binary_weights(weights):
    weights = np.ascontiguousarray(weights)
    original_shape = weights.shape
    binary_repr = np.unpackbits(weights.view(np.uint8))
    return binary_repr, original_shape

def binary_to_float(binary_weights, original_shape):
    packed_bits = np.packbits(binary_weights)
    float_array = packed_bits.view(dtype=np.uint8).view(dtype=np.float32)
    return float_array.reshape(original_shape)

def flip_bits(binary_weights, percentage):
    bit_array = binary_weights.copy()
    total_bits = bit_array.size
    num_bits_to_flip = int(total_bits * percentage)
    indices_to_flip = np.random.choice(total_bits, num_bits_to_flip, replace=False)
    bit_array[indices_to_flip] = 1 - bit_array[indices_to_flip]
    return bit_array

def evaluate_model(model, dataset):
    loss, acc = model.evaluate(dataset, verbose=0)
    return acc

# Start flipping
layer_accuracies = {}

for layer in model.layers:
    if not layer.trainable or not hasattr(layer, 'weights') or len(layer.get_weights()) == 0:
        continue

    layer_name = layer.name
    try:
        kernel, *rest = layer.get_weights()
    except Exception as e:
        continue

    original_weights = layer.get_weights()
    binary_kernel, original_shape = get_binary_weights(kernel)
    accuracies = []

    print(f"\nLayer: {layer_name}")
    for pct in BIT_FLIP_RANGE:
        tf.keras.backend.clear_session()

        model_copy = MobileNetV2(weights=None, classes=10, input_shape=(32, 32, 3))
        model_copy.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model_copy.load_weights("trained_models/mobilenetv2_cifar10.h5")

        modified_bits = flip_bits(binary_kernel, pct)
        modified_kernel = binary_to_float(modified_bits, original_shape)
        new_weights = [modified_kernel] + rest

        model_copy.get_layer(layer_name).set_weights(new_weights)
        acc = evaluate_model(model_copy, test_dataset)

        print(f"Bit Flip: {pct*100:.1f}%, Accuracy: {acc:.4f}")
        accuracies.append(acc)

    # Save results
    layer_accuracies[layer_name] = accuracies
    out_path = os.path.join(OUTPUT_DIR, f"{layer_name}.png")
    log_path = os.path.join(LOG_DIR, f"{layer_name}.json")

    plt.figure()
    plt.plot(BIT_FLIP_RANGE, accuracies, marker='o')
    plt.title(f"Accuracy vs Bit Flip for {layer_name}")
    plt.xlabel("Bit Flip Percentage")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

    with open(log_path, 'w') as f:
        json.dump({"bit_flip_percentages": BIT_FLIP_RANGE, "accuracies": accuracies}, f)

# Combined Plot
plt.figure(figsize=(12, 8))
for layer, acc in layer_accuracies.items():
    plt.plot(BIT_FLIP_RANGE, acc, label=layer)
plt.xlabel("Bit Flip Percentage")
plt.ylabel("Accuracy")
plt.title("Layer-wise Bit Flip Sensitivity (MobileNetV2)")
plt.legend(loc="upper right", fontsize='small')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "all_layers_combined.png"))
# plt.show()
