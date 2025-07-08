import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def load_dataset(img_height, img_width, batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)
    return dataset

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def get_binary_weights(weights):
    weights = np.ascontiguousarray(weights)
    weights_shape = weights.shape
    int_weights = weights.view(dtype=np.uint32)
    binary_repr = np.unpackbits(int_weights.view(np.uint8))
    return binary_repr, weights_shape

def flip_bits(binary_weights, percentage):
    bit_array = binary_weights.copy()
    total_bits = bit_array.size
    num_bits_to_flip = int(total_bits * percentage)
    indices_to_flip = np.random.choice(total_bits, num_bits_to_flip, replace=False)
    bit_array[indices_to_flip] = 1 - bit_array[indices_to_flip]
    return bit_array

def binary_to_float(binary_weights, original_shape):
    packed_bits = np.packbits(binary_weights)
    int_repr = packed_bits.view(dtype=np.uint32)
    float_repr = int_repr.view(dtype=np.float32)
    return float_repr.reshape(original_shape)

def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return accuracy

def main():
    if len(sys.argv) < 2:
        print("Usage: python automate_layer.py <layer_name>")
        print("Example: python automate_layer.py conv2d_1")
        sys.exit(1)

    layer_name = sys.argv[1]
    model_path = "trained_models/cifar_model.keras"
    img_height, img_width = 32, 32
    batch_size = 32

    dataset = load_dataset(img_height, img_width, batch_size)
    percentages = [i * 0.001 for i in range(0, 11)]  # 0% to 1% in 0.1% steps
    accuracies = []

    for percentage in percentages:
        K.clear_session()
        model = load_model(model_path)

        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")
            sys.exit(1)

        weights = layer.get_weights()
        if not weights:
            print(f"Layer '{layer_name}' has no weights to modify.")
            accuracies.append(None)
            continue

        kernel, *rest = weights
        binary_weights, _ = get_binary_weights(kernel)
        modified_binary_weights = flip_bits(binary_weights, percentage)
        modified_kernel = binary_to_float(modified_binary_weights, kernel.shape)
        modified_weights = [modified_kernel] + rest
        layer.set_weights(modified_weights)

        accuracy = evaluate_model(model, dataset)
        accuracies.append(accuracy)
        print(f"Bit Flip: {percentage*100:.3f}%, Accuracy: {accuracy:.4f}")

    # Plotting
    plt.plot([p * 100 for p in percentages], accuracies, marker='o')
    plt.xlabel("Percentage of Bits Flipped (%)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Bit Flip in Layer '{layer_name}'")
    plt.grid(True)

    os.makedirs("figures", exist_ok=True)
    filename = f"figures/{layer_name}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved to: {filename}")

if __name__ == "__main__":
    main()
