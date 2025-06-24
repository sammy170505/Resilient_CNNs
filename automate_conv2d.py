import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import get_file
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def load_dataset(dummy, img_height, img_width, batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)
    return dataset

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        #model.summary()
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

    # for idx in indices_to_flip:
    #     bit_array[idx] = 0 if bit_array[idx] == 1 else 1
    bit_array[indices_to_flip] = 1 - bit_array[indices_to_flip]

    #modified_binary_weights = [''.join(bit_array[i:i+32]) for i in range(0, total_bits, 32)]
    return bit_array

def binary_to_float(binary_weights, original_shape):
    packed_bits = np.packbits(binary_weights)
    int_repr = packed_bits.view(dtype=np.uint32)
    float_repr = int_repr.view(dtype=np.float32)
    return float_repr.reshape(original_shape)

def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset, verbose=2)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    return accuracy

def main():
    model_path = "/Users/courtneymoane/Desktop/research_project/trained_models/cifar_model.keras"  
    
    img_height = 32
    img_width = 32
    batch_size = 32
    dataset = load_dataset(None, img_height, img_width, batch_size)

    percentages = [i * 0.001 for i in range(0, 11)]
    accuracies = []

    #load_model(model_path)

    for percentage in percentages:

        # New potential plan to remove layer to change, manipulate the layed, build back model to test
        # and reset the model to the original state at each iteration

        # save_weights()? it can overwrite existing weights of an existing model

        K.clear_session()
        model = load_model(model_path)
        if model is None:
            return

        layer = model.get_layer('conv2d')
        kernel, bias = layer.get_weights()

        #print(f"Original kernel: {kernel}")

        binary_weights, original_shape = get_binary_weights(kernel)


        modified_binary_weights = flip_bits(binary_weights, percentage)
        modified_kernel = binary_to_float(modified_binary_weights, kernel.shape)
        #print(f"Modified kernel: {modified_kernel}")

        layer.set_weights([modified_kernel, bias])
        #model.set_layer(layer)

        #layer2 = model.get_layer('conv2d')
        #print(layer)
        #print("\n\n")
        #print(layer2)

        accuracy = evaluate_model(model, dataset)
        #percentages.append(percentage)
        accuracies.append(accuracy)
        print(f"Bit Flip: {percentage}%, Accuracy: {accuracy}")


    plt.plot(percentages, accuracies, marker='o')
    plt.xlabel("Percentage of bits flipped")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs. Bit Flip Percentage for Layer Conv2d")
    plt.grid(True)
    plt.savefig("figures/conv2d 0-0.1.png")
    plt.show()

if __name__ == "__main__":

    main()