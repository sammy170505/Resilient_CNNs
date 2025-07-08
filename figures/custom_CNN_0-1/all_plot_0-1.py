import matplotlib.pyplot as plt

# Bit flip percentages
percentages = [i * 0.1 for i in range(11)]

# Accuracy results for each layer (from your message)
accuracies_conv2d = [0.7306, 0.7138, 0.1020, 0.1198, 0.6377, 0.1430, 0.5840, 0.1064, 0.1068, 0.1107, 0.1030]
accuracies_conv2d_1 = [0.7306, 0.0965, 0.1222, 0.1145, 0.1041, 0.1428, 0.0907, 0.1402, 0.1884, 0.0905, 0.1342]
accuracies_conv2d_2 = [0.7306, 0.1056, 0.0994, 0.1065, 0.1359, 0.1229, 0.1245, 0.1074, 0.1430, 0.1278, 0.1216]
accuracies_dense = [0.7306, 0.1121, 0.1341, 0.1340, 0.1053, 0.1449, 0.1128, 0.1417, 0.1692, 0.1821, 0.1511]
accuracies_dense_1 = [0.7306, 0.7210, 0.6860, 0.3471, 0.3183, 0.2889, 0.6496, 0.1737, 0.3790, 0.2428, 0.6823]

# Plot setup
plt.figure(figsize=(10, 6))

plt.plot(percentages, accuracies_conv2d, marker='o', label='conv2d')
plt.plot(percentages, accuracies_conv2d_1, marker='o', label='conv2d_1')
plt.plot(percentages, accuracies_conv2d_2, marker='o', label='conv2d_2')
plt.plot(percentages, accuracies_dense, marker='o', label='dense')
plt.plot(percentages, accuracies_dense_1, marker='o', label='dense_1')

plt.xlabel("Percentage of bits flipped")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Bit Flip Percentage for Different Layers")
plt.legend()
plt.grid(True)

# Save to the same folder as before
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/custom_CNN_0-1/all_layers_comparison.png")
plt.show()
