import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Problem 9: Research into Famous Image Recognition Models
# - AlexNet (2012): Introduced deep convolutional networks for image classification and won the ImageNet challenge.
# - VGG16 (2014): A deep CNN architecture with 16 layers, known for its simplicity and effectiveness.
# - ResNet (2015): Introduced residual learning to solve the vanishing gradient problem, allowing very deep networks.
# - Inception (GoogLeNet, 2014): Used multi-scale convolutional filters within a single layer to improve efficiency.
# - EfficientNet (2019): Optimized network width, depth, and resolution scaling to achieve better performance with fewer parameters.

# Problem 11: Investigation into Filter Size
# - Why 3x3 Filters?
#   - More efficient than larger filters while still capturing spatial features.
#   - Stacking multiple 3x3 layers provides the same receptive field as larger filters but with fewer parameters.
# - Effect of 1x1 Filters:
#   - Used for dimensionality reduction without affecting spatial resolution.
#   - Helps in feature recombination, allowing better information flow in networks like GoogLeNet.

# Problem 10: Calculating Output Size and Number of Parameters
def calculate_output_size(input_size, filter_size, stride, padding=0):
    return (input_size + 2 * padding - filter_size) // stride + 1

def calculate_parameters(in_channels, out_channels, filter_size, bias=True):
    params = in_channels * out_channels * filter_size * filter_size
    if bias:
        params += out_channels
    return params

# Problem 10: Given three convolutional layers
conv_layers = [
    (144, 3, 3, 6, 1, 0),  # Input size: 144x144, 3 channels, 3x3 filter, 6 output channels, stride 1, no padding
    (60, 24, 3, 48, 1, 0),  # Input size: 60x60, 24 channels, 3x3 filter, 48 output channels, stride 1, no padding
    (20, 10, 3, 20, 2, 0)   # Input size: 20x20, 10 channels, 3x3 filter, 20 output channels, stride 2, no padding
]

print("Problem 10: Calculating Output Size and Number of Parameters")
for i, (input_size, in_channels, filter_size, out_channels, stride, padding) in enumerate(conv_layers, 1):
    output_size = calculate_output_size(input_size, filter_size, stride, padding)
    num_params = calculate_parameters(in_channels, out_channels, filter_size)
    print(f"Layer {i}: Output Size = {output_size}x{output_size}, Parameters = {num_params}")

# Problem 1: Implementing a Convolutional Layer
print("\nProblem 1: Implementing a Convolutional Layer")

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.01, padding=0, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.padding = padding
        self.stride = stride
        Fh, Fw = kernel_size
        scale = np.sqrt(2.0 / (in_channels * Fh * Fw))  # He Initialization
        self.weights = np.random.randn(out_channels, in_channels, Fh, Fw) * scale
        self.biases = np.zeros(out_channels)

    def pad_input(self, X):
        if self.padding > 0:
            return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        return X

    def compute_output_size(self, N_in, P, F, S):
        return (N_in + 2 * P - F) // S + 1

    def forward(self, X):
        self.X = self.pad_input(X)
        N, C, H, W = self.X.shape
        M, _, Fh, Fw = self.weights.shape
        
        H_out = self.compute_output_size(H, self.padding, Fh, self.stride)
        W_out = self.compute_output_size(W, self.padding, Fw, self.stride)
        
        out = np.zeros((N, M, H_out, W_out))
        for n in range(N):
            for m in range(M):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        
                        # Ensure slice boundaries are within input dimensions:
                        h_end = min(h_start + Fh, H)
                        w_end = min(w_start + Fw, W)
                        
                        # Slice input and filter to have the same shape:
                        X_slice = self.X[n, :, h_start:h_end, w_start:w_end]
                        weights_slice = self.weights[m, :, :X_slice.shape[1], :X_slice.shape[2]]
                        
                        out[n, m, i, j] = np.sum(X_slice * weights_slice) + self.biases[m]
                        
        return np.maximum(0, out)  # ReLU Activation

    def calculate_parameters(self):
        Fh, Fw = self.kernel_size
        return (self.in_channels * Fh * Fw * self.out_channels) + self.out_channels

# Problem 2: Test with a Simple CNN
print("\nProblem 2: Test with a Simple CNN")

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_sample = x_test[0]
plt.imshow(x_sample, cmap='gray')
plt.title(f"Sample Image - Label: {y_test[0]}")
plt.show()

# Preprocess image
x_sample = x_sample.astype(np.float32) / 255.0
x_sample = np.expand_dims(x_sample, axis=(0, 1))

# Initialize CNN and test the image
cnn = Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=0)
output = cnn.forward(x_sample)
print(f"Output shape after Conv Layer: {output.shape}")

# Problem 3: Explanation of the CNN Layer Implementation
print("\nProblem 3: Explanation of the CNN Layer Implementation")
print("The Conv2d class implements a basic 2D convolutional layer. It includes:")
print("- He initialization for weights.")
print("- Padding option to handle edge cases.")
print("- A forward pass that applies the convolution with ReLU activation.")

# Problem 4: Testing CNN Layer Output Visualization
print("\nProblem 4: Testing CNN Layer Output Visualization")

plt.imshow(output[0][0], cmap='gray')
plt.title("Processed Image After CNN Layer")
plt.show()

# Problem 5: Test Model with Multiple Layers
print("\nProblem 5: Test Model with Multiple Layers")

# Define a simple CNN with multiple layers
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=0)
        self.conv2 = Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.conv2.forward(X)
        return X

# Initialize and test the CNN with two layers
cnn = SimpleCNN()
output = cnn.forward(x_sample)
print(f"Output shape after two Conv layers: {output.shape}")

# Problem 6: Testing CNN with Larger Input Images
print("\nProblem 6: Testing CNN with Larger Input Images")

# Process a larger image (28x28)
x_large_sample = x_test[1]
plt.imshow(x_large_sample, cmap='gray')
plt.title(f"Larger Sample Image - Label: {y_test[1]}")
plt.show()

x_large_sample = x_large_sample.astype(np.float32) / 255.0
x_large_sample = np.expand_dims(x_large_sample, axis=(0, 1))

output = cnn.forward(x_large_sample)
print(f"Output shape after Conv layers on larger image: {output.shape}")

# Problem 7: Visualization of Multiple Layer Outputs
print("\nProblem 7: Visualization of Multiple Layer Outputs")

plt.imshow(output[0][0], cmap='gray')
plt.title("Processed Image After Multiple Conv Layers")
plt.show()

# Problem 8: Investigating the Effect of Stride and Padding
print("\nProblem 8: Investigating the Effect of Stride and Padding")

cnn = Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=2, padding=1)
output = cnn.forward(x_sample)
print(f"Output shape with stride 2 and padding 1: {output.shape}")

# Problem 10: Final Output of Convolution Layers
print("\nOutput Size and Parameters for Convolutional Layers:")
for i, (input_size, in_channels, filter_size, out_channels, stride, padding) in enumerate(conv_layers, 1):
    output_size = calculate_output_size(input_size, filter_size, stride, padding)
    num_params = calculate_parameters(in_channels, out_channels, filter_size)
    print(f"Layer {i}: Output Size = {output_size}x{output_size}, Parameters = {num_params}")

# Problem 11: Full Functionality Test
print("\nProblem 11: Full Functionality Test")

# Complete CNN forward pass on the sample image
cnn_test = SimpleCNN()
final_output = cnn_test.forward(x_sample)
print(f"Final Output shape after two convolutional layers: {final_output.shape}")
