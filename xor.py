import numpy as np

# Define I/O
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
labels = np.array([[0],
                   [1],
                   [1],
                   [0]])

# Define activation methods
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define hyperparameters
epochs = 10_000
np.random.seed(42)
learning_rate = 0.1

input_size = 2
hidden_size = 2
output_size = 1

# Define weights and biases
wts_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_hidden = np.random.uniform(size=(1, hidden_size))

wts_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_output = np.random.uniform(size=(1, output_size))

# Training the model
y_hat = None
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(inputs, wts_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, wts_hidden_output) + bias_output
    y_hat = sigmoid(output_layer_input)

    # Calculate loss (Mean Squared Error)
    loss = labels - y_hat
    mse = np.mean(np.square(loss))

    # Backpropagation
    error_output_layer = loss * sigmoid_derivative(y_hat)
    error_hidden_layer = error_output_layer.dot(wts_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    wts_hidden_output += hidden_layer_output.T.dot(error_output_layer) * learning_rate
    bias_output += np.sum(error_output_layer, axis=0, keepdims=True) * learning_rate
    wts_input_hidden += inputs.T.dot(error_hidden_layer) * learning_rate
    bias_hidden += np.sum(error_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print progress
    if epoch % 1000 == 0:
        print(f'{epoch = }, {mse = }')

# Print final prediction
print("Final predicted output:")
print(y_hat)
