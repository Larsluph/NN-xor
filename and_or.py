import numpy as np

# Define I/O
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
labels = np.array([[0, 0],
                   [0, 1],
                   [0, 1],
                   [1, 1]])

# Define activation methods
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define hyperparameters
epochs = 100_000
np.random.seed(42)
learning_rate = .1

input_size = 2
output_size = 2

# Define weights
weights = np.random.uniform(size=(input_size, output_size))
bias = np.random.uniform(size=(1, output_size))

# Train the model
y_hat = None
for epoch in range(epochs):
    # Forward propagation
    y_hat = sigmoid(np.dot(inputs, weights) + bias)

    # Calculate loss
    loss = labels - y_hat

    # Backpropagation
    y_error = loss * sigmoid_derivative(y_hat)

    # Update weights and biases
    weights += inputs.T.dot(y_error) * learning_rate
    bias += np.sum(y_error) * learning_rate

    # Print progress
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss: {np.mean(np.abs(loss))}')

# Print final prediction
print('Predictions:')
print(y_hat)
