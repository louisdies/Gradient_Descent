import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def gradient_descent(W, b, X, y, m, alpha, iterations):
    W = np.zeros((n,1))
    for i in range(iterations + 1):
        predictions = X @ W + b
        dW = (1 / m) * X.T @ (predictions - y)
        db = (1 / m) * np.sum(predictions - y)

        W -= alpha * dW
        b -= alpha * db
        cost = cost_function(W, b, X, y, m)

        if i % 1000 == 0:
            print(f'Iteration: {i}, Cost: {cost}')

    return W, b


def predicted_value(X, W, b):
    expected_value = X @ W + b
    return expected_value

def cost_function(W, b, X, y, m):
    predictions = X @ W + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Fetch the data
california_housing = fetch_california_housing()
#description = california_housing.DESCR
#print(description)

# Assign data to variables
X = california_housing.data
y = california_housing.target
y = y.reshape(-1, 1)

# Normalize the features
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X = (X - mean_X) / std_X

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obtain matrix dimensions
m, n = X_train.shape

# Initialize weights and bias
W = np.zeros((n, 1))
b = 0

cost = cost_function(W, b, X_train, y_train, m)
print(f'Initial cost: {cost}')


alpha = [1, 0.01, 0.1, 0.5]  # Reduced learning rate

iterations = 5000

for i in alpha:
    W_updated, b_updated = gradient_descent(W, b, X_train, y_train, m, i, iterations)
    print(f'Updated weights: \n{W_updated} for alpha equals {i}')
    print(f'Updated bias: {b_updated} for alpha equals {i}')


# Calculate the predicted values
prediction = predicted_value(X_test, W_updated, b_updated)

# Denormalize the predictions
prediction_denorm = prediction * std_X[-1] + mean_X[-1]

# Denormalize the first feature of X_test
X_test_denorm = X_test[:, -1] * std_X[-1] + mean_X[-1]

# Denormalize y_test
y_test_denorm = y_test * std_X[-1] + mean_X[-1]

# Plot actual values
plt.scatter(X_test_denorm, y_test_denorm, alpha=0.5, label='Actual')

# Plot predicted values
plt.scatter(X_test_denorm, prediction_denorm, alpha=0.5, label='Predicted')

plt.xlabel('Feature -1')
plt.ylabel('Price')
plt.legend()
plt.show()