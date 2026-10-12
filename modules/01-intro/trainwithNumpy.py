import numpy as np

X = np.array([[5, 2, 10], [8, 3, 5], [10, 4, 2], [6, 2, 8], [9, 3, 4]])

y = np.array([160, 231, 281, 181, 250])

# Add a column of ones for the intercept
ones = np.ones((X.shape[0], 1))
X_with_intercept = np.column_stack((ones, X))

# Find least-squares coefficients
weights, residuals, rank, singular_values = np.linalg.lstsq(X_with_intercept, y, rcond=None)

# Generate predictions
predictions = X_with_intercept @ weights

# Calculate training MSE
mse = np.mean((y - predictions) ** 2)

print("Weights:", weights)
print("Residuals:", residuals)
print("Rank:", rank)
print("Singular_values:", singular_values)
print("Predictions:", predictions)
print("Training MSE:", mse)
