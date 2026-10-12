# %%
# PREDICTION FUNCTION
def predict(x, intercept, weight):
    return intercept + weight * x


# %%
# Calculate loss
def calculate_mse(actual, predicted):
    total_error = 0

    for y, y_hat in zip(actual, predicted, strict=False):
        total_error += (y - y_hat) ** 2

    return total_error / len(actual)


# %%
# ------------------------------------------------------------
# 1. TRAINING
# ------------------------------------------------------------

# House size (in units of 10 m²)
sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# Price (in thousands of rand)
prices = [145, 160, 180, 190, 210, 230, 250, 255, 280, 300]

train_sizes = sizes[:8]
train_prices = prices[:8]

test_sizes = sizes[8:]
test_prices = prices[8:]

best_mse = float("inf")
best_intercept = None
best_weight = None

for intercept in range(100, 250, 5):
    for weight in range(5, 20):
        # TODO: Generate predictions using TRAINING sizes only
        predictions = []

        # Your code here

        for size in train_sizes:
            prediction = predict(size, intercept, weight)
            predictions.append(prediction)

        mse_training = calculate_mse(train_prices, predictions)

        if mse_training < best_mse:
            best_mse = mse_training
            best_intercept = intercept
            best_weight = weight


# ------------------------------------------------------------
# 2. EVALUATION
# ------------------------------------------------------------

# TODO: Generate predictions for the TEST sizes
# using best_intercept and best_weight

test_predictions = []

# Your code here

for size in test_sizes:
    prediction = predict(size, best_intercept, best_weight)
    test_predictions.append(prediction)

mse_test = calculate_mse(test_prices, test_predictions)


# ------------------------------------------------------------
# 3. RESULTS
# ------------------------------------------------------------

print("Best intercept:", best_intercept)
print("Best weight:", best_weight)
print("Training MSE:", best_mse)
print("Test MSE:", mse_test)
