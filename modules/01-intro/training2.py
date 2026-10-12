def float_range(start, end, step):
    """A helper function to allow loops with float step values."""
    current = start
    while current < end:
        yield current
        current += step


def predict(size, intercept, weight):
    """Calculates y = mx + b (price = weight * size + intercept)"""
    return (weight * size) + intercept


def calculate_mse(actual, predicted):
    """Calculates Mean Squared Error between actual values and predictions."""
    total_error = 0
    n = len(actual)
    for i in range(n):
        total_error += (actual[i] - predicted[i]) ** 2
    return total_error / n


def evaluate_model(sizes, prices, intercept, weight):
    predictions = []
    for size in sizes:
        prediction = predict(size, intercept, weight)
        predictions.append(prediction)

    mse = calculate_mse(prices, predictions)

    return mse


def train_model(train_sizes, train_prices, intercept_range, weight_range):
    best_mse = float("inf")
    best_intercept = None
    best_weight = None

    for intercept in intercept_range:
        for weight in weight_range:
            predictions = []

            for size in train_sizes:
                prediction = predict(size, intercept, weight)
                predictions.append(prediction)

            mse_training = calculate_mse(train_prices, predictions)

            if mse_training < best_mse:
                best_mse = mse_training
                best_intercept = intercept
                best_weight = weight

    return (best_intercept, best_weight, best_mse)


sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
prices = [145, 160, 180, 190, 210, 230, 250, 255, 280, 300]

# Training data
train_sizes = sizes[:6]
train_prices = prices[:6]

# Validation data
val_sizes = sizes[6:8]
val_prices = prices[6:8]

# Test data
test_sizes = sizes[8:]
test_prices = prices[8:]


model_a = train_model(train_sizes, train_prices, range(100, 250, 5), range(5, 20))

model_b = train_model(train_sizes, train_prices, range(0, 201), range(0, 41))

# ------------------------------------------------------------
# VALIDATE MODEL A
# ------------------------------------------------------------

intercept_a, weight_a, train_mse_a = model_a

val_mse_a = evaluate_model(val_sizes, val_prices, intercept_a, weight_a)


# ------------------------------------------------------------
# VALIDATE MODEL B
# ------------------------------------------------------------

intercept_b, weight_b, train_mse_b = model_b

val_mse_b = evaluate_model(val_sizes, val_prices, intercept_b, weight_b)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

if val_mse_a < val_mse_b:
    test_mse = evaluate_model(test_sizes, test_prices, intercept_a, weight_a)
    print("Selected Model: A")
    print("Intercept: ", intercept_a)
    print("Weight: ", weight_a)
    print("Training MSE: ", train_mse_a)
    print("Validation MSE: ", val_mse_a)
    print("Test MSE:", test_mse)

else:
    test_mse = evaluate_model(test_sizes, test_prices, intercept_b, weight_b)
    print("Selected Model: B")
    print("Intercept: ", intercept_b)
    print("Weight: ", weight_b)
    print("Training MSE: ", train_mse_b)
    print("Validation MSE: ", val_mse_b)
    print("Test MSE:", test_mse)
