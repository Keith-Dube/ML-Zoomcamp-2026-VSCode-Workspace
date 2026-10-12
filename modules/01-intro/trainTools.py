import numpy as np


def predict(sizes, intercept, weight):
    # TODO: Generate predictions using NumPy
    predictions = weight * sizes + intercept

    return predictions


def calculate_mse(actual, predicted):
    # TODO: Calculate MSE using NumPy
    errors = actual - predicted
    sqrdErrors = errors**2
    mse = np.mean(sqrdErrors)
    return mse


def main():
    sizes = np.array([4, 5, 6, 7, 8])
    actual_prices = np.array([145, 160, 180, 190, 210])

    predictions = predict(sizes, 75, 17)
    mse = calculate_mse(actual_prices, predictions)

    print("Predictions:", predictions)
    print("MSE:", mse)


if __name__ == "__main__":
    main()
