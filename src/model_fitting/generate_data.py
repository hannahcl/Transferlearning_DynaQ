import numpy as np
from nptyping import NDArray, Shape, Float
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def model(coefs, x, max_order = 5, eps=0.0):
    coefs = np.concatenate(
        (np.zeros(max_order - (len(coefs)-1)), coefs))

    order = len(coefs)
    y = np.zeros_like(x)
    for i in range(order):
        y += coefs[i] * (x ** (order - (i+1)))
    y += eps
    return y

def generate_data(model_coeffs, n_samples):

    x = np.random.uniform(-2, 2, n_samples)
    eps = np.random.normal(0, 1, n_samples)
    y = model(model_coeffs, x, eps=eps)

    return x, y

def plot_4_order_model():
    n_samples = 200
    true_model_coeffs = np.array([1., 0., -4., 0.])

    x_sim, y_sim  = generate_data(true_model_coeffs, n_samples)
    fitted_model_coefs = np.polyfit(x_sim, y_sim, 3)

    x_true = np.linspace(-2, 2, n_samples)
    y_true = model(true_model_coeffs, x_true)
    y_fitted = model(fitted_model_coefs, x_true)

    plt.scatter(x_sim, y_sim, label='Simulated Data', color='black')
    plt.plot(x_true, y_true, label='True Model', color='red')
    plt.plot(x_true, y_fitted, label='Fitted Model', color='blue')
    plt.legend()
    plt.show()

def compare_models():
    n_samples = 500
    n_folds = 5
    max_order = 5
    true_model_coeffs = np.array([0., 1., 0., -4., 0.])

    x_sim, y_sim  = generate_data(true_model_coeffs, n_samples)
    kf = KFold(n_splits=n_folds)

    validation_errors = []

    for order in range(max_order+1):
        validation_errors_per_order = []
        for train_index, test_index in kf.split(x_sim):
            y_train, x_train = y_sim[train_index], x_sim[train_index]
            fitted_model_coefs = np.polyfit(x_train, y_train, order)

            y_test, x_test = y_sim[test_index], x_sim[test_index]
            y_fitted = model(fitted_model_coefs, x_test, max_order)
        
            plt.scatter(x_test, y_fitted, label='Fitted Model', color='blue')
            plt.scatter(x_test, y_test, label='Test Data', color='black')
            plt.show()

            error = np.mean((y_fitted - y_test)**2)
            validation_errors_per_order.append(error)
        validation_errors.append(validation_errors_per_order)


    plt.boxplot(validation_errors)
    plt.xlabel('Polynomial Order')
    plt.ylabel('Validation Mean Squared Error')
    plt.title('Validation Mean Squared Error vs Polynomial Order')
    plt.xticks(range(1, max_order + 2), range(0, max_order + 1))
    plt.show()

if __name__ == "__main__":
    plot_4_order_model()
    # compare_models()





