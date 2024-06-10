import numpy as np
from nptyping import NDArray, Shape, Float
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def model(coefs: NDArray[Shape['4'], Float], x: float, eps: float = 0.0) -> float:
    y = coefs[3] + coefs[2]*x + coefs[1]*(x**2) + coefs[0]*(x**3) + eps
    return y

def generate_data(model_coeffs: NDArray[Shape['4'], Float], n_samples: int) -> NDArray[Shape['20'], Float]:

    x = np.random.uniform(-2, 2, n_samples)
    eps = np.random.normal(0, 1, n_samples)

    y = model(model_coeffs, x, eps)

    return x, y

def fit_model(x: NDArray[Shape['20'], Float], y: NDArray[Shape['20'], Float]
              )-> NDArray[Shape['20'], Float]:
    return np.polyfit(x, y, 3)

def ev


if __name__ == "__main__":

    n_samples = 200
    true_model_coeffs = np.array([1., 0., -4., 0.])

    x_generated, y_generated  = generate_data(true_model_coeffs, n_samples)
    fitted_model_coefs = fit_model(x_generated, y_generated)

    x_true = np.linspace(-2, 2, n_samples)
    y_true = model(true_model_coeffs, x_true)

    y_fitted = model(fitted_model_coefs, x_true)

    plt.scatter(x_generated, y_generated, label='Generated Data', color='black')
    plt.plot(x_true, y_true, label='True Model', color='red')
    plt.plot(x_true, y_fitted, label='Fitted Model', color='blue')
    plt.legend()
    plt.show()


