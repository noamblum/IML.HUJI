from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f_true = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    def epsilon(x):
        if type(x) == np.number:
            return np.random.normal(0, noise)
        return np.random.normal(0, noise, x.shape)
    f = lambda x: (f_true(x) + epsilon(x))
    X = pd.DataFrame({"x": np.random.uniform(-1.2, 2, n_samples)})
    y = f(X)["x"]
    y.name = "y"
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=(2/3))
    fig = go.Figure()
    sorted_x = X["x"].sort_values()
    fig.add_traces([
        go.Scatter(x=sorted_x, y=f_true(sorted_x), name="True values", line=dict(color='black', width=1.5)),
        go.Scatter(x=train_X, y=train_y, name="Train set", mode='markers', marker=dict(color='royalblue', symbol="x")),
        go.Scatter(x=test_X, y=test_y, name="Test set", mode='markers', marker=dict(color='firebrick', symbol="circle")),
    ])
    fig.update_layout(title=f"Train and Test Sets Compared to True Model. Noise: {noise}", xaxis_title="x", yaxis_title="y")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()