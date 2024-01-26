"""predict_module.py.

This module provides a Predict class for learning and predicting using different
regression models.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures


class Predict:
    """A class for learning and predicting using Linear Regression (LR), Bayesian Ridge
    (BR), or SGDRegressor (AR) models.

    Parameters
    ----------
    model_name : str, optional
        Regression model name ("LR" for Linear Regression, "BR" for Bayesian Ridge, "AR" for SGDRegressor),
        by default "LR".

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The regression model.

    Methods
    -------
    learn(x: np.ndarray, y: np.ndarray) -> None:
        Executes the learning process.

    predict(x_new: np.ndarray) -> np.ndarray:
        Predicts value(s) using the linear model.
    """

    def __init__(self, model_name: str = "LR"):
        """Deafault constructor :param model_name:Regression model name."""
        if model_name == "LR":
            self.model = LinearRegression()
        elif model_name == "BR":
            self.model = BayesianRidge()
        elif model_name == "AR":
            self.model = SGDRegressor()
        pass

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        """Execute learning process :param x: Training data, ndarray of shape
        (n_samples, n_features) :param y: Target values, ndarray of shape (n_samples,)
        :return: None."""
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(x)
        self.model.fit(poly_features, y)

        # self.model.fit(x, y)

        m = self.model.coef_[0]
        b = self.model.intercept_
        print("slope=", m, "intercept=", b)

        print("poly_features:", poly_features)
        plt.scatter(x, f, color="black")
        # predicted_values = [self.model.coef_ * i + self.model.intercept_ for i in x]
        y_predicted = self.model.predict(poly_features)
        print("poly_features:", poly_features)
        print("y_predicted:", y_predicted)
        plt.plot(poly_features, y_predicted, "b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Predict value(s) using linear model :param x_new: Samples, ndarray of shape
        (n_samples, n_features) :return: Mean of predictive distribution of query
        points."""
        return self.model.predict(x_new)


if __name__ == "__main__":
    # Example:
    # def func(vars):
    #     x_loc = vars[0]
    #     y_loc = vars[0]
    #     return x_loc**2 + y_loc**2

    def func(vars: List[int]) -> float:
        """Example function."""
        x_loc = vars[0]
        return x_loc**2

    # vars - samples(features)
    # vars = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    num_features = 1
    num_samples = 20
    x = [[n] for n in range(num_samples)]
    # print(vars)
    # exit(0)
    # vars = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]

    # f - target values
    f = [func(var) for var in x]

    x = np.array(x)
    f = np.array(f)

    for n in range(num_samples):
        print(x[n][0], f[n])

    # print(xy)
    # print(f)

    pr = Predict()
    pr.learn(x, f)

    # # xy_new = np.array([[2, 2], [4, 4]])
    # x_new = np.array([[8], [9], [10]])
    # print(pr.predict(x_new))
