from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X)

        if len(X) == 1:
            self.var_ = 0
        elif not self.biased_: # Unbiased sample variance mean
            self.var_ = np.sum(np.power(X - self.mu_ ,2)) / (len(X) - 1)
        else: # Biased sample variance mean
            self.var_ = np.mean(np.power(X - self.mu_,2))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        coefficient = 1/np.sqrt(2 * np.pi * self.var_)
        return coefficient * np.exp((-1 * np.power(X - self.mu_, 2)) / (2 * self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n_over_2 = len(X) / 2
        coefficient = 0 - n_over_2 * np.log(2 * np.pi) - n_over_2 * np.log(sigma)
        return coefficient - (1/(2 * sigma)) * np.sum(np.power(X - mu, 2))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.fitted_ = True

        self.mu_ = np.average(X, axis=0) # 0 is axis of samples
        n_samples = X.shape[0]
        centered_samples = X - self.mu_
        self.cov_ = (1 / (n_samples - 1)) * (np.transpose(centered_samples) @ centered_samples)
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        dim = self.mu_.shape[0]
        if X.shape[1] != dim:
            raise ValueError("Number of features must be equal to number of features fitted")
        
        coefficient = 1 / (np.sqrt(np.power(2 * np.pi, dim) * np.linalg.det(self.cov_)))
        centered_samples = X - self.mu_
        return coefficient * np.exp(-0.5 * MultivariateGaussian.mahalanobis_distance(centered_samples, centered_samples, self.cov_))

    @staticmethod
    def mahalanobis_distance(X1: np.ndarray, X2: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate the Mahalanobis distance (https://en.wikipedia.org/wiki/Mahalanobis_distance)
            between two arrays vectors and a covariance matrix

        Args:
            X1 (np.ndarray of shape (n_samples, n_features)): The first vector array
            X2 (np.ndarray of shape (n_samples, n_features)): The second vector array
            cov (np.ndarray of shape (n_features, n_features)): The covariance matrix
        """
        cov_inv = np.linalg.inv(cov)
        stage_1 = X1 @ cov_inv
        return np.einsum('ij,ji->i', stage_1, np.transpose(X2))

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        dim = mu.shape[0]
        n_samples = X.shape[0]

        coefficient = -0.5 * dim * n_samples * np.log(2 * np.pi) - 0.5 * n_samples * np.linalg.det(cov)
        centered_samples = X - mu
        return coefficient - 0.5 * np.sum(MultivariateGaussian.mahalanobis_distance(centered_samples, centered_samples, cov), axis = 0)
