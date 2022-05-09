from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        
        def gen_threshold_vector(values: np.ndarray) -> np.ndarray:
            """Generates a vector containing the best threshold, best loss, and best sign

            Args:
                values (ndarray of shape (n_samples, )): The feature vector

            Returns:
                np.ndarray: A vector as follwing: [thr, err, sign]
            """
            res = np.array([self._find_threshold(values, y, -1) ,self._find_threshold(values, y, 1)])
            minimizer = np.argmin(res, axis=0)[1]
            return np.array([res[minimizer, 0], res[minimizer, 1], minimizer * 2 - 1])

        possible_thresholds = np.apply_along_axis(gen_threshold_vector, 1, X)
        self.j_ = np.argmin(possible_thresholds, axis=0)[1]
        self.threshold_ = possible_thresholds[self.j_, 0]
        self.sign_ = possible_thresholds[self.j_, 0]


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        cutoff = (X[:, self.j_] >= self.threshold_).astype(int)
        cutoff = cutoff * 2 - 1
        return cutoff * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = values.shape[0]
        sorted_labels = labels[values.argsort()], dtype=int
        d = np.abs(sorted_labels)
        sorted_labels = np.sign(sorted_labels, dtype=int)
        sorted_values = values.sort()
        min_loss = np.Infinity
        for i in range(n_samples + 1):
            lower_labels = np.ones(i, dtype=int) * (-sign)
            higher_labels = np.ones(n_samples - i, dtype=int) * (sign)
            l = np.concatenate([lower_labels, higher_labels])
            cur_loss = np.sum((sorted_labels == l) * d) / n_samples
            if cur_loss < min_loss:
                min_loss = cur_loss
                min_loss_ind = i
        thr = sorted_values[min_loss_ind] if min_loss_ind < n_samples else sorted_values[-1] + 1 # The +1 is to set the threshold above all samples
        return thr, min_loss
        

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return np.sum((self._predict(X) == np.sign(y)) * y) / y.shape[0]
