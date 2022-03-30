from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)
    df=df[df['date'].str.len() > 1]
    df=df[df['price'] >= 0]

    # No features related to date showed any correlation to the house price - not year renovated, not sale date.
    # Lot sqft showed no correlation
    # condition did not show any correlation, only grade
    # lattitude is the geo-feature that showed the most correlation - probably means northern neighborhoods are more expansive

    features = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'sqft_living15', 'zip_short', 'lat']]
    labels = df['price']

    return features, labels
    



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sd_labels = np.std(y)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for feature in X.columns:
        covar = np.cov(X[feature], y)[0][1]
        sd_feature = np.std(X[feature])
        corr = covar/ (sd_feature * sd_labels)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=X[feature], y=y, name= "Price", mode="markers")
        )
        fig.update_layout(title = f"Correlation between {feature} and house price: {corr}",
                        yaxis_title="Price")
        fig.write_image(os.path.join(output_path, f"{feature}.png"))



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data('datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, "figures/house_price")

    # # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
