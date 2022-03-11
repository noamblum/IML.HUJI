from turtle import color
from unicodedata import name
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    TRUE_EXPECTATION = 10
    TRUE_VARIANCE = 1
    N_SAMPLES = 1000

    # Question 1
    SAMPLES = np.random.normal(TRUE_EXPECTATION, TRUE_VARIANCE, (N_SAMPLES,))
    estimator = UnivariateGaussian().fit(SAMPLES)
    print(f"({estimator.mu_ }, {estimator.var_})")
    

    # Question 2
    def estimate_expected(n_samples):
        return UnivariateGaussian().fit(SAMPLES[0:n_samples]).mu_

    def estimate_variance(n_samples):
        return UnivariateGaussian().fit(SAMPLES[0:n_samples]).var_

    sizes = np.arange(10, N_SAMPLES + 10, 10)
    expected_values_diff = np.abs(np.vectorize(estimate_expected)(sizes) - TRUE_EXPECTATION)
    variances_diff = np.abs(np.vectorize(estimate_variance)(sizes) - TRUE_VARIANCE)
    
    difference_fig = go.Figure()
    difference_fig.add_trace(
        go.Scatter(x=sizes, y=expected_values_diff, name= "Expected Value",
            line=dict(color='firebrick', width=2))
    )
    difference_fig.add_trace(
        go.Scatter(x=sizes, y=variances_diff, name= "Variance",
            line=dict(color='royalblue', width=2))
    )
    difference_fig.update_layout(title = "Quality of estimation by amount of samples in estimation",
                        xaxis_title="Amount of Samples",
                        yaxis_title="Distance from Real Value")
    difference_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_fig = go.Figure()
    pdf_fig.add_trace(go.Scatter(x=SAMPLES, y=estimator.pdf(SAMPLES), mode='markers',
                        marker=dict(color='firebrick'), showlegend=False))
    pdf_fig.update_layout(title = "Estimated Point Density Function of Samples",
                        xaxis_title="Sample Value",
                        yaxis_title="Estimated PDF Value")
    pdf_fig.show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    #test_multivariate_gaussian()
