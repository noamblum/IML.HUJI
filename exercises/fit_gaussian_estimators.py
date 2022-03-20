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
    difference_fig.update_layout(title = "Difference Between Estimation and True Value Converges to 0",
                        xaxis_title="Amount of Samples",
                        yaxis_title="Distance from Real Value")
    difference_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_fig = go.Figure()
    pdf_fig.add_trace(go.Scatter(x=SAMPLES, y=estimator.pdf(SAMPLES), mode='markers',
                        marker=dict(color='firebrick'), showlegend=False))
    pdf_fig.update_layout(title = "Estimated Point Density Function of Samples Creates Gaussian",
                        xaxis_title="Sample Value",
                        yaxis_title="Estimated PDF Value")
    pdf_fig.show()



def test_multivariate_gaussian():
    
    TRUE_MU = np.array([0, 0, 4, 0])
    TRUE_COVARIANCE = np.array([[1, 0.2, 0, 0.5],
                                [0.2, 2, 0, 0],
                                [0, 0, 1, 0],
                                [0.5, 0, 0, 1]])
    N_SAMPLES = 1000

    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(TRUE_MU, TRUE_COVARIANCE, N_SAMPLES)
    estimator = MultivariateGaussian().fit(samples)
    print("Mu:\n", estimator.mu_, "\n")
    print("Cov:\n", estimator.cov_)

    # Question 5 - Likelihood evaluation
    def generate_mu_array() -> np.ndarray:
        values = np.linspace(-10, 10, 200)
        combs = np.array(np.meshgrid(values, 0, values, 0)).T.reshape(-1,4)
        return values, combs
    
    values, mu_array = generate_mu_array()
    log_likelihood_func = lambda x : MultivariateGaussian.log_likelihood(x, TRUE_COVARIANCE, samples)
    log_likelihood = np.apply_along_axis(log_likelihood_func, 1, mu_array)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=values, y=values, z=log_likelihood.reshape(200,200).T,
                            colorbar=dict(title="Log Likelihood")))
    fig.update_layout(title = "Log likelihood of samples with mean [f1, 0, f3, 0] Increses as It Approaches True Value of [0, 0, 4, 0]",
                        xaxis_title="Value of f3",
                        yaxis_title="Value of f1")
    fig.show()    
    

    # Question 6 - Maximum likelihood
    print("Maximizing Values:")
    print("f1: %.3f" % mu_array[np.argmax(log_likelihood)][0])
    print("f3: %.3f" % mu_array[np.argmax(log_likelihood)][2])
    


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
