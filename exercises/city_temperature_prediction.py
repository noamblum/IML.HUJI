import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.dropna(inplace=True)
    df = df[df['Temp'] >= -15] # Remove clearly false data of -72 degrees
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    df_il = df[df['Country'] == "Israel"]
    df_il['Year'] = df['Year'].astype(str)
    il_fig_scatter = px.scatter(df_il, x="DayOfYear", y="Temp", color="Year",
                        title="Temperature by Day of Year Creates a 3rd Degree Polynomial.")
    il_fig_scatter.show()
    std_by_month = df_il.groupby(['Month']).Temp.agg(std='std')
    std_month_bar = px.bar(std_by_month, x=std_by_month.index, y="std", title="Standard Deviation for the Different Months")
    std_month_bar.show()

    # Question 3 - Exploring differences between countries
    df_country_month = df.groupby(['Country', 'Month']).Temp.agg(mean='mean', std='std').reset_index()
    country_by_month_plot = px.line(df_country_month, x="Month", y="mean", color="Country", error_y="std",
                                    title="Average Monthly Temperature by Country")
    country_by_month_plot.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(df_il['DayOfYear'], df_il['Temp'], 0.75)
    loss_by_degree = []
    k_arr = np.arange(1,11)
    for k in k_arr:
        estimator = PolynomialFitting(k)
        estimator.fit(train_x, train_y)
        l = estimator.loss(test_x, test_y)
        print("Degree %d - %.2f" % (k,l))
        loss_by_degree.append(l)
    loss_by_degree = np.array(loss_by_degree)
    loss_by_degree_bar = px.bar(x=k_arr, y=loss_by_degree, title="MSE Value by Polynomial Degree")
    loss_by_degree_bar.show()

    # Question 5 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(5)
    best_model.fit(df_il['DayOfYear'], df_il['Temp'])
    loss_by_country = pd.DataFrame(columns=['Country', 'Loss'])
    for other_country in df['Country'].unique():
        if other_country == "Israel": continue
        df_country = df[df['Country'] == other_country]
        loss_by_country.loc[loss_by_country.shape[0]] =\
            [other_country, best_model.loss(df_country['DayOfYear'], df_country['Temp'])]
    loss_by_country_bar = px.bar(loss_by_country, x="Country", y="Loss", 
                                    title="Israel-fitted Model Loss By Country")
    loss_by_country_bar.show()
        
