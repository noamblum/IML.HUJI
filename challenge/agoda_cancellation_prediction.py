from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from data_cleaner import DataCleaner, TARGET_NAME
from IMLearn import BaseEstimator
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def load_data(filename: str, data_cleaner: DataCleaner, has_labels=True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    has_labels: bool
        Specifies whether the loaded data is train or test data

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename)
    clean_data = data_cleaner.run(full_data).df

    if has_labels:
        features = clean_data.drop(TARGET_NAME, axis=1)
        labels = clean_data[TARGET_NAME]
        return features, labels
    return clean_data



def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    df = pd.read_csv("datasets/agoda_cancellation_train.csv")

    # # Load tarining data
    # data_cleaner = DataCleaner()
    # df, cancellation_labels = load_data("datasets/agoda_cancellation_train.csv", data_cleaner)
    
    # # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(df, cancellation_labels)

    # test_set = load_data("datasets/test_set_week_1.csv", data_cleaner, has_labels=False)
    # evaluate_and_export(estimator, test_set, "challenge/316139922_207540782_318263035.csv")

    # # clean_data = pd.read_csv('clean_data.csv')
    # # df, cancellation_labels = clean_data.drop(TARGET_NAME, axis=1),clean_data[TARGET_NAME]
    l=[]
    for _ in range(10):
        dc = DataCleaner()
        train_df, _, test_df, __ = split_train_test(df, df['cancellation_datetime'], 0.8)

        train_df = dc.run(train_df).df
        test_df = dc.run(test_df.drop(['cancellation_datetime'], axis=1)).df

        train_X = train_df.drop(['cancellation_bin'], axis = 1)
        train_y = train_df['cancellation_bin']
        test_X = test_df.drop(['cancellation_bin'], axis = 1)
        test_y = test_df['cancellation_bin']
        # Fit model over data
        estimator = AgodaCancellationEstimator().fit(train_X, train_y)

        # Store model predictions over test set
        # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
        y = estimator.predict(test_X)
        temp = pd.DataFrame({'id': test_X['h_booking_id'], 'pred': test_y})
        test_y= temp.groupby('id', sort=False)['pred'].max().to_numpy()
        print(np.mean(y==test_y))
        l.append(np.mean(y==test_y))
    print(np.mean(l))