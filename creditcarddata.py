import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler

# returns train, validate
def get_creditcard_data_normalized():
    df = pd.read_csv('data/creditcard.csv')
    df.columns = map(str.lower, df.columns)
    df.rename(columns={'class': 'label'}, inplace=True)

    # normally distribute amount column
    df['log10_amount'] = np.log10(df.amount + 0.00001)


    # dropping redundant columns
    df = df.drop(['time', 'amount'], axis=1)

    fraud = df[df.label == 1]
    clean = df[df.label == 0]

    clean = clean.sample(frac=1).reset_index(drop=True)

    # training set: exlusively non-fraud transactions
    X_train = clean.iloc[:200_000].drop('label', axis=1)

    # testing  set: the remaining non-fraud + all the fraud 
    X_test = clean.iloc[200_000:].append(fraud).sample(frac=1)

    X_train, X_validate = train_test_split(X_train, test_size=0.2)

    # manually splitting the labels from the test df
    X_test, y_test = X_test.drop('label', axis=1).values, X_test.label.values
    
    pipeline = Pipeline([('normalizer', Normalizer()), ('scaler', MinMaxScaler())])
    pipeline.fit(X_train)

    # transform the training and validation data with these parameters
    return pipeline.transform(X_train), pipeline.transform(X_validate), pipeline.transform(X_test), y_test
