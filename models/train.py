import pandas as pd
import numpy as np
import pickle
import requests

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train(df_train, y_train, categorical, numerical, C=1.0) :
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model, categorical, numerical):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def run(model_output_path='') : 
    # Dataframe definition
    df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    df.columns = df.columns.str.lower().str.replace(' ', '_') 
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    # Numerical and categorical column definition
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]

    # Training and prediction of model
    C=1.0
    dv, model = train(df_full_train, df_full_train.churn.values, categorical, numerical, C)
    
    # Logging the AUC with test data for diagnostic purposes
    """
    y_pred = predict(df_test, dv, model, categorical, numerical)
    y_test = df_test.churn.values
    auc = roc_auc_score(y_test, y_pred)
    print(auc)
    """

    # Save the model
    filename_def = f'model_C={C}.bin'
    output_file = model_output_path+filename_def

    f_out = open(output_file, 'wb') 
    pickle.dump((dv, model), f_out)
    f_out.close()

    with open(output_file, 'wb') as f_out: 
        pickle.dump((dv, model), f_out)

    # Loading the model and using it for a prediction, will leave for deployment exercises 
    """
    # Load the model
    input_file = 'model_C=1.0.bin'

    with open(input_file, 'rb') as f_in: 
        dv, model = pickle.load(f_in)

    customer = {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'no',
        'multiplelines': 'no_phone_service',
        'internetservice': 'dsl',
        'onlinesecurity': 'no',
        'onlinebackup': 'yes',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 1,
        'monthlycharges': 29.85,
        'totalcharges': 29.85
    }

    X = dv.transform([customer])

    y_pred = model.predict_proba(X)[0, 1]

    print('input:', customer)
    print('output:', y_pred)

    url = 'http://localhost:8888/predict'

    customer = {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'no',
        'multiplelines': 'no_phone_service',
        'internetservice': 'dsl',
        'onlinesecurity': 'no',
        'onlinebackup': 'yes',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'two_year',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 1,
        'monthlycharges': 29.85,
        'totalcharges': 29.85
    }

    response = requests.post(url, json=customer).json()

    response

    if response['churn']:
        print('sending email to', 'asdx-123d')
    """

run()