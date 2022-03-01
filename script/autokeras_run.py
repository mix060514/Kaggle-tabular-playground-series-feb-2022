from sklearn.preprocessing import LabelEncoder
import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


str_execute_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
MODEL_PICKLE_FN_NAME = f'extratree-{str_execute_time}.pkl'
DATA_PATH = '../data'
PICKLE_PATH = '../data'

def make_pkl():
    train = pd.read_csv(f'{DATA_PATH}/train.csv')
    test = pd.read_csv(f'{DATA_PATH}/test.csv')
    with open(f'{PICKLE_PATH}/train.pkl', 'wb') as outf:
        pickle.dump(train, outf)
    with open(f'{PICKLE_PATH}/test.pkl', 'wb') as outf:
        pickle.dump(test, outf)

def read_data():
    with open(f'{PICKLE_PATH}/train.pkl', 'rb') as inpf:
        train = pickle.load(inpf)
    with open(f'{PICKLE_PATH}/test.pkl', 'rb') as inpf:
        test = pickle.load(inpf)
    return train, test


train, test = read_data()
print(train.head())

enc = LabelEncoder()
enc.fit(train['target'])

train['target'] = enc.transform(train['target'])


non_duplicate =  ~train.iloc[:, 1:-1].duplicated()
train = train[non_duplicate]
train = train.drop('row_id', axis=1)
test = test.drop('row_id', axis=1)
train.shape

import autokeras as ak

import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=3
)  # It tries 3 different models.
# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    train.drop('target', axis=1),
    # The name of the label column.
    train["target"],
    epochs=10,
)
# Predict with the best model.
predicted_y = clf.predict(test)
pd.Series(predicted_y.squeeze()).unique()

y_pred = enc.inverse_transform(pd.Series(predicted_y.squeeze()).astype(int))
time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
df_pred = pd.concat([test['row_id'], pd.Series(y_pred.reshape(-1), name='target')], axis=1)
name = f'{PICKLE_PATH}/submission-{time}.csv'
print(name)
df_pred.to_csv(name, index=False)
df_pred

# !kaggle competitions submit -c tabular-playground-series-feb-2022 -f {name} -m "by AutoKeras"

