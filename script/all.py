#!/usr/bin/env python
# coding: utf-8


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotnine
from plotnine import *


str_execute_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
MODEL_PICKLE_FN_NAME = f'extratree-{str_execute_time}.pkl'
DATA_PATH = '../data'
PICKLE_PATH = '../data'


def draw_bacteria(query="Streptococcus_pyogenes"):
    se_ = train.query('target==@query').drop(['row_id', 'target'], axis=1).sum()
    se_ = (se_ - se_.min()) / (se_.max() - se_.min())
    se_
    plt.figure(figsize=(25, 8))
    se_.sort_values().plot.bar()

def draw_bacteria(se_):
    se_ = (se_ - se_.min()) / (se_.max() - se_.min())
    se_ = se_.sort_values()
    df_ = se_.rename('react').reset_index()
    df_ = df_.rename(columns={'index': 'gene'})
    df_['color'] = 'no-color'
    df_.loc[df_.head(10).index, 'color'] = 'min 10'
    df_.loc[df_.tail(10).index, 'color'] = 'max 10'
    df_.gene = pd.Categorical(df_.gene, categories=df_.gene, ordered=True)
    return ggplot(df_, aes(x='gene', y='react', color='color')) + geom_col() + geom_hline(yintercept=.5)




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
train.head()
test.head()
train.shape, test.shape
train['target'].value_counts().plot.bar()
train.iloc[0]





non_duplicate =  ~train.iloc[:, 1:-1].duplicated()
train = train[non_duplicate]
train.shape


y = train['target']
X = train.drop(['row_id', 'target'], axis=1)


epsilon = 1e-5
tmp_X = X + epsilon
tmp = tmp_X.sum(1)
X = tmp_X.apply(lambda x: x / tmp)


A_column = X.columns.str.extract(r'A(\d)').astype(int).iloc[:, 0].to_list()
T_column = X.columns.str.extract(r'T(\d)').astype(int).iloc[:, 0].to_list()
C_column = X.columns.str.extract(r'C(\d)').astype(int).iloc[:, 0].to_list()
G_column = X.columns.str.extract(r'G(\d)').astype(int).iloc[:, 0].to_list()
tmp_X = X.copy()


tmp_X['A'] = X.mul(A_column, axis=1).sum(1) / sum(A_column)
tmp_X['T'] = X.mul(T_column, axis=1).sum(1) / sum(T_column)
tmp_X['C'] = X.mul(C_column, axis=1).sum(1) / sum(C_column)
tmp_X['G'] = X.mul(G_column, axis=1).sum(1) / sum(G_column)
X = tmp_X.copy()

X['CG'] = X['C'] * X['G']
X['AT'] = X['A'] * X['T']

# 0.9626814604819395
# 0.9622171171611515
# 0.9622904345275918, 0.9669094286133242
# 0.9726037440735129
# 0.9711373967447089, 0.9717483747983773



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = ExtraTreesClassifier(n_estimators=5000)
model.fit(X_train, y_train)
MODEL_PICKLE_FN_NAME

model.score(X_train, y_train)
model.score(X_test, y_test)

X_pred = test.drop(['row_id'], axis=1)
X_pred.head()

epsilon = 1e-5
tmp_X = X_pred + epsilon
tmp = tmp_X.sum(1)
X_pred = tmp_X.apply(lambda x: x / tmp)

A_column = X_pred.columns.str.extract(r'A(\d)').astype(int).iloc[:, 0].to_list()
T_column = X_pred.columns.str.extract(r'T(\d)').astype(int).iloc[:, 0].to_list()
C_column = X_pred.columns.str.extract(r'C(\d)').astype(int).iloc[:, 0].to_list()
G_column = X_pred.columns.str.extract(r'G(\d)').astype(int).iloc[:, 0].to_list()
tmp_X = X_pred.copy()

tmp_X['A'] = X_pred.mul(A_column, axis=1).sum(1) / sum(A_column)
tmp_X['T'] = X_pred.mul(T_column, axis=1).sum(1) / sum(T_column)
tmp_X['C'] = X_pred.mul(C_column, axis=1).sum(1) / sum(C_column)
tmp_X['G'] = X_pred.mul(G_column, axis=1).sum(1) / sum(G_column)
X_pred = tmp_X.copy()

X_pred['CG'] = X_pred['C'] * X_pred['G']
X_pred['AT'] = X_pred['A'] * X_pred['T']

y_pred = model.predict(X_pred)
y_pred

time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
df_pred = pd.concat([test['row_id'], pd.Series(y_pred.reshape(-1), name='target')], axis=1)
name = f'{PICKLE_PATH}/submission-{time}.csv'
print(name)
df_pred.to_csv(name, index=False)
df_pred




# get_ipython().system('kaggle competitions submit -c tabular-playground-series-feb-2022 -f {name} -m "Extratree embedding with random forest"')
# !kaggle competitions submit -c tabular-playground-series-feb-2022 -f {name} -m "Extratree embedding with random forest"











