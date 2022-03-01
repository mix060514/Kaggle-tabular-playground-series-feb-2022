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
train.shape

import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco
from sklearn.model_selection import train_test_split

N_THREADS = 3
MEMORY_LIMIT = 10
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 300
TARGET_NAME = 'target'

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

tr_data, te_data = train_test_split(
    train, 
    test_size=TEST_SIZE, 
    stratify=train[TARGET_NAME], 
    random_state=RANDOM_STATE
)

print(f'Data splitted. Parts sizes: tr_data = {tr_data.shape}, te_data = {te_data.shape}')

tr_data.head()
tr_data.shape


task = Task('multiclass')
task
roles = {
    'target': TARGET_NAME, 
    'drop': ['row_id'],
}

automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    memory_limit = MEMORY_LIMIT, 
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE}
)
oof_pred = automl.fit_predict(tr_data, roles = roles, verbose = 1)




