# !pip install lightautoml[all]

import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

automl = TabularAutoML(
    task = Task(
        name = 'binary',
        metric = lambda y_true, y_pred: f1_score(y_true, (y_pred > 0.5)*1))
)
oof_pred = automl.fit_predict(
    df_train,
    roles = {'target': 'Survived', 'drop': ['PassengerId']}
)
test_pred = automl.predict(df_test)

pd.DataFrame({
    'PassengerId':df_test.PassengerId,
    'Survived': (test_pred.data[:, 0] > 0.5)*1
}).to_csv('submit.csv', index = False)
