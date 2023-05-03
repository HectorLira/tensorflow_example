import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import tensorflow as tf

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

boston.head()


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
FEATURES = ['LSTAT', 'RM']
LABEL = 'MEDV'


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
df_train = pd.concat([X_train, Y_train], axis=1)
df_eval = pd.concat([X_test, Y_test], axis=1)


def train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )

def eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )

# def prediction_input_fn(df):
#  return tf.estimator.inputs.pandas_input_fn(
#     x = df,
#     y = None,
#     batch_size = 128,
#     shuffle = False,
#     queue_capacity = 1000
#   )

def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns

def print_rmse(model, df):
  metrics = model.evaluate(input_fn = eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'boston_trained'

plt.hist(df_train.MEDV)
plt.show()

tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True)

linear_model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols(), model_dir = OUTDIR)
linear_model.train(input_fn = train_input_fn(df_train, num_epochs = 10))
# predictions = model.predict(input_fn = prediction_input_fn(df_test))

print_rmse(linear_model, df_eval)
# 12.391966819763184

tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True)

dnn_model = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8],
      feature_columns = make_feature_cols(), model_dir = OUTDIR)
dnn_model.train(input_fn = train_input_fn(df_train, num_epochs = 100))

print_rmse(dnn_model, df_eval)
# 3.889941930770874