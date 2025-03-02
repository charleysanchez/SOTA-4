import os
import csv
import numpy as np
from datetime import datetime
import pandas as pd


def convert_to_unix_time(timestamp_str):
    """Convert ISO 8601 timestamp to Unix time (float)."""
    try:
        return pd.to_datetime(timestamp_str).timestamp()
    except:
        return np.nan

def load_SUBJECT(filename):
    """Load single subject file and convert timestamps."""
    np.random.seed(0)

    # Read CSV using pandas
    df = pd.read_csv(filename)

    # Convert timestamp column to Unix time
    df.iloc[:, 0] = df.iloc[:, 0].apply(convert_to_unix_time)

    # Convert to numpy arrays
    timestamps = df.iloc[:, 0].values.astype(np.float64)  # Unix timestamps
    X = df.iloc[:, 1:9].values.astype(np.float32)  # First 8 EMG channels
    y = df.iloc[:, 9:].values.astype(np.float32)  # Labels

    return timestamps, X, y


def load_CUSTOMDATA(ROOT):
    """ load entire custom dataset """
    xs = []
    ys = []
    train = os.path.join(ROOT, 'subject1.csv')
    timestamps_tr, Xtr, Ytr = load_SUBJECT(train)

    test = os.path.join(ROOT, 'subject2.csv')
    timestamps_te, Xte, Yte = load_SUBJECT(test)
    return timestamps_tr, Xtr, Ytr, timestamps_te, Xte, Yte


def get_CUSTOMDATA(num_training=652845, num_validation=72534, num_test=363251, subtract_mean=True):
    """
    Load the customdata dataset and perform preprocessing.
    """
    script_dir = os.path.abspath(os.path.dirname(__file__))
    sota_dir = os.path.join(script_dir, 'SOTA-4')

    # loop to continue backwards until the SOTA-4 directory is found
    while not os.path.exists(sota_dir):
        script_dir = os.path.dirname(script_dir)
        sota_dir = os.path.join(script_dir, 'SOTA-4')
        if script_dir == '/': # if hit root, raise error
            raise FileNotFoundError("SOTA-4 directory not found!")

    customdata_dir = os.path.join(sota_dir, 'custom_data/kao_data')
    timestamps_train, X_train, y_train, timestamps_test, X_test, y_test = load_CUSTOMDATA(customdata_dir)

    # create a mask to get appropriate validation elements out of X_train
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    timestamps_val = timestamps_train[mask]

    # repeat for training examples
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    timestamps_train = timestamps_train[mask]

    if subtract_mean:
      mean_signal = np.mean(X_train, axis=0)
      X_train -= mean_signal
      X_val -= mean_signal
      X_test -= mean_signal

    return {
        'timestamps_train': timestamps_train,
        'X_train': X_train, 'y_train': y_train,
        'timestamps_val': timestamps_val,
        'X_val': X_val, 'y_val': y_val,
        'timestamps_test': timestamps_test,
        'X_test': X_test, 'y_test': y_test
    }

    