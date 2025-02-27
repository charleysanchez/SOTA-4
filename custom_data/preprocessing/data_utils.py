import os
import csv
import numpy as np

def load_SUBJECT(filename):
    """ load single subject file """
    # with open(filename, 'r') as csvfile:
    np.random.seed(0)
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=float, usecols=range(1, 15))
    np.random.shuffle(data)
    X = data[:, :8]
    y = data[:, 8:]
    return X, y


def load_CUSTOMDATA(ROOT):
    """ load entire custom dataset """
    xs = []
    ys = []
    train = os.path.join(ROOT, 'subject1.csv')
    Xtr, Ytr = load_SUBJECT(train)

    test = os.path.join(ROOT, 'subject2.csv')
    Xte, Yte = load_SUBJECT(test)
    return Xtr, Ytr, Xte, Yte


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
    X_train, y_train, X_test, y_test = load_CUSTOMDATA(customdata_dir)

    # create a mask to get appropriate validation elements out of X_train
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    # repeat for training examples
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    if subtract_mean:
      mean_signal = np.mean(X_train, axis=0)
      X_train -= mean_signal
      X_val -= mean_signal
      X_test -= mean_signal

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

    