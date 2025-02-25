import os
import csv
import numpy as np

def load_SUBJECT(filename):
    """ load single subject file """
    # with open(filename, 'r') as csvfile:
    data_load = np.genfromtxt(filename, delimiter=',')


def load_CUSTOMDATA(ROOT):
    """ load entire custom dataset """
    xs = []
    ys = []
    train = os.path.join(ROOT, 'custom_data/data/subject1.csv')
    Xtr, Ytr = load_SUBJECT(train)

    test = os.path.join(ROOT, 'custom_data/data/subject2.csv')
    Xte, Yte = load_SUBJECT(test)
    return Xtr, Ytr, Xte, Yte