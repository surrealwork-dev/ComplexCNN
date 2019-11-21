# Import RadioML Data

import pickle
from keras.utils import to_categorical
import numpy as np
from process_radioML_data import *

X, lbl, snrs, classes = read_in_RML()
out = partition_train_test(X, lbl, classes, maxtrain=100, maxtest=200)
[X_train, Y_train, X_test, Y_test] = \
        [ np.array([x.transpose() for x in a]) for a in out]

#################

from fm_ops import *
import torch
from torch import nn

#num_insigs = X_train.shape[0]
#num_classes = len(classes)
class convMod(nn.Module):
    def __init__(self, num_insigs, num_classes):
        super(convMod, self).__init__()

        self.num_insigs = num_insigs
        self.num_classes = num_classes

        self.wfmc = fullConv1d(num_insigs)
        self.r = nn.ReLU()
        self.fc1 = nn.Linear(num_classes)
        self.sm = nn.Softmax()
        
    def forward(self, x):
        wfm, wp = wfmc(x)
        r = self.r(wfm)
        fc1 = self.fc1(r)
        out = self.sm(fc1)

        return out, wp


    
