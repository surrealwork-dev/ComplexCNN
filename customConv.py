import numpy as np
from imp import reload
import for_upload.wFmConv as wfc
from process_radioML_data import *

# Read in Input
def read_in_input():
    X,lbl,snrs,classes = read_in_RML()
    X_train, Y_train, X_test, Y_test = partition_train_test(X,lbl,classes,maxtrain=1000, maxtest=500)
    return X_train, Y_train, X_test, Y_test

def make_filters(outdim, num_channels=2, window_length=3, filter_values=None):
    if filter_values:
        filter_array = np.zeros((outdim, num_channels, window_length))
        for i in range(outdim):
            for j in range(len(filter_values)-2):
                for k in range(3):
                    j_ind = j - 3
                    filter_array[i][j_ind][k] = filter_values[i][j][k]
    else:
        filter_array = np.random.rand(outdim, num_channels, window_length)
    return filter_array
    
# Define complex convolution viewed as over a 1-D array of complex numbers
def complex_conv(in_array, outdim, filter_array):
    # Create a feature map for the output.
    num_filters, num_channels, window_length = filter_array.shape
    output = np.zeros((in_array.shape[0], num_filters))

    # Slide a filter-sized window across the input array.
    for filter_num in range(in_array.shape[-1] - 2 - 1):
        window = np.array([[in_array[i][j] for j in range(filter_num, filter_num + window_length)] for i in range(in_array.shape[0])])
        feature = wfc.calc_wfm( np.transpose(window), 
np.transpose(filter_array[filter_num]))

        output[0][filter_num] = feature[0]
        output[1][filter_num] = feature[1]
    return output

def tReLU(complex_feature):
    if type(complex_feature) not in [list, tuple, np.ndarray]:
        complex_feature = [ np.real(complex_feature), np.imag(complex_feature) ]
    
    r, theta = wfc.identify(complex_feature)
    out_r = np.exp(max([np.log(r), 0]))
    out_theta = theta 
    return out_r, out_theta

def vec_trelu(complex_vector):
    out = np.zeros_like(complex_vector)
    for i in range(complex_vector.shape[-1]):
        r,theta = tReLU([complex_vector[0][i], complex_vector[1][i]])
        out[0][i] = r*np.cos(theta)
        out[1][i] = r*np.sin(theta)
    return out

def dist_invariantFC(in_array, weights):
    u = np.zeros(in_array.shape[0])
    m = wfc.calc_wfm(in_array, weights)
    for ind in range(in_array.shape[1]):
        comp_num = [in_array[0][ind], in_array[1][ind]]
        uind = wfc.get_manifold_distance(comp_num, list(m))
        u[ind] = uind
    return u
