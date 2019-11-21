import pickle
import numpy as np

radioMLDict = '/home/allisonrmcalister/RML2016.10a_dict.pkl'

def to_categorical(in_array):
    num_categories = max(in_array) + 1
    in_array = np.array(in_array)
    categorical = np.zeros((in_array.shape[0], num_categories))
    for i, lbl in enumerate(in_array):
        categorical[i][lbl] = 1
    return categorical

def read_in_RML():
    with open(radioMLDict, 'rb') as f:
        snr_mod_dict = pickle.load(f, encoding='latin1')
    
    snrs, mods = map( lambda j: sorted( list( set( map( lambda x: x[j], snr_mod_dict.keys())))), [1,0])
    
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(snr_mod_dict[ (mod, snr) ] )
            for i in range( snr_mod_dict[ (mod, snr) ].shape[0] ):
                lbl.append( (mod, snr) )
    X = np.vstack(X)
    
    classes = mods

    return X, lbl, snrs, classes

def partition_train_test(X, lbl, mods, random_seed=2016, maxtrain=100000, \
                        maxtest=10000):
    np.random.seed(random_seed)
    n_examples  =   X.shape[0]
    n_train     =   min(int(n_examples * 0.5), maxtrain)
    n_test      =   min(n_examples-n_train, maxtest)
    perm        =   np.random.permutation(n_examples)
    train_idx   =   perm[:n_train]
    test_idx    =   perm[-n_test:]
    X_train     =   X[train_idx]
    X_test      =   X[test_idx]
    mods_train  =   [ lbl[ind][0] for ind in train_idx ]
    mods_test   =   [ lbl[ind][0] for ind in test_idx ]
    Y_train     =   [ mods.index(m) for m in mods_train ]
    Y_train     =   to_categorical(Y_train)
    Y_test      =   [ mods.index(m) for m in mods_test ]
    Y_test      =   to_categorical(Y_test)

    return X_train, Y_train, X_test, Y_test

def main(maxtrain=None, maxtest=None):
    X, lbl, snrs, classes = read_in_RML()
    if not maxtrain:
        maxtrain    =   5000
    if not maxtest:
        maxtest     =   1000
    X_train, Y_train, X_test, Y_test = partition_train_test(X, lbl, \
                classes, maxtrain=maxtrain, maxtest=maxtest)
    return X_train, Y_train, X_test, Y_test
