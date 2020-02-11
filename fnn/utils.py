import numpy as np
import warnings

import scipy
from scipy.linalg import hankel, orthogonal_procrustes
from scipy.signal import periodogram, argrelextrema, savgol_filter
from scipy.spatial.distance import pdist, squareform, directed_hausdorff
from scipy.spatial import procrustes

from sklearn.decomposition import TruncatedSVD, FastICA, PCA

import matplotlib.pyplot as plt

from tica import tICA

def train_tica(X, num_hidden=10, time_lag=10, random_seed=None):
    """
    Instantiate and fit a tICA model, and return a function that embeds
    additional datasets
    
    Inputs:
    - X : ndarray with shape (n_timepoints, t_lag, n_features)
    - num_hidden : int, the number of embedding coordinates
    - time_lag : int, the time lag to use to construct the embedding
    - random_seed : ignored
    
    Returns:
    - embed_func : a function that takes additional coordinates and embeds them
    """
    
    tica = tICA(n_components = num_hidden, lag_time=time_lag)
    tica.fit([np.reshape(X, (X.shape[0], -1))])
    embed_func = lambda y : tica.transform([np.reshape(y, (y.shape[0], -1))])[0]
    return embed_func

def train_etd(X, num_hidden=10, random_seed=0):
    """
    Instantiate and fits eigen-time-delay, or Broomhead-King coordinates, and 
    return a function that embeds additional datasets.
    
    Inputs:
    - X : ndarray with shape (n_timepoints, t_lag, n_features)
    - num_hidden : int, the number of embedding coordinates
    - random_seed : int, the seed for the random number generator
    
    Returns:
    - embed_func : a function that takes additional coordinates and embeds them
    """
    #svd = TruncatedSVD(n_components=num_hidden, n_iter=7, random_state=random_seed);
    svd = PCA(n_components=num_hidden);
    svd.fit(np.reshape(X, (X.shape[0], -1)));
    embed_func = lambda y : svd.transform(np.reshape(y, (y.shape[0], -1)))
    return embed_func

def train_ica(X, num_hidden=10, random_seed=0):
    """
    Instantiate and fit an ICA model and return a function that embeds 
    additional datasets.
    
    Inputs:
    - X : ndarray with shape (n_timepoints, t_lag, n_features)
    - num_hidden : int, the number of embedding coordinates
    - random_seed : int, the seed for the random number generator
    
    Returns:
    - embed_func : a function that takes additional coordinates and embeds them
    """
    #svd = TruncatedSVD(n_components=num_hidden, n_iter=7, random_state=random_seed);
    ica = FastICA(n_components=num_hidden, random_state=random_seed);
    ica.fit(np.reshape(X, (X.shape[0], -1)));
    embed_func = lambda y : ica.transform(np.reshape(y, (y.shape[0], -1)))
    return embed_func


def hankel_matrix(data, p=-1, q=None):
    """
    Create the Hankel matrix for a univariate time series. p specifies the width 
    of the matrix
    """
    if p==-1:
        p = len(data)
    if not q:
        q = p
    
    last = data[-p:]
    first = data[-(p+q):-p]
    
    h_mat = hankel(first,last)
    
    return h_mat

def train_test(dataset, sample_size, time_window, std=1.0, split=0.5):
    """
    Given a raw 1D time series, perform a standard rescale, and then find the 
    hankel matrix for the train and test partitions. The 

    dataset : ndarray, a 1D time series
    sample_size : int, the length of the train series
    std : float, the number of standard deviations by which to rescale
    split : float, the relative split between test/train
    """
    n = len(dataset)
    n_split = int((split/(1-split))*sample_size)

    assert n > sample_size + n_split, "Not enough data to make complete split"
    
    hm_train = hankel_matrix(dataset, sample_size, q=time_window)[np.newaxis, ...].T
    hm_test = hankel_matrix(dataset[:(n_split+time_window)], sample_size, q=time_window)[np.newaxis, ...].T
    
    #hm = hankel_matrix(dataset, n-time_window, q=time_window)
    #hm = hm[np.newaxis, ...].T
    #hm_train, hm_test = hm[:sample_size], hm[-n_split:]
    
    mn_train, std_train = np.mean(hm_train), np.std(hm_train)
    
    X_train, X_test = [(item - mn_train)/(std*std_train) for item in (hm_train, hm_test)]
    
    return X_train, X_test


