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

def standardize_ts(a, scale=1.0):
    """
    Standardize a T x D time series along its first dimension
    For dimensions with zero variance, divide by one instead of zero
    """
    stds = np.std(a, axis=0, keepdims=True)
    stds[stds==0] = 1
    return (a - np.mean(a, axis=0, keepdims=True))/(scale*stds)

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

def fixed_aspect_ratio(ratio):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    xvals, yvals = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
    
    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    plt.gca().set_aspect(ratio*(xrange/yrange), adjustable='box')

def plot3dproj(x, y, z, *args, color=(0,0,0), shadow_dist=1.0, color_proj=None, 
    elev_azim=(39,-47), show_labels=False, **kwargs):
    """
    Create a three dimensional plot, with projections onto the 2D coordinate
    planes
    
    x, y, z : 1D arrays of coordinates to plot
    *args : arguments passed to the matplotlib plt.plot functions
    color : 3-tuple, the RGB color (with each element in [0,1]) to use for the
        three dimensional line plot
    color_proj : 3-tuple, the RGB color (with each element in [0,1]) to use for the
        two dimensional projection plots. Defaults to a lighter version of the 
        plotting color
    shadow_dist : relative distance of axes to their shadow. If a single value, 
        then the same distance is used for all three axies. If a triple, then 
        different values are used for all axes
    elev_azim : 2-tupe, the starting values of elevation and azimuth when viewing
        the figure
    show_labels : bool, show numerical labels on the axes
    """

    if not color_proj:
        color_proj = lighter(color, .6)


    if np.isscalar(shadow_dist) == 1:
        sdist_x = shadow_dist
        sdist_y = shadow_dist
        sdist_z = shadow_dist
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection= '3d')
    
    ax.plot(x, z, *args, zdir='y', zs=sdist_y*np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir='x', zs=sdist_x*np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir='z', zs=sdist_z*np.min(z), color=color_proj, **kwargs)
    ax.plot(x, y, z, *args, color=color, **kwargs)

    ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    ax.set_aspect('auto', adjustable='box') 
    
    ratio = 1.0
    xvals, yvals = ax.get_xlim(), ax.get_ylim()
    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')

    if not show_labels:
        ax.set_xticklabels([])                               
        ax.set_yticklabels([])                               
        ax.set_zticklabels([])
    #plt.show()

    return ax

def lighter(clr, f=1/3):
    """
    An implementation of Mathematica's Lighter[] 
    function for RGB colors
    clr : 3-tuple or list, an RGB color
    f : float, the fraction by which to brighten
    """
    gaps = [f*(1 - val) for val in clr]
    new_clr = [val + gap for gap, val in zip(gaps, clr)]
    return new_clr