"""
Resolvent learning module for Koopman generator approximation.
Implements the updated resolvent-based learning algorithm.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
from scipy.linalg import lstsq


def compute_koopman_generator(eta_flow_all, eta_sample_all, mu, lamda, t_data, span):
    """
    Compute Koopman generator matrix using the resolvent-based learning method.
    
    Parameters:
    -----------
    eta_flow_all : ndarray
        Flow basis functions [M, NN, monomial_count]
    eta_sample_all : ndarray
        Sample basis functions [M, monomial_count]
    mu : float
        Resolvent parameter
    lamda : float
        Regularization parameter
    t_data : ndarray
        Time points
    span : float
        Time span
        
    Returns:
    --------
    L : ndarray
        Koopman generator matrix [monomial_count, monomial_count]
    """
    M, NN, monomial_count = eta_flow_all.shape
    
    # Step 1: Compute resolvent weights
    # Gauss-Legendre quadrature
    x_gauss, w_gauss = leggauss(NN)
    x_gauss_transformed = 0.5 * (x_gauss + 1) * span
    
    # Precompute exponential terms
    exp_term = mu**2 * np.exp(-mu * t_data)
    exp_span = np.exp(-mu * span)
    
    Y = np.zeros((M, monomial_count))
    resolvent_R = np.zeros((M, monomial_count))
    
    for j in range(monomial_count):
        eta_flow = eta_flow_all[:, :, j]
        
        # Compute y_data and integrate
        y_data = exp_term * eta_flow
        interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
        interpolated_values = interpolator(x_gauss_transformed)
        inte = np.dot(interpolated_values, w_gauss) * 0.5 * span
        
        Y[:, j] = inte
        resolvent_R[:, j] = exp_span * eta_flow[:, -1]
    
    # Solve for resolvent weights
    matrix_A = eta_sample_all - resolvent_R
    resolvent_weights = lstsq(matrix_A, Y / mu**2)[0]
    
    # Step 2: Compute Koopman generator from resolvent weights
    resolvent_learn = eta_sample_all @ resolvent_weights
    
    YL = resolvent_learn * (lamda - mu) + eta_sample_all
    YR = resolvent_learn * lamda * mu - lamda * eta_sample_all
    
    pinv_L = np.linalg.pinv(YL)
    L = pinv_L @ YR
    
    return L


def compute_rmse(weights, coeff_exact):
    """
    Compute root mean square error between computed and exact weights.
    
    Parameters:
    -----------
    weights : ndarray
        Computed weights
    coeff_exact : ndarray
        Exact coefficients
        
    Returns:
    --------
    rmse : float
        Root mean square error
    """
    den = np.sqrt(coeff_exact.shape[0] * coeff_exact.shape[1])
    rmse = np.linalg.norm(coeff_exact - weights) / den
    return rmse
