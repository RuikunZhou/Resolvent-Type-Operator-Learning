"""
Basis functions module for polynomial monomials.
Supports arbitrary dimensions and polynomial degrees.
"""

import numpy as np
import itertools


def precompute_basis_functions(sample, flow_data, monomial_degrees):
    """
    Precompute all polynomial monomial basis functions.
    
    Parameters:
    -----------
    sample : ndarray
        Initial conditions [M, dim]
    flow_data : ndarray
        Trajectory data [M, dim, NN]
    monomial_degrees : list or int
        Degree of monomials for each dimension.
        If int, same degree is used for all dimensions.
        
    Returns:
    --------
    eta_flow_all : ndarray
        Flow basis functions [M, NN, monomial_count]
    eta_sample_all : ndarray
        Sample basis functions [M, monomial_count]
    """
    M, dim, NN = flow_data.shape
    
    # Handle uniform degree specification
    if isinstance(monomial_degrees, int):
        monomial_degrees = [monomial_degrees] * dim
    
    assert len(monomial_degrees) == dim, \
        f"monomial_degrees length ({len(monomial_degrees)}) must match dimension ({dim})"
    
    # Compute total number of monomials
    monomial_count = np.prod(monomial_degrees)
    
    eta_flow_all = np.zeros((M, NN, monomial_count))
    eta_sample_all = np.zeros((M, monomial_count))
    
    # Generate all combinations of polynomial degrees
    degree_ranges = [range(deg) for deg in monomial_degrees]
    
    for j, degrees in enumerate(itertools.product(*degree_ranges)):
        # Compute monomial for flow data
        monomial_flow = np.ones((M, NN))
        for d, deg in enumerate(degrees):
            monomial_flow *= np.power(flow_data[:, d, :], deg)
        eta_flow_all[:, :, j] = monomial_flow
        
        # Compute monomial for sample data
        monomial_sample = np.ones(M)
        for d, deg in enumerate(degrees):
            monomial_sample *= np.power(sample[:, d], deg)
        eta_sample_all[:, j] = monomial_sample
    
    return eta_flow_all, eta_sample_all


def get_weight_indices(monomial_degrees):
    """
    Get indices for extracting weights corresponding to linear terms.
    
    The indices correspond to monomials where one variable has degree 1
    and all others have degree 0, in the order [x_0, x_1, x_2, ...].
    
    Parameters:
    -----------
    monomial_degrees : list
        Degree of monomials for each dimension
        
    Returns:
    --------
    indices : list
        Indices for weight extraction, ordered by dimension
    """
    dim = len(monomial_degrees)
    indices = []
    
    # Generate all degree combinations
    degree_ranges = [range(deg) for deg in monomial_degrees]
    
    # Find index for each linear term x_d (where x_d has degree 1, others degree 0)
    for d in range(dim):
        for idx, degrees in enumerate(itertools.product(*degree_ranges)):
            # Check if this is the linear term for dimension d
            is_linear_term = all(
                (i == d and deg == 1) or (i != d and deg == 0)
                for i, deg in enumerate(degrees)
            )
            if is_linear_term:
                indices.append(idx)
                break
    
    return indices
