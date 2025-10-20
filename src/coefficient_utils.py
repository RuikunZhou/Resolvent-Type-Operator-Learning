"""
Utility functions for coefficient extraction from symbolic expressions.
"""

import numpy as np
import sympy
from itertools import product
import os


def extract_exact_coefficients(monomial_degrees, f_symbolic, symbols):
    """
    Extract exact coefficients from symbolic expressions.
    
    Args:
        monomial_degrees: List of polynomial degrees for each dimension
        f_symbolic: List of symbolic expressions for dynamics
        symbols: List of sympy symbols
    
    Returns:
        Coefficient matrix of shape (dim, monomial_count)
    """
    dim = len(f_symbolic)
    monomial_count = int(np.prod(monomial_degrees))
    
    # Generate all basis functions (monomials)
    basis_functions = []
    for powers in product(*[range(deg) for deg in monomial_degrees]):
        # Create monomial: symbols[0]^powers[0] * symbols[1]^powers[1] * ...
        basis = sympy.Integer(1)
        for sym, power in zip(symbols, powers):
            if power > 0:
                basis *= sym**power
        basis_functions.append(basis)
    
    # Initialize coefficient matrix using numpy
    coeff_matrix = np.zeros((dim, monomial_count))
    
    # Extract coefficients for each equation
    for i in range(dim):
        # Expand the expression to get all terms
        expanded_expr = sympy.expand(f_symbolic[i])
        # Get coefficient dictionary
        coeff_dict = expanded_expr.as_coefficients_dict()
        
        # Extract coefficient for each basis function
        for j, basis in enumerate(basis_functions):
            coeff_matrix[i, j] = float(coeff_dict.get(basis, 0))
    
    # Return as numpy array
    coeff_matrix_np = coeff_matrix
    
    return coeff_matrix_np


def exact_coefficients(system_name, monomial_degrees, f_symbolic, 
                       symbols, save_folder="results"):
    """
    Load exact coefficients from file if it exists, otherwise compute and save.
    
    Args:
        system_name: Name of the dynamical system
        monomial_degrees: List of polynomial degrees for each dimension
        f_symbolic: List of symbolic expressions for dynamics
        symbols: List of sympy symbols
        save_folder: Folder to save/load coefficients
    
    Returns:
        Coefficient matrix of shape (dim, monomial_count)
    """
    # Create filename based on system and monomial degrees
    degree_str = '_'.join(map(str, monomial_degrees))
    filename = os.path.join(save_folder, f'{system_name}_exact_coeff_deg{degree_str}.npy')
    
    # Try to load existing file
    if os.path.exists(filename):
        print(f"Loading exact coefficients from {filename}")
        coeff_exact = np.load(filename)
    else:
        print(f"Computing exact coefficients...")
        coeff_exact = extract_exact_coefficients(monomial_degrees, f_symbolic, symbols)
        
        # Save for future use
        os.makedirs(save_folder, exist_ok=True)
        np.save(filename, coeff_exact)
        print(f"Saved exact coefficients to {filename}")
    
    return coeff_exact
