"""
Example: Reversed Van der Pol Oscillator
Demonstrates the resolvent learning framework for a 2D polynomial system.
"""

import numpy as np
import sympy
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    generate_trajectory_data,
    save_trajectory_data,
    precompute_basis_functions,
    compute_koopman_generator,
    compute_rmse,
    plot_rmse_vs_mu,
    get_weight_indices,
    exact_coefficients
)

# ============================================================================
# User-Defined System (Modify this section for your own dynamics)
# ============================================================================

# Define system parameters
mu = 1.0  # Van der Pol parameter

# Define symbolic variables (must match dimension)
x1, x2 = sympy.symbols('x1 x2')

# Define dynamics as symbolic expressions
# f[i] represents dx_i/dt
f_symbolic = [
    -x2,                          # dx1/dt
    x1 - mu * (1 - x1**2) * x2   # dx2/dt
]

# System name (will be used for saving files)
SYSTEM_NAME = 'Van_der_Pol'

# ============================================================================
# Derived Configuration (Auto-generated from user definitions)
# ============================================================================

DIM = len(f_symbolic)
SYMBOLS = [x1, x2][:DIM]  # Extract only the symbols needed

# Convert symbolic expressions to numerical function
def ode_function(t, var):
    # Create substitution dictionary
    subs_dict = {SYMBOLS[i]: var[i] for i in range(DIM)}
    
    # Evaluate symbolic expressions
    result = [float(f_symbolic[i].subs(subs_dict)) for i in range(DIM)]
    
    return np.array(result)


# ============================================================================
# Configuration Parameters
# ============================================================================

# Data generation parameters
M_FOR_1D = 10  # Number of samples per dimension
X_LIM = 1  # Spatial limit for initial conditions
SPAN = 1  # Time span
GEN_FREQUENCY = 10000  # High frequency for data generation
SAVE_FREQUENCIES = [10, 50, 100]  # Frequencies to save

# Learning parameters
FREQUENCY = 100  # Frequency to use for learning
MONOMIAL_DEGREES = [3, 3]  # Polynomial degrees for each dimension
LAMDA = 1e8  # Regularization parameter

# Mu configuration
# Option 1: Set to None to generate trend plot with multiple mu values
# Option 2: Set to a specific value (e.g., 2.0) to compute single result
MU_MODE = None  # Set to 2.0 for single mu evaluation, None for trend analysis

# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"Koopman Resolvent Learning: {SYSTEM_NAME}")
    print("=" * 70)
    
    # Step 1: Data Generation/Loading
    print("\n[Step 1] Generating trajectory data...")
    tic_data = time.time()
    
    M = M_FOR_1D ** DIM
    
    # Check if data at target frequency already exists
    sample_file = os.path.join("data", f'{SYSTEM_NAME}_SampleData_samples_{M_FOR_1D}_span_{SPAN}_x_{X_LIM}.npy')
    flow_file_target = os.path.join("data", f'{SYSTEM_NAME}_FlowData_{FREQUENCY}_samples_{M_FOR_1D}_span_{SPAN}_x_{X_LIM}.npy')
    
    if os.path.exists(sample_file) and os.path.exists(flow_file_target):
        print(f"✓ Loading existing data from {os.path.dirname(sample_file)}/...")
        sample = np.load(sample_file)
        flow_data_target = np.load(flow_file_target)
        data_gen_time = time.time() - tic_data
        print(f"✓ Data loaded.")
    else:
        print(f"Generating new trajectory data at high frequency...")
        sample, flow_data = generate_trajectory_data(
            ode_function=ode_function,
            dim=DIM,
            M_for_1d=M_FOR_1D,
            x_lim=X_LIM,
            span=SPAN,
            gen_frequency=GEN_FREQUENCY,
            random_seed=42
        )
        
        data_gen_time = time.time() - tic_data
        print(f"✓ Data generation completed in {data_gen_time:.2f} seconds")
        
        # Save data at different frequencies
        save_trajectory_data(
            sample, flow_data, SYSTEM_NAME, M_FOR_1D, SPAN, X_LIM,
            SAVE_FREQUENCIES, data_folder="data"
        )
        
        # Load the target frequency data that was just saved
        print(f"Loading data at target frequency {FREQUENCY}...")
        flow_data_target = np.load(flow_file_target)
        print(f"✓ Loaded flow data with shape: {flow_data_target.shape}")
    
    # Step 3: Precompute basis functions
    print(f"\n[Step 3] Precomputing basis functions (degrees={MONOMIAL_DEGREES})...")
    tic_basis = time.time()
    
    eta_flow_all, eta_sample_all = precompute_basis_functions(
        sample, flow_data_target, MONOMIAL_DEGREES
    )
    
    basis_time = time.time() - tic_basis
    print(f"✓ Basis precomputation completed in {basis_time:.2f} seconds")
    print(f"  Monomial count: {eta_flow_all.shape[2]}")
    
    # Get weight indices
    weight_indices = get_weight_indices(MONOMIAL_DEGREES)
    
    # Step 4: Load or compute exact coefficients
    print(f"\n[Step 4] Loading/computing exact coefficients...")
    coeff_exact = exact_coefficients(
        SYSTEM_NAME, MONOMIAL_DEGREES, f_symbolic, SYMBOLS, save_folder="results"
    )
    print(f"✓ Exact coefficients shape: {coeff_exact.shape}")
    
    # Step 5: Resolvent Learning
    print(f"\n[Step 5] Resolvent learning...")
    
    NN = flow_data_target.shape[2]
    t_data = np.linspace(0, SPAN, NN)
    
    os.makedirs("results", exist_ok=True)
    
    if MU_MODE is None:
        # Option 1: Trend analysis with multiple mu values
        print("Mode: Trend analysis (multiple mu values)")
        mu_values = np.concatenate((
            np.array([0.02]),
            np.arange(0.25, 4, 0.25),
            np.arange(4, 21, 1)
        ))
        
        error_values = []
        weights_dict = {}
        
        tic_learning = time.time()
        for mu in mu_values:
            print(f"  Processing mu = {mu:.2f}...", end=' ')
            
            # Compute Koopman generator
            L = compute_koopman_generator(
                eta_flow_all, eta_sample_all, mu, LAMDA, t_data, SPAN
            )
            
            # Extract weights for linear terms
            logfree_weights = np.vstack([L[:, idx] for idx in weight_indices])
            
            # Compute RMSE
            rmse = compute_rmse(logfree_weights, coeff_exact)
            error_values.append(rmse)
            weights_dict[mu] = logfree_weights
            
            print(f"RMSE = {rmse:.2e}")
        
        learning_time = time.time() - tic_learning
        print(f"✓ Learning completed in {learning_time:.2f} seconds")
        
        # Save and plot results
        error_values = np.array(error_values)
        np.save(f'{SYSTEM_NAME}_error_values_f{FREQUENCY}_span{SPAN}.npy', error_values)
        
        # Find best mu value
        best_idx = np.argmin(error_values)
        best_mu = mu_values[best_idx]
        best_rmse = error_values[best_idx]
        
        plot_rmse_vs_mu(
            mu_values, error_values, SYSTEM_NAME, FREQUENCY, SPAN,
            LAMDA, MONOMIAL_DEGREES,
            save_path=f'{SYSTEM_NAME}_RMSE_vs_Mu_f{FREQUENCY}_span{SPAN}.png'
        )
        
    else:
        # Option 2: Single mu evaluation
        mu = MU_MODE
        print(f"Mode: Single mu evaluation (mu = {mu})")
        
        tic_learning = time.time()
        
        # Compute Koopman generator
        L = compute_koopman_generator(
            eta_flow_all, eta_sample_all, mu, LAMDA, t_data, SPAN
        )
        
        # Extract weights for linear terms
        logfree_weights = np.vstack([L[:, idx] for idx in weight_indices])
        
        # # print the computed weights and exact coefficients
        # print(f"\nComputed weights:\n{logfree_weights}")
        # print(f"\nExact coefficients:\n{coeff_exact}\n")    
        
        # Compute RMSE
        rmse = compute_rmse(logfree_weights, coeff_exact)
        
        learning_time = time.time() - tic_learning
        print(f"✓ Learning completed in {learning_time:.2f} seconds")
        
        # Save weights
        weight_file = os.path.join(
            "results",
            f'{SYSTEM_NAME}_weights_M{M_FOR_1D}_f{FREQUENCY}_mu{mu}_span{SPAN}.npy'
        )
        np.save(weight_file, logfree_weights)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"Results for mu = {mu}")
        print(f"{'='*70}")
        print(f"RMSE: {rmse:.6e}")
        print(f"\nComputed weights:\n{logfree_weights}")
        print(f"\nExact coefficients:\n{coeff_exact}")
        print(f"\nWeight file saved to: {weight_file}")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Data generation time: {data_gen_time:.2f} seconds")
    print(f"Basis precomputation time: {basis_time:.2f} seconds")
    print(f"Learning time: {learning_time:.2f} seconds")
    print(f"Total time: {data_gen_time + basis_time + learning_time:.2f} seconds")
    if MU_MODE is None:
        print(f"\nBest mu value: {best_mu:.2f} with RMSE = {best_rmse:.6e}")
    print(f"{'='*70}")
