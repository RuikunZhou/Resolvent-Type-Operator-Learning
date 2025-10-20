"""
Data generation module for ODE trajectory computation.
Supports parallel computation using multiprocessing.
"""

import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing
import os


def solve_single_trajectory(initial_condition, ode_function, t_span, t_eval, ode_params):
    """
    Solve ODE for a single initial condition.
    
    Parameters:
    -----------
    initial_condition : array-like
        Initial state
    ode_function : callable
        ODE function with signature f(t, y, **params)
    t_span : tuple
        Time span (t_start, t_end)
    t_eval : array-like
        Time points for evaluation
    ode_params : dict
        Additional parameters for ODE function
        
    Returns:
    --------
    solution : ndarray
        Trajectory data [dim, time_points]
    """
    solution = solve_ivp(
        lambda t, y: ode_function(t, y, **ode_params),
        t_span,
        initial_condition,
        t_eval=t_eval,
        method='Radau',
        dense_output=True,
        atol=1e-10,
        rtol=1e-9
    )
    return solution.y


def generate_trajectory_data(ode_function, dim, M_for_1d, x_lim, span, 
                            gen_frequency, ode_params=None, random_seed=42):
    """
    Generate trajectory data for multiple initial conditions in parallel.
    
    Parameters:
    -----------
    ode_function : callable
        ODE function with signature f(t, y, **params)
    dim : int
        System dimension
    M_for_1d : int
        Number of samples per dimension
    x_lim : float
        Spatial limit for initial conditions
    span : float
        Time span for integration
    gen_frequency : int
        Number of time points per unit time
    ode_params : dict, optional
        Additional parameters for ODE function
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    sample : ndarray
        Initial conditions [M, dim]
    flow_data : ndarray
        Trajectory data [M, dim, time_points]
    """
    if ode_params is None:
        ode_params = {}
    
    np.random.seed(random_seed)
    
    M = M_for_1d ** dim
    t_span = [0, span]
    NN = gen_frequency * span + 1
    t_eval = np.linspace(0, span, NN)
    
    # Generate initial conditions
    sample = np.random.uniform(-x_lim, x_lim, (M, dim))
    initial_setups = [sample[i] for i in range(M)]
    
    # Use all available CPU cores
    num_processes = multiprocessing.cpu_count()
    print(f'Using {num_processes} CPU cores for parallel computation')
    
    print('Starting ODE integration...')
    tic = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            solve_single_trajectory,
            [(setup, ode_function, t_span, t_eval, ode_params) for setup in initial_setups]
        )
    
    # Stack results
    flow_data = np.stack(results, axis=0)
    
    total_time = time.time() - tic
    print(f"Data generation completed in {total_time:.2f} seconds")
    
    return sample, flow_data


def save_trajectory_data(sample, flow_data, system_name, M_for_1d, span, x_lim, 
                         frequencies, data_folder="data"):
    """
    Save trajectory data at different sampling frequencies.
    
    Parameters:
    -----------
    sample : ndarray
        Initial conditions
    flow_data : ndarray
        Full trajectory data
    system_name : str
        Name of the dynamical system
    M_for_1d : int
        Number of samples per dimension
    span : float
        Time span
    x_lim : float
        Spatial limit
    frequencies : list
        List of sampling frequencies to save
    data_folder : str, optional
        Folder to save data
    """
    os.makedirs(data_folder, exist_ok=True)
    
    M, dim, NN = flow_data.shape
    
    # Save initial conditions
    filenameX = os.path.join(
        data_folder,
        f'{system_name}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy'
    )
    np.save(filenameX, sample)
    print(f"Saved initial conditions to {filenameX}")
    
    # Save flow data at different frequencies
    for freq in frequencies:
        loc = np.arange(0, NN, max(1, round((NN - 1) / span / freq)))
        filenameY = os.path.join(
            data_folder,
            f'{system_name}_FlowData_{freq}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy'
        )
        np.save(filenameY, flow_data[:, :, loc])
        print(f"Saved flow data (frequency={freq}) to {filenameY}")
