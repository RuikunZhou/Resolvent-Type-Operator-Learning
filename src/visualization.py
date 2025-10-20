"""
Visualization module for plotting results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rmse_vs_mu(mu_values, error_values, system_name, frequency, span, 
                    lamda, monomial_degrees, save_path=None):
    """
    Plot RMSE vs mu parameter.
    
    Parameters:
    -----------
    mu_values : array-like
        Mu parameter values
    error_values : array-like
        RMSE values
    system_name : str
        Name of the dynamical system
    frequency : int
        Sampling frequency
    span : float
        Time span
    lamda : float
        Regularization parameter
    monomial_degrees : list
        Monomial degrees for each dimension
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(mu_values, error_values, label='Updated Method', alpha=0.7, color='red')
    plt.xlabel(r'$\mu$ in RTM', fontsize=16)
    plt.ylabel(r'$\mathcal{E}_{\operatorname{RMSE}}^{\text{W}}$', fontsize=16)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create title with monomial degree info
    degree_str = 'x'.join(map(str, monomial_degrees))
    plt.title(f'{system_name}: f={frequency}, span={span}, degrees={degree_str}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    
    plt.show()


def save_results(weights, error_values, system_name, M_for_1d, frequency, 
                span, monomial_degrees, results_folder="results"):
    """
    Save computed weights and error values.
    
    Parameters:
    -----------
    weights : dict
        Dictionary mapping mu values to weight arrays
    error_values : dict
        Dictionary mapping mu values to RMSE values
    system_name : str
        Name of the dynamical system
    M_for_1d : int
        Number of samples per dimension
    frequency : int
        Sampling frequency
    span : float
        Time span
    monomial_degrees : list
        Monomial degrees for each dimension
    results_folder : str, optional
        Folder to save results
    """
    import os
    os.makedirs(results_folder, exist_ok=True)
    
    # Save error values
    degree_str = '_'.join(map(str, monomial_degrees))
    error_file = os.path.join(
        results_folder,
        f'{system_name}_errors_M{M_for_1d}_f{frequency}_span{span}_deg{degree_str}.npy'
    )
    np.save(error_file, error_values)
    print(f'Saved errors to {error_file}')
    
    # Save all weights
    for mu, wts in weights.items():
        weight_file = os.path.join(
            results_folder,
            f'{system_name}_weights_M{M_for_1d}_f{frequency}_mu{mu}_span{span}_deg{degree_str}.npy'
        )
        np.save(weight_file, wts)
