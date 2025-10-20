# Koopman Resolvent Learning Framework

A general framework for learning Koopman generators using resolvent-based methods for polynomial dynamical systems.

## Overview

This package implements an efficient resolvent-based learning algorithm for approximating Koopman generators of polynomial dynamical systems. The framework supports:

- **Arbitrary dimensions**: 2D, 3D, and higher-dimensional systems
- **Flexible polynomial bases**: User-defined monomial degrees for each dimension
- **Parallel data generation**: Efficient trajectory computation using multiprocessing
- **Two evaluation modes**: 
  - Trend analysis with multiple μ values
  - Single μ evaluation for quick results

## Installation

No installation required. Simply ensure you have the following dependencies:

```bash
pip install numpy scipy matplotlib
```

## Project Structure

```
koopman_resolvent_learning/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_generator.py        # Trajectory data generation
│   ├── basis_functions.py       # Polynomial basis computation
│   ├── resolvent_learning.py    # Core resolvent learning algorithm
│   └── visualization.py         # Plotting and result saving
├── examples/
│   ├── example_van_der_pol.py   # 2D Van der Pol oscillator
│   └── example_lorenz_scaled.py # 3D Scaled Lorenz system
└── README.md                     # This file
```

## Quick Start

### Defining Your Own System

The framework makes it easy to define custom polynomial systems using SymPy. Simply modify the user-defined section:

```python
# ============================================================================
# User-Defined System (Modify this section for your own dynamics)
# ============================================================================

# Define system parameters
mu = 1.0

# Define symbolic variables (must match dimension)
x1, x2 = sympy.symbols('x1 x2')

# Define dynamics as symbolic expressions
# f[i] represents dx_i/dt
f_symbolic = [
    -x2,                          # dx1/dt
    x1 - mu * (1 - x1**2) * x2   # dx2/dt
]

# Define domain for initial conditions [min, max] for each dimension
domain = [[-2.5, 2.5], [-3.5, 3.5]]

# System name (will be used for saving files)
SYSTEM_NAME = 'Van_der_Pol'
```

The system definition automatically:
- Converts symbolic expressions to numerical ODE function
- Extracts exact coefficients for validation
- Handles arbitrary polynomial systems

### Example 1: Van der Pol Oscillator (2D)

```bash
cd examples
python example_van_der_pol.py
```

This example demonstrates:
- 2D system with polynomial nonlinearity
- Monomial degrees: [3, 3]
- Trend analysis mode (multiple μ values)

**Key Configuration:**
```python
MONOMIAL_DEGREES = [3, 3]  # Polynomial degrees for [x1, x2]
MU_MODE = None             # Trend analysis mode
FREQUENCY = 100            # Sampling frequency
```

### Example 2: Scaled Lorenz System (3D)

```bash
cd examples
python example_lorenz_scaled.py
```

This example demonstrates:
- 3D system with polynomial nonlinearity
- Monomial degrees: [2, 2, 2]
- Single μ evaluation mode (μ = 2.0)

**Key Configuration:**
```python
MONOMIAL_DEGREES = [2, 2, 2]  # Polynomial degrees for [x, y, z]
MU_MODE = 2.0                  # Single mu evaluation
FREQUENCY = 10                 # Sampling frequency
```

## Configuration Options

### Data Generation Parameters

- `M_FOR_1D`: Number of samples per dimension (total samples = M_FOR_1D^dim)
- `X_LIM`: Spatial limit for initial conditions [-X_LIM, X_LIM]
- `SPAN`: Time span for integration
- `GEN_FREQUENCY`: High frequency for accurate data generation (e.g., 10000)
- `SAVE_FREQUENCIES`: List of frequencies to save downsampled data

### Learning Parameters

- `FREQUENCY`: Sampling frequency to use for learning
- `MONOMIAL_DEGREES`: List of polynomial degrees for each dimension
  - Example: `[3, 3]` for 2D system with cubic polynomials
  - Example: `[2, 2, 2]` for 3D system with quadratic polynomials
- `LAMDA`: Regularization parameter (typically 1e8)

### Mu Evaluation Modes

**Option 1: Trend Analysis** (set `MU_MODE = None`)
- Evaluates multiple μ values: [0.02, 0.25, 0.5, ..., 20]
- Generates RMSE vs μ plot
- Useful for understanding parameter sensitivity

**Option 2: Single Evaluation** (set `MU_MODE = 2.0`)
- Evaluates at a single μ value
- Reports RMSE and saves weights
- Faster computation for quick results

## Creating Custom Examples

To create a new example for your dynamical system:

1. **Define the ODE function:**
```python
def ode_function(t, var, **params):
    x, y, z = var
    # Your system dynamics here
    return np.array([dx_dt, dy_dt, dz_dt])
```

2. **Define exact coefficients** (if known):
```python
def extract_exact_coefficients(monomial_degrees):
    # Map polynomial terms to coefficient matrix
    coeff = np.zeros((DIM, monomial_count))
    # Fill in coefficients
    return coeff
```

3. **Configure parameters:**
```python
SYSTEM_NAME = 'My_System'
DIM = 3
MONOMIAL_DEGREES = [2, 2, 2]
MU_MODE = 2.0  # or None for trend analysis
```

4. **Run the workflow:**
```python
# Generate data
sample, flow_data = generate_trajectory_data(...)

# Precompute basis
eta_flow_all, eta_sample_all = precompute_basis_functions(...)

# Learn Koopman generator
resolvent_weights = compute_resolvent_weights(...)
logfree_weights = compute_logfree_weights(...)
```

## Output Files

The framework generates the following outputs:

### Data Files
- `data/{SYSTEM}_SampleData_*.npy`: Initial conditions
- `data/{SYSTEM}_FlowData_*.npy`: Trajectory data at various frequencies

### Results
- `results/{SYSTEM}_weights_*.npy`: Learned Koopman generator weights
- `{SYSTEM}_error_values_*.npy`: RMSE values for trend analysis
- `{SYSTEM}_RMSE_vs_Mu_*.png`: Visualization of RMSE vs μ

## Performance Tips

1. **Parallel Processing**: Data generation uses all available CPU cores
2. **Frequency Selection**: 
   - Use high `GEN_FREQUENCY` (e.g., 10000) for accurate trajectories
   - Use moderate `FREQUENCY` (e.g., 10-100) for learning to balance accuracy and speed
3. **Monomial Degrees**: Higher degrees increase accuracy but computational cost grows as ∏(degrees)
4. **Sample Size**: Total samples = M_FOR_1D^dim, be mindful of exponential growth

## Mathematical Background

The framework implements the updated resolvent-based learning method for Koopman generators. Given trajectory data, it:

1. Computes polynomial basis functions
2. Solves resolvent equation: (μ²I - L)R(μ) = I
3. Reconstructs Koopman generator using log-free formulation
4. Evaluates RMSE against exact coefficients (if available)

**Key equation:**
- YL = resolvent_learn * (λ - μ) + η_sample
- YR = resolvent_learn * λ * μ - λ * η_sample
- L = (YL)^† @ YR

## Citation

If you use this framework in your research, please cite the original work on log-free learning of Koopman generators.

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact information].
