# Koopman Resolvent Learning Framework - Complete Structure

## Directory Structure

```
koopman_resolvent_learning/
├── src/                              # Core library modules
│   ├── __init__.py                   # Package initialization
│   ├── data_generator.py             # Trajectory data generation
│   ├── basis_functions.py            # Polynomial basis computation
│   ├── resolvent_learning.py         # Resolvent learning algorithm
│   ├── coefficient_utils.py          # Symbolic coefficient extraction
│   └── visualization.py              # Plotting and results
│
├── examples/                         # Example applications
│   ├── example_van_der_pol.py        # 2D Van der Pol oscillator
│   ├── example_lorenz_scaled.py      # 3D Scaled Lorenz system
│   └── example_template.py           # Template for new systems
│
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Setup script
├── .gitignore                        # Git ignore patterns
├── README.md                         # Main documentation
├── CHANGELOG.md                      # API changes log
├── COEFFICIENT_EXTRACTION_UPDATE.md  # Coefficient extraction guide
└── QUICK_REFERENCE.md               # Quick reference guide (if created)
```

## Module Descriptions

### Core Library (`src/`)

#### `data_generator.py`
- `generate_trajectory_data()`: Generate ODE trajectories in parallel
- `save_trajectory_data()`: Save trajectories at multiple frequencies
- Uses multiprocessing for efficient computation

#### `basis_functions.py`
- `precompute_basis_functions()`: Compute polynomial monomial bases
- `get_weight_indices()`: Get indices for weight extraction
- Supports arbitrary dimensions and polynomial degrees

#### `resolvent_learning.py`
- `compute_resolvent_weights()`: Solve resolvent equation
- `compute_koopman_generator()`: Compute Koopman generator matrix L
- `compute_rmse()`: Evaluate accuracy vs exact coefficients

#### `coefficient_utils.py` (NEW)
- `extract_exact_coefficients()`: Extract coefficients from symbolic expressions
- `load_or_compute_exact_coefficients()`: Smart caching of coefficients
- Automatic file management

#### `visualization.py`
- `plot_rmse_vs_mu()`: Generate RMSE vs μ plots
- `save_results()`: Save weights and errors
- Publication-quality figures

## Example Files

### Van der Pol (`example_van_der_pol.py`)
- **System**: 2D autonomous oscillator
- **Dynamics**: dx₁/dt = -x₂, dx₂/dt = x₁ - μ(1-x₁²)x₂
- **Features**: 
  - Symbolic system definition
  - Automatic coefficient extraction
  - Trend analysis mode (multiple μ)
  - Full workflow demonstration

### Scaled Lorenz (`example_lorenz_scaled.py`)
- **System**: 3D chaotic attractor
- **Dynamics**: Standard Lorenz with scaled parameters
- **Features**:
  - 3D system example
  - Single μ evaluation mode
  - Shows higher-dimensional usage

### Template (`example_template.py`)
- Copy-paste starting point for new systems
- Comprehensive comments
- All configuration options explained

## Workflow Summary

### For New Users

1. **Install dependencies**:
   ```bash
   ./setup.sh
   ```

2. **Run an example**:
   ```bash
   cd examples
   python3 example_van_der_pol.py
   ```

3. **Create your own system**:
   - Copy `example_template.py`
   - Define dynamics symbolically
   - Configure parameters
   - Run!

### For Developers

**Key API Pattern**:
```python
# 1. Generate data
sample, flow_data = generate_trajectory_data(ode_function, ...)

# 2. Compute basis
eta_flow, eta_sample = precompute_basis_functions(sample, flow_data, degrees)

# 3. Load/compute exact coefficients (with caching)
coeff_exact = load_or_compute_exact_coefficients(system_name, degrees, f_symbolic, symbols)

# 4. Learn generator
resolvent_weights = compute_resolvent_weights(eta_flow, eta_sample, mu, t_data, span)
L = compute_koopman_generator(eta_sample, resolvent_weights, mu, lamda)

# 5. Extract and evaluate
weights = np.vstack([L[:, idx] for idx in weight_indices])
rmse = compute_rmse(weights, coeff_exact)
```

## Key Features

### 1. Symbolic System Definition
```python
x, y = sympy.symbols('x y')
f_symbolic = [-y, x - mu*(1-x**2)*y]
```
- Define dynamics once
- Automatic coefficient extraction
- ODE function auto-generated

### 2. Smart Coefficient Caching
- First run: computes and saves
- Subsequent runs: loads instantly
- Filename: `{system}_exact_coeff_deg{degrees}.npy`

### 3. Flexible Evaluation Modes
- **Trend Analysis** (`MU_MODE = None`): Test multiple μ values
- **Single Evaluation** (`MU_MODE = 2.0`): Quick verification

### 4. Parallel Data Generation
- Uses all CPU cores automatically
- Progress reporting
- Time benchmarks

### 5. Publication-Ready Outputs
- RMSE vs μ plots
- Saved weight matrices
- Error tracking

## Configuration Parameters

### Essential Parameters
- `SYSTEM_NAME`: Identifier for your system
- `DIM`: System dimension
- `MONOMIAL_DEGREES`: Polynomial degrees per dimension
- `M_FOR_1D`: Samples per dimension (total = M_FOR_1D^DIM)
- `FREQUENCY`: Sampling frequency for learning
- `MU_MODE`: None (trend) or float (single evaluation)

### Advanced Parameters
- `LAMDA`: Regularization (default: 1e8)
- `SPAN`: Integration time (default: 1)
- `GEN_FREQUENCY`: Data generation frequency (default: 10000)
- `SAVE_FREQUENCIES`: List of frequencies to save

## File Outputs

### Data Files (`data/`)
```
{SYSTEM}_SampleData_samples_{M}_span_{span}_x_{xlim}.npy
{SYSTEM}_FlowData_{freq}_samples_{M}_span_{span}_x_{xlim}.npy
```

### Results Files (`results/`)
```
{SYSTEM}_exact_coeff_deg{degrees}.npy           # Exact coefficients (cached)
{SYSTEM}_weights_M{M}_f{freq}_mu{mu}_span{span}.npy  # Learned weights
{SYSTEM}_error_values_f{freq}_span{span}.npy    # RMSE array
{SYSTEM}_RMSE_vs_Mu_f{freq}_span{span}.png      # Visualization
```

## Recent Updates

### v2.0 - Coefficient Extraction Refactor
- Moved `extract_exact_coefficients()` to `coefficient_utils.py`
- Added `load_or_compute_exact_coefficients()` with caching
- Removed duplicate code from examples
- Automatic file management

### v1.1 - API Simplification
- Renamed `compute_logfree_weights()` → `compute_koopman_generator()`
- Returns full L matrix instead of extracted weights
- Weight extraction moved to user code
- More flexible and extensible

## Dependencies

```
numpy>=1.20.0      # Numerical computations
scipy>=1.7.0       # ODE solver, interpolation
matplotlib>=3.4.0  # Visualization
sympy>=1.9.0       # Symbolic math
```

## Performance Notes

- **2D systems**: ~60-120 seconds total (M=100, freq=100)
- **3D systems**: ~150-300 seconds total (M=1000, freq=10)
- Data generation is the bottleneck (parallelized)
- Learning is fast once data is generated
- Coefficient extraction: instant after first computation

## Getting Help

1. Read `README.md` for overview
2. Check `QUICK_REFERENCE.md` for common tasks
3. See `CHANGELOG.md` for API changes
4. Review `example_template.py` for full configuration options
5. Study working examples: `example_van_der_pol.py` and `example_lorenz_scaled.py`

## Contributing

To add a new feature:
1. Add implementation to appropriate `src/` module
2. Update `src/__init__.py` exports
3. Add example usage in `examples/`
4. Update documentation

## License

[Specify license]

## Citation

[Add citation information]
