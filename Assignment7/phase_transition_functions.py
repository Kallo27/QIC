###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 7 - ISING MODEL


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

import ising_model as im

from scipy.optimize import curve_fit


# ===========================================================================================================
# ENERGY GAP
# ===========================================================================================================

def plot_energy_gaps(N_values, l_values, eigenvalues, no_deg = True):
  """
  plot_energy_gaps :
    Plot the energy gap (between first excited state and ground state) 
    as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvalues : np.ndarray
    Precomputed eigenvalues for every (N, l).
  no_deg : bool
    Flag for computing the energy gap with or without degeneration.
    Default is True (no degeneration).
  
  Returns
  -------
  min_gaps, min_ls : tuple of list
    List of energy gaps and corresponding values of lambda.
  """  
  plt.figure(figsize=(8, 5))
  
  # Select first or second eigenvalue 
  n = 2 if no_deg is True else 1
  
  # Save minimum gaps
  min_gaps = []
  min_ls = []
  
  # Loop over the values of N
  for N in N_values:    
    gaps = []
    # Loop over the first k levels
    for l in l_values:
      gap = (eigenvalues[(N, l)][n] - eigenvalues[(N, l)][0]) / N
      gaps.append(gap)

    # Find the minimum gap and its corresponding λ value
    min_gap = min(gaps)
    min_index = gaps.index(min_gap)
    min_l = l_values[min_index]
    
    # Append to list 
    min_gaps.append(min_gap)
    min_ls.append(min_l)
    
    # Plot the energy gaps
    plt.plot(l_values, gaps, label=f"N = {N}")
    
    # Highlight the minimum gap
    plt.scatter(min_l, min_gap, color="orange")
  
  plt.axvline(x = 1, linestyle="--", color = "red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Energy')
  plt.title(f'Normalized energy gap vs λ')
  plt.legend(loc="upper right")

  plt.grid()
  plt.show()
  
  return min_gaps, min_ls

# ===========================================================================================================

def plot_pt_gap(N_values, ls):
  """
  plot_pt_gap:
    Plot minimum energy gap for different N (phase transition analysis).

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  ls : list of float
    Values of l corresponding to the minimum energy gap.
    
  Returns
  -------
  None
  """
  plt.figure(figsize=(6, 4))

  plt.plot(ls, N_values, marker='o', linestyle='-', color='orange')

  plt.xlim(0.3, 1)
  plt.xlabel('L values')
  plt.ylabel('N values')
  plt.title('Critical point for different N (energy gap)')
  plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  plt.tight_layout()
  plt.show()
  
# ===========================================================================================================
# MAGNETIZATION
# ===========================================================================================================

def magnetization(ground_state, N):
  """
  magnetization:
    Computes the magnetization of the ground state vector for an N-spin system.

  Parameters
  ----------
  ground_state : np.ndarray
    Ground state vector of the system.
  N : int
    Number of spins in the system.

  Returns
  -------
  M : float
    Expectation value of the normalized total magnetization operator.
  """
  _, _, s_z = im.pauli_matrices()  # Retrieve sparse Pauli matrices
  
  M_z = sp.csr_matrix((2**N, 2**N), dtype=complex)
  
  for i in range(N):
    M_z_i = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_z, sp.identity(2**(N - i - 1), format='csr')))
    M_z += M_z_i
    
  M_z /= N
  
  M = ground_state.conj().transpose().dot(M_z.dot(ground_state))
  return M

# ===========================================================================================================

def plot_magnetization(N_values, l_values, eigenvectors):
  """
  plot_magnetization :
    Plot the magnetization as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvecttors : np.ndarray
    Precomputed eigenvectors for every (N, l).
  
  Returns
  -------
  None
  """  
  plt.figure(figsize=(8, 5))
    
  # Loop over the values of N (many plots)
  for N in N_values:    
    Ms = []
    # Loop over the first k levels
    for l in l_values:
      M = magnetization(eigenvectors[(N, l)][0], N)
      Ms.append(M)

    plt.plot(l_values, Ms, marker='^', linestyle='--', label = f"N={N}", markersize=3)
  
  plt.axvline(x = 1, linestyle="--", color = "red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Magnetization')
  plt.title(f'Magnetization vs λ')
  plt.xscale('log')
  plt.legend(loc="lower left")
  plt.grid()
  plt.show()
  
# ===========================================================================================================
  
def plot_pt_magnetization(N_values, l_values, eigenvectors):
  """
  plot_pt_magnetization:
    Plot inflection point for different N (phase transition analysis).

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvecttors : np.ndarray
    Precomputed eigenvectors for every (N, l).
  
  Returns
  -------
  None
  """
  infl_points = []  # To store inflection points for each N

  for N in N_values:    
    Ms = []
    # Compute magnetizations for all `l` values
    for l in l_values:
      M = magnetization(eigenvectors[(N, l)][0], N)  # Assuming ptf.magnetization is defined
      Ms.append(M)
    
    # Compute second derivative
    second_derivative = np.gradient(np.gradient(Ms))
    
    # Find inflection points (where the sign of the second derivative changes)
    sign_changes = np.diff(np.sign(second_derivative))
    infl_points_index = np.where(sign_changes != 0)[0]
    
    # Store the first inflection point index for this N (or process as needed)
    if len(infl_points_index) > 0:
      infl_points.append(l_values[infl_points_index[0]])
    else:
      infl_points.append(None)  # No inflection point found for this N
  
  # Define the model function: -1 + a/N
  def model(N, a):
    return 1 + a / N

  # Perform the curve fitting
  params, covariance = curve_fit(model, N_values, infl_points)

  # Extract fitted parameter
  a_fit = params[0]

  # Generate fitted values for plotting
  N_fit = np.linspace(min(N_values), max(N_values), 500)
  lambda_fit = model(N_fit, a_fit)
  
  # Plot the inflection points as a function of N
  plt.figure(figsize=(6, 4))
  plt.plot(infl_points, N_values, marker='o', linestyle='-', color='orange', label='Data')
  plt.plot(lambda_fit, N_fit, color='red', linestyle='--', label=f'Fit: $+1 {a_fit:.3f}/N$', zorder=2)
  plt.ylabel('N')
  plt.xlabel('Inflection point')
  plt.title('Critical point for different N (magnetization)')
  plt.xlim(0.3, 1)
  plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  plt.legend(fontsize=10)
  plt.tight_layout()
  plt.show()

# ===========================================================================================================
# ENTROPY
# ===========================================================================================================
  
def rdm(psi, N, D, keep_indices):
  """
  rdm :
    Computes the reduced density matrix of a quantum state by tracing out the 
    degrees of freedom of the environment.

  Parameters
  ----------
  psi : np.ndarray
    Wavefunction of the quantum many-body system, represented as a complex vector of 
    size D^N.
  N : int
    Number of subsystems.
  D : int
    Dimension of each subsystem.
  keep_indices : list of int
    Indices of the sites to retain in the subsystem (all other sites are traced out).

  Returns
  -------
  reduced_density_matrix : np.ndarray
    Reduced density matrix for the subsystem specified by keep_indices, which is a 
    square matrix of size (D^len(keep_indices), D^len(keep_indices)).
  """
  # Check correct values for 'keep_indices'
  if not all(0 <= idx < N for idx in keep_indices):
    raise ValueError(f"'keep_indices' must be valid indices within range(n_sites), got {keep_indices}")
    
  # Compute subsystem and environment dimensions
  n_keep = len(keep_indices)
  subsystem_dim = D ** n_keep
  env_dim = D ** (N - n_keep)

  # Reshape the wavefunction into a tensor
  psi_tensor = psi.reshape([D] * N)

  # Reorder the axes to group subsystem (first) and environment (second)
  all_indices = list(range(N))
  env_indices = [i for i in all_indices if i not in keep_indices] # complement of keep_indices
  reordered_tensor = np.transpose(psi_tensor, axes=keep_indices + env_indices)

  # Partition into subsystem and environment (reshape back)
  psi_partitioned = reordered_tensor.reshape((subsystem_dim, env_dim))

  # Compute the reduced density matrix
  rdm = np.dot(psi_partitioned, psi_partitioned.conj().T)

  return rdm

# ===========================================================================================================

def von_neumann_entropy(state_vector, N, D, keep_indices):
  """
  Computes the Von Neumann entropy for a given quantum state vector.

  Parameters
  ----------
  state_vector : np.ndarray
    The quantum state vector of the entire system, assumed to be normalized.
  N : int
    Number of subsystems.
  D : int
    Dimension of each subsystem.
  keep_indices : list of int
    Indices of the sites to retain in the subsystem.

  Returns
  -------
  entropy : float
    The Von Neumann entropy of the subsystem.
  """
  # Compute the reduced density matrix and its eigenvalues
  reduced_density_matrix = rdm(state_vector, N, D, keep_indices)
  eigenvalues = np.linalg.eigvalsh(reduced_density_matrix)

  # Filter out small eigenvalues to avoid log(0)
  non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-12]

  # Compute the Von Neumann entropy
  entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))
  return entropy

# ===========================================================================================================

def plot_entropy(N_values, l_values, eigenvectors):
  """
  plot_entropy :
    Plot the Von Neumann entropy as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvectors : dict
    Precomputed eigenvectors for every (N, l).
  
  Returns
  -------
  None
  """  
  plt.figure(figsize=(8, 5))
    
  # Loop over the values of N
  for N in N_values:    
    Ss = []
    # Loop over the values of N
    for l in l_values:
      S = von_neumann_entropy(eigenvectors[(N, l)][0], N, 2, list(range(N // 2)))
      Ss.append(S)

    plt.plot(l_values, Ss, marker='^', linestyle='--', label = f"N={N}", markersize=3)
  
  plt.axvline(x = 1, linestyle="--", color = "red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Von Neumann entropy')
  plt.title(f'Entropy vs λ')
  plt.xscale('log')
  plt.legend(loc="lower left")
  plt.grid()
  plt.show()

# ===========================================================================================================

def fit_entropy_scaling(N_values, S_values):
  """
  fit_entropy_scaling :
    Fit the entropy data vs ln(N) to estimate the central charge c.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  S_values : list of float
    Values of S, precomputed Von Neumann entropy (lambda = 1).
  
  Returns
  -------
  fit_params : np.ndarray
    The parameters of the fit.
  fit_errors : np.ndarray
    The standard errors of the fit parameters.
  """
  # Define the scaling function
  def scaling_fn(ln_N, c, const):
    return c / 6 * ln_N + const

  # Convert system sizes to natural log
  ln_N = np.log(N_values)

  # Perform the curve fitting
  fit_params, covariance = curve_fit(scaling_fn, ln_N, S_values)
  fit_errors = np.sqrt(np.diag(covariance))

  # Plot the data and the fit
  plt.figure(figsize=(8, 6))
  plt.plot(ln_N, S_values, 'o', label='Data')
  plt.plot(ln_N, scaling_fn(ln_N, *fit_params), '-', label=f'Fit: c = {fit_params[0]:.4f}')
  plt.xlabel('ln(N)', fontsize=12)
  plt.ylabel('Von Neumann entropy', fontsize=12)
  plt.title('Entropy scaling with N', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True)
  plt.show()

  return fit_params, fit_errors

# ===========================================================================================================

def analyze_entropy_scaling(N_values, eigenvectors):
  """
  analyze_entropy_scaling : 
    Analyze entropy scaling for multiple system sizes.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  eigenvectors : dict
    Precomputed eigenvectors for every (N, l).

  Returns
  -------
  None
  """
  normalized_entropies = []

  # Compute normalized entropies for all system sizes
  for N in N_values:
    D = 2
    entropy = von_neumann_entropy(eigenvectors[(N, 1)][0], N, D, list(range(N // 2)))
    normalized_entropies.append(entropy)

  # Fit entropy scaling
  fit_params, fit_errors = fit_entropy_scaling(N_values, normalized_entropies)

  print(f"Estimated central charge: {fit_params[0]} +/- {fit_errors[0]}")
  
# ===========================================================================================================
# TWO POJNT CORRELATION
# ===========================================================================================================  

def two_point_correlation(psi, N, i):
  """
  two_point_correlation : 
    Compute the two-point correlation function C_{i,i+1} = <psi|σ_z^i σ_z^i+1|psi>
    for a given quantum state using sparse matrices.

  Parameters
  ----------
  psi : np.ndarray
    Ground state wavefunction of the system.
  N : int
    Number of spins in the system.
  i : int
    Index of the first spin.

  Returns
  -------
  correlation : float
    Two-point correlation function C_{i,i+1}.
  """
  # Validate input indices
  if not (0 <= i < N):
    raise ValueError(f"Index i must be in range [0, N-1], got i={i}.")

  # Pauli z matrix as a sparse matrix
  s_z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

  # Initialize the operator as the identity matrix
  operator = sp.identity(1, format="csr")

  # Construct σ_z^i σ_z^i+1 using Kronecker products
  for k in range(N):
    if k == i or k == i + 1:
      operator = sp.kron(operator, s_z, format="csr")
    else:
      operator = sp.kron(operator, sp.identity(2, format="csr"), format="csr")

  # Compute the expectation value
  psi_sparse = sp.csr_matrix(psi).reshape(-1, 1)
  correlation = np.abs(np.real((psi_sparse.getH() @ (operator @ psi_sparse)).toarray().item()))

  return correlation

# ===========================================================================================================  

def plot_correlations(N_values, l_values, eigenvectors):
  """
  plot_correlations : 
    Compute and plot the two-point correlation function for different eigenvectors and lambdas.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvectors : dict
    Precomputed eigenvectors for every (N, l).
    
  Returns
  -------
  None
  """
  plt.figure(figsize=(8, 5))

  # Loop over system sizes
  for N in N_values:
    correlations = []

    # Choose random spins i
    i = np.random.choice(range(N))

    # Loop over lambda values
    for l in l_values:
      correlation = two_point_correlation(eigenvectors[(N, l)][0], N, i)
      correlations.append(correlation)

    plt.plot(l_values, correlations, marker='^', linestyle='--', label=f"N={N}", markersize=3)

  plt.axvline(x=1, linestyle="--", color="red", label="Critical point")

  # Plot formatting
  plt.xlabel("Interaction strength (λ)")
  plt.ylabel("Two-point correlation function")
  plt.title(f"Two-point correlation function vs λ")
  plt.xscale("log")
  plt.legend(loc="lower right")
  plt.grid()
  plt.show()

# ===========================================================================================================  

def fit_correlation_scaling(N_values, C_values):
  """
  fit_correlation_scaling :
    Fit the two-point correlation function data to the finite-size scaling relation.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  C_values : list of float
    Correlation function values for each N.
    
  Returns
  -------
  fit_params : np.ndarray
    The parameters of the fit.
  fit_errors : np.ndarray
    The standard errors of the fit parameters.
  """
  # Convert inputs to numpy arrays for easier filtering
  N_values = np.array(N_values)
  C_values = np.array(C_values)

  # Filter out invalid values
  valid_indices = C_values > 0
  if not np.all(valid_indices):
    print(f"Warning: Found non-positive values in correlation data. These will be excluded.")

  # Use only valid values
  log_N = np.log(N_values[valid_indices])
  log_C = np.log(C_values[valid_indices])

  # Define the scaling function
  def scaling_fn(log_N, eta, const):
    return -eta / 2 * log_N + const

  # Perform the curve fitting
  fit_params, covariance = curve_fit(scaling_fn, log_N, log_C)
  fit_errors = np.sqrt(np.diag(covariance))

  # Plot the data and the fit
  plt.figure(figsize=(8, 6))
  plt.plot(log_N, log_C, 'o', label='Data')
  plt.plot(log_N, scaling_fn(log_N, *fit_params), '-', label=f'Fit: η = {fit_params[0]:.4f}')
  plt.xlabel('ln(N)')
  plt.ylabel('ln(two-point correlation function)')
  plt.title('Correlation scaling with N (at λ=1)')
  plt.legend(fontsize=10)
  plt.grid(True)
  plt.show()

  return fit_params, fit_errors

# ===========================================================================================================  

def analyze_correlation_scaling(N_values, eigenvectors):
  """
  analyze_correlation_scaling :
    Analyze the finite-size scaling of the two-point correlation function.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  eigenvectors : dict
    Precomputed eigenvectors for every (N, l).

  Returns
  -------
  None
  """
  correlation_values = []
  
  # Compute two-point correlations for each system size
  for N in N_values:
    i = 4
    correlation = two_point_correlation(eigenvectors[(N, 1)][0], N, i)
    correlation_values.append(correlation)

  # Fit finite-size scaling
  fit_params, fit_errors = fit_correlation_scaling(N_values, correlation_values)

  print(f"Estimated η: {fit_params[0]} +/- {fit_errors[0]}")
