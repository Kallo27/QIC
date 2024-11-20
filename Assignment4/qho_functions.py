###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 4 - QUANTUM HARMONIC OSCILLATOR


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import debugger as db
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.linalg import eigh
from scipy.special import factorial
from scipy.stats import linregress


# ===========================================================================================================
# FINITE DIFFERENCE METHOD
# ===========================================================================================================

def kinetic_matrix(L, N=1000, order=2):
  """
  kinetic_matrix: 
    Computes the kinetic energy matrix for the finite difference method.

  Parameters
  ----------
  L : float
    Half-width of the spatial domain.
  N : int, optional
    Number of discretization points. Default is 1000.
  order : int, optional
    Order of the finite difference approximation (2, 4, 6, 8). Default is 2.

  Returns
  -------
  K : np.ndarray
    The kinetic energy matrix.
  """
  # Constants
  hbar = 1.0  # Reduced Planck constant (set to 1 in atomic units)
  m = 1.0     # Mass of the particle (set to 1 in atomic units)
  
  # Grid and factors
  dx = 2 * L / N
  factor = hbar**2 / (m * dx**2)
  K = np.zeros((N, N))
  
  if order == 2:
    K += np.diag(factor * np.ones(N))  # Main diagonal
    K += np.diag(-factor / 2 * np.ones(N-1), k=1)  # Upper diagonal
    K += np.diag(-factor / 2 * np.ones(N-1), k=-1)  # Lower diagonal
    
  elif order == 4:
    K += np.diag(5 * factor / 4 * np.ones(N))  # Main diagonal
    K += np.diag(-2 * factor / 3 * np.ones(N-1), k=1)  # First upper diagonal
    K += np.diag(-2 * factor / 3 * np.ones(N-1), k=-1)  # First lower diagonal
    K += np.diag(factor / 24 * np.ones(N-2), k=2)  # Second upper diagonal
    K += np.diag(factor / 24 * np.ones(N-2), k=-2)  # Second lower diagonal
    
  elif order == 6:
    K += np.diag(49 * factor / 36 * np.ones(N))  # Main diagonal
    K += np.diag(-3 * factor / 4 * np.ones(N-1), k=1)  # First upper diagonal
    K += np.diag(-3 * factor / 4 * np.ones(N-1), k=-1)  # First lower diagonal
    K += np.diag(3 * factor / 40 * np.ones(N-2), k=2)  # Second upper diagonal
    K += np.diag(3 * factor / 40 * np.ones(N-2), k=-2)  # Second lower diagonal
    K += np.diag(-factor / 180 * np.ones(N-3), k=3)  # Third upper diagonal
    K += np.diag(-factor / 180 * np.ones(N-3), k=-3)  # Third lower diagonal
    
  elif order == 8:
    K += np.diag(205 * factor / 144 * np.ones(N))  # Main diagonal
    K += np.diag(-4 * factor / 5 * np.ones(N-1), k=1)  # First upper diagonal
    K += np.diag(-4 * factor / 5 * np.ones(N-1), k=-1)  # First lower diagonal
    K += np.diag(1 * factor / 10 * np.ones(N-2), k=2)  # Second upper diagonal
    K += np.diag(1 * factor / 10 * np.ones(N-2), k=-2)  # Second lower diagonal
    K += np.diag(-4 * factor / 315 * np.ones(N-3), k=3)  # Third upper diagonal
    K += np.diag(-4 * factor / 315 * np.ones(N-3), k=-3)  # Third lower diagonal
    K += np.diag(factor / 1120 * np.ones(N-4), k=4)  # Fourth lower diagonal
    K += np.diag(factor / 1120 * np.ones(N-4), k=-4)  # Fourth lower diagonal
    
  else:
    db.checkpoint(debug=True, msg1="APPROXIMATION ORDER", msg2="Unsupported order. Please choose order = 2, 4, 6 or 8.", stop=True)
  
  return K

# ===========================================================================================================

def hamiltonian(omega, L, N=1000, order=2):
  """
  hamiltonian: 
    Constructs the Hamiltonian matrix for the harmonic oscillator
    (using the finite difference numerical method).

  Parameters
  ----------
  omega : float
    Angular frequency of the oscillator.
  L : float
    Half-width of the spatial domain.
  N : int, optional
    Number of grid points. Default is 1000.
  order : int, optional
    Order of finite difference approximation. Default is 2.

  Returns
  -------
  H : np.ndarray
    The Hamiltonian matrix.
  """
  # Constants
  m = 1.0 # Mass of the particle (set to 1 in atomic units)
  
  # Discretization
  dx = 2 * L / N
  x = np.linspace(-L, L, N) + dx / 2

  # Construct the Hamiltonian matrix
  K = kinetic_matrix(L, N, order)
  
  V_diag = 0.5 * m * omega**2 * x**2
  V = np.diag(V_diag)

  H = K + V
  return H

# ===========================================================================================================

def harmonic_oscillator_spectrum(omega, L, N=1000, order=2):
  """
  harmonic_oscillator_spectrum:
    Computes the eigenvalues and eigenfunctions of the harmonic oscillator
    (using the finite difference numerical method).

  Parameters
  ----------
  omega : float
    Angular frequency of the oscillator.
  L : float
    Half-width of the spatial domain.
  N : int, optional
    Number of grid points. Default is 1000.
  order : int, optional
    Order of finite difference approximation. Default is 2.

  Returns
  -------
  energies, psi : tuple
    Energies and normalized wavefunctions.
  """
  # Eigenvalues and eigenfunctions computation
  H = hamiltonian(omega, L, N, order)
  energies, psi = eigh(H)
  
  # Eigenfunctions flipping (for consistence with analytical solutions)
  center_index = N // 2

  for i in range(len(psi)):
    if i % 2 == 0:
      # Ensure the wavefunction is positive at the center
      if ((i//2)%2==0 and psi[:, i][center_index] < 0) or ((i//2)%2!=0 and psi[:, i][center_index] > 0):
        psi[:, i] *= -1
    else :
      # Find the first peak after the center
      for j in range(center_index, len(psi[:, i]) - 1):
        if abs(psi[:, i][j]) > abs(psi[:, i][j + 1]):  # First peak condition
          first_peak_index = j
          break
      # Adjust sign based on desired pattern
      if (i % 4 == 1 and psi[:, i][first_peak_index] < 0):  # Positive peaks
        psi[:, i] *= -1
      elif (i % 4 == 3 and psi[:, i][first_peak_index] > 0):  # Negative peaks
        psi[:, i] *= -1
  
  # Normalization and transposition
  dx = 2 * L / N
  psi = psi.T / np.sqrt(np.sum(np.abs(psi.T)**2, axis = 0) * dx)  
  return energies, psi


# ===========================================================================================================
# ANALYTICAL SOLUTION
# ===========================================================================================================

def hermite(x, n):
  """
  hermite:
    Hermite polinomial of order 'n', 
    defined over the real space grid 'x'.

  Parameters
  ----------
  x : np.ndarray
    Real space grid.
  n : int
    Order of the polinomial.

  Returns
  -------
  herm_pol: np.ndarray
    Hermite polinomial of order 'n'.
  """
  # Pre-condition: n>=0
  if n < 0:
    db.checkpoint(debug=True, msg=f"The order of the Hermite polynomial is not valid (n={n}, expected n>=0)", stop=True)

  # Coefficients set to 0 except for the one of order n.
  herm_coeffs = np.zeros(n + 1)
  herm_coeffs[n] = 1
  
  # Actual computation of the polinomial over the space grid.
  herm_pol = np.polynomial.hermite.hermval(x, herm_coeffs)
  return herm_pol

# ===========================================================================================================

def harmonic_en(omega=1.0, n=0):
  """
  harmonic_en:
    Energy levels for an harmonic potential.

  Parameters
  ----------
  omega : float, optional
    Angular frequency of the harmonic potential. Default is 1.0.
  n : int, optional
    Energy level. Default is 0.

  Returns
  -------
  energy: float
    Energy of level 'n'.
  """
  # Constants set to 1 in atomic units.
  hbar = 1.0
  
  # Pre-condition: n>=0
  if n < 0:
    db.checkpoint(debug=True, msg=f"The order of the energy level is not valid (n={n}, expected n>=0)", stop=True)
    
  # Complete wavefunction.
  energy = hbar * omega * (n + 1/2)
  return energy

# ===========================================================================================================

def harmonic_wfc(omega, L, N, n=0):
  """
  harmonic_wfc:
    Wavefunction of order 'n' for a harmonic potential, 
    defined over the real space grid 'x'.
  
    V(x) = 0.5 * m * omega * x**2
        
  Parameters
  ----------
  omega: float, optional
    Angular frequency of the harmonic potential.
  L : float
    Half-width of the spatial domain.
  N : int, optional
    Number of grid points. Default is 1000.
  n : int, optional
    Order of the wavefunction. By default 0 (ground state).

  Returns
  -------
  psi: np.ndarray
    Normalized wavefunction of order 'n'.
  """
  # Constants set to 1 in atomic units.
  hbar = 1.0
  m = 1.0
  
  # Grid
  dx = 2 * L / N
  x = np.linspace(-L, L, N) + dx / 2
  
  # Components of the analytical solution for stationary states.
  prefactor = 1 / np.sqrt(2**n * factorial(n)) * ((m * omega) / (np.pi * hbar))**0.25
  x_coeff = np.sqrt(m * omega / hbar)
  exponential = np.exp(- (m * omega * x**2) / (2 * hbar))
  
  # Complete wavefunction.
  psi = prefactor * exponential * hermite(x_coeff * x, n)
  
  # Normalization condition.
  psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)
  print()
  return psi_normalized
  

# ===========================================================================================================
# PLOTTING FUNCTIONS FOR ENERGY LEVELS AND WAVEFUNCTIONS
# ===========================================================================================================

def generate_colors(n):
  """
  generate_colors: 
    Generates a colormap with n distinct colors.

  Parameters
  ----------
  n : int
    Number of distinct colors to generate.

  Returns
  -------
  cmap : list of tuple
    List of RGB tuples representing the generated colormap.
  """
  palette = sns.color_palette("husl", n)
  cmap = [tuple(color) for color in palette]
  return cmap

# ===========================================================================================================

def plot_wf_en(omega, L, N=1000, k=10, order=None):
  """
  plot_wf_en: 
    Plots wavefunctions and energy levels of the quantum harmonic oscillator.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points. Default is 1000.
  k : int
    Number of energy levels and wavefunctions to plot. Default is 10
  order : int, optional
    Order of finite difference approximation (if None, uses analytical results).
    Default is None.

  Returns
  -------
  None
  """
  # Grid
  dx = 2 * L / N
  x = np.linspace(-L, L, N) + dx / 2
  
  # Compute energy levels and wavefunctions
  if order is not None:
    energies, wavefunctions = harmonic_oscillator_spectrum(omega, L, N, order)
  else:
    energies = [harmonic_en(omega, k) for k in range(0, k)]
    wavefunctions = [harmonic_wfc(omega, L, N, k) for k in range(0, k)]

  # Build subplots and generate cmap.
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
  colors = generate_colors(k)
  
  # Plot the wavefunctions in the first subplot
  for n in range(k):
    ax1.plot(x, wavefunctions[n], label=f"$\psi(x)$ of order {n}", color=colors[n], linewidth=1)
  
  ax1.set_xlabel("Position $x$")
  ax1.set_ylabel("Amplitude")
  ax1.set_title("Wavefunctions of quantum harmonic oscillator")
  ax1.grid(True, linestyle='--', alpha=0.7)
  ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

  # Plot the potential and energy levels in the second subplot
  ax2.plot(x, 0.5 * omega**2 * x**2, label="Harmonic potential $V(x)$", color="red", linestyle="--", linewidth=1.5)
  for n in range(k):
    ax2.axhline(energies[n], label=f"Energy of order {n}", color=colors[n], linestyle="-.", linewidth=1, alpha=0.8)
   
  ax2.set_xlabel("Position $x$")
  ax2.set_ylabel("Energy")
  ax2.set_title("Energy levels and harmonic potential")
  ax2.grid(True, linestyle='--', alpha=0.7)
  ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

  # Show
  fig.tight_layout()
  plt.show()


# ===========================================================================================================
# CORRECTNESS FUNCTIONS
# ===========================================================================================================

def check_schroedinger(omega, L, N, k, orders):
  """
  check_schroedinger:
    Checks how well the computed wavefunctions and energies satisfy
    the Schrödinger equation for the harmonic oscillator.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenstates to check.
  orders : list of int
    List of finite difference orders to evaluate.

  Returns
  -------
  diff : ndarray of shape (len(orders), k)
    Differences between computed and expected results for each order and eigenstate.
  """
  # Initialize 'diff'
  diff = np.ndarray((len(orders), k))
  dx = 2 * L / N
  
  # Loop through the orders
  for i, order in enumerate(orders):
    # Build Hamiltonian and compute energy levels and eigenfunctions 
    H = hamiltonian(omega, L, N, order)
    energies, psi = harmonic_oscillator_spectrum(omega, L, N, order)
    
    # Compute difference between energy and expected value
    for j in range(k):
      lhs = np.vdot(psi[j], np.dot(H, psi[j])) * dx
      diff[i, j] = np.abs(lhs - energies[j])   
    
  return diff

# ===========================================================================================================

def energy_difference(omega, L, N, k, orders, rel=True):
  """
  energy_difference:
    C the absolute or relative differences between analytical
    and computed eigenvalues for the harmonic oscillator.

  Parameters
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenvalues to compare.
  orders : list of int
    List of finite difference orders to evaluate.
  rel : bool, optional
    If True, compute relative differences. Default is True.

  Returns
  -------
  differences : ndarray of shape (len(orders), k)
    Energy differences for each order and eigenvalue.
  """
  # Initialize 'differences'
  differences = np.ndarray((len(orders), k))
  
  # Compute expected energies
  an_en = [harmonic_en(omega, k) for k in range(0, k)]
  
  # Loop through the orders
  for i, order in enumerate(orders):
    # Compute numerical energies
    comp_en, _ = harmonic_oscillator_spectrum(omega, L, N, order)
    
    # WARNING: if k is higher than the available energies, pad energies with 'np.nan'.
    if len(comp_en) < k:
      db.checkpoint(debug= True, msg1=f"Warning: Number of computed energies is less than {k}. Using available energies.", stop=False)
      comp_en = np.pad(comp_en, (0, k - len(comp_en)), constant_values=np.nan)
    
    # Compute absolute difference (and if rel is True, relative)    
    differences[i] = np.abs(comp_en[:k] - an_en)
    if rel:
      differences[i] /= comp_en[:k]
  return differences
  
# ===========================================================================================================

def wfc_difference(omega, L, N, k, orders):
  """
  wfc_difference:
    Computes the differences between analytical and computed eigenstates
    for the harmonic oscillator using the dot product.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenstates to compare.
  orders : list of int
    List of finite difference orders to evaluate.

  Returns
  -------
  differences : ndarray of shape (len(orders), k)
    Differences between analytical and computed wavefunctions for each order.
  """
  # Grid
  dx = 2 * L / N
  x = np.linspace(-L, L, N) + dx / 2
  
  # Initialize 'difference'
  differences = np.zeros((len(orders), k))
  
  # Loop through the orders
  for i, order in enumerate(orders):
    _, comp_wfc = harmonic_oscillator_spectrum(omega, L, N, order)
    
    # WARNING: if k is higher than the available energies, pad energies with zeros.
    if comp_wfc.shape[0] < k:
      db.checkpoint(debug=True, msg1=f"Warning: Number of computed eigenstates is less than {k}. Using available eigenstates.", stop=False)
      padding = np.zeros((k - comp_wfc.shape[0], N))
      comp_wfc = np.vstack([comp_wfc, padding])
    
    # Compute 1 - |dot product|
    for j in range(k):
      an_wfc = harmonic_wfc(omega, L, N, j)
      differences[i, j] = 1 - np.abs((np.dot(comp_wfc[j], an_wfc)) * dx) 
  return differences

# ===========================================================================================================

def plot_schroedinger(omega, L, N, k, orders):
  """
  plot_schroedinger:
    Generates a heatmap of the differences in how well computed
    wavefunctions and energies satisfy the Schrödinger equation.

  Parameters
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenvalues to compare.
  orders : list of int
    List of finite difference orders to evaluate.

  Returns
  -------
  None
  """
  plt.figure(figsize=(8, 4))
  
  differences = check_schroedinger(omega, L, N, k, orders)

  step = max(1, k // 10)  # Ensure at most 10 labels
  xtickslab = [f"$E_{{{i}}}$" if i % step == 0 else "" for i in range(k)]
  sns.heatmap(differences, cmap='cividis', xticklabels=xtickslab, 
              yticklabels=[f"Order {order}" for order in orders], cbar_kws={'label': 'Difference (expected 0)'})
  
  plt.title("Schroedinger equation check for computational eigenstates")
  plt.xlabel("Eigenstate index")
  plt.ylabel("Order")
  plt.tight_layout()
  plt.show()

# ===========================================================================================================

def plot_energy_orders(omega, L, N, k, orders):
  """
  plot_energy_orders:
    Generates a heatmap of energy differences between analytical
    and computed eigenvalues for different orders.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenvalues to compare.
  orders : list of int
    List of finite difference orders to evaluate.
  rel : bool, optional
    If True, compute relative differences. Default is False.

  Returns
  -------
  differences : ndarray of shape (len(orders), k)
    Energy differences for each order and eigenvalue.
  """
  plt.figure(figsize=(8, 4))
  
  differences = energy_difference(omega, L, N, k, orders)
  
  step = max(1, k // 10)  # Ensure at most 10 labels
  xtickslab = [f"$E_{{{i}}}$" if i % step == 0 else "" for i in range(k)]
  sns.heatmap(differences, cmap='cividis', xticklabels=xtickslab, 
              yticklabels=[f"Order {order}" for order in orders], cbar_kws={'label': 'Relative energy difference'})
  
  plt.title("Energy difference for different eigenvalues at different orders")
  plt.xlabel("Eigenvalue index")
  plt.ylabel("Order")
  plt.tight_layout()
  plt.show()

# ===========================================================================================================

def plot_wfc_orders(omega, L, N, k, orders):
  """
  plot_wfc_orders:
    Generates a heatmap of differences between analytical and computed
    eigenstates for different orders of finite differences.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenvalues to compare.
  orders : list of int
    List of finite difference orders to evaluate.

  Returns
  -------
  None
  """
  plt.figure(figsize=(8, 4))
  
  differences = wfc_difference(omega, L, N, k, orders)

  step = max(1, k // 10)  # Ensure at most 10 labels
  xtickslab = [f"$E_{{{i}}}$" if i % step == 0 else "" for i in range(k)]
  sns.heatmap(differences, cmap='cividis', xticklabels=xtickslab, 
              yticklabels=[f"Order {order}" for order in orders], cbar_kws={'label': 'Difference (expected 0)'})
  
  plt.title("Dot product between expected and computed eigenstates")
  plt.xlabel("Eigenstate index")
  plt.ylabel("Order")
  plt.tight_layout()
  plt.show()

# ===========================================================================================================

def plot_loglog_fit(omega, L, N, k, order):
  """
  plot_loglog_fit:
    Performs a log-log regression on the energy differences for a
    specific order and plots the data with the regression fit.

  Parameters
  ----------
  omega : float
    Angular frequency of the harmonic oscillator.
  L : float
    Half-width of the spatial domain.
  N : int
    Number of grid points.
  k : int
    Number of eigenvalues to compare.
  order : int
    Finite difference order to evaluate.

  Returns
  -------
  None
  """
  # Calculate energy differences for the given N and order
  differences = energy_difference(omega, L, N, k, [order], rel=False).flatten()

  # Generate eigenvalue indices
  eigen_indices = np.arange(1, k + 1)

  # Perform linear regression in log-log space
  log_x = np.log(eigen_indices)
  log_y = np.log(differences)
  slope, intercept, r_value, _, _ = linregress(log_x, log_y)

  # Predicted values from the fit
  fitted_values = np.e ** (intercept + slope * log_x)

  # Plot the data and the fit
  plt.figure(figsize=(8, 6))
  plt.loglog(eigen_indices, differences, marker='o', label=f"Order {order}")
  plt.loglog(eigen_indices, fitted_values, linestyle='--', label=f"Fit: $y \propto k^{{{slope:.2f}}}$")
  plt.title(f"Energy differences for order {order} in log-log scale")
  plt.xlabel("Eigenvalue index")
  plt.ylabel("Energy difference")
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.legend(loc="best")
  plt.tight_layout()
  plt.show()

  # Print the slope and intercept
  print(f"Fit parameters:\nIntercept = {intercept:.2f}, Slope = {slope:.2f}, $R^2$ = {r_value**2:.3f}")


# ===========================================================================================================
# STABILITY FUNCTIONS
# ===========================================================================================================

def check_stability(omega, L, N, k, order, num_runs):
  eigenvals_runs = []
  eigenvecs_runs = []
  
  for _ in range(num_runs):
    eigenvalues, eigenvectors = harmonic_oscillator_spectrum(omega, L, N, order)  # Adjust parameters as needed
    eigenvals_runs.append(eigenvalues[:k])          # Store the first k eigenvalues
    eigenvecs_runs.append(eigenvectors[:k])     # Store the first k eigenvectors
  
  eigenvals_runs = np.array(eigenvals_runs)
  eigenvecs_runs = np.array(eigenvecs_runs)
  
  check_eigen = eigenvals_runs[-1]
  
  eigenvalues_mean = np.mean(eigenvals_runs, axis=0)
  eigenvalues_std = np.std(eigenvals_runs, axis=0)
  dot_matrix = np.zeros((k, num_runs - 1))
  
  for i in range(k):
    for j in range(1, num_runs):
      dot_product = np.dot(eigenvecs_runs[j, i, :], eigenvecs_runs[j - 1, i, :])
      dot_matrix[i, j - 1] = np.abs(1 - np.abs(dot_product))
      
  eigvec_dot_mean = np.mean(dot_matrix, axis=1)
  mean_relative_error = eigenvalues_mean - check_eigen
  
  return mean_relative_error, eigenvalues_std, eigvec_dot_mean, dot_matrix

# ===========================================================================================================

def plot_stability(omega, L, N, k, orders, num_runs):
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))

  # Initialize lists for data storage
  eigenvalues_means = []
  eigenvalues_stds = []
  eigvec_dot_means = []
  dot_matrices = []

  # Run stability check for each order and collect results
  for order in orders:
    eigenvalues_mean, eigenvalues_std, eigvec_dot_mean, dot_matrix = check_stability(omega, L, N, k, order, num_runs)
    eigenvalues_means.append(eigenvalues_mean)
    eigenvalues_stds.append(eigenvalues_std)
    eigvec_dot_means.append(eigvec_dot_mean)
    dot_matrices.append(dot_matrix)

  # Convert collected data into arrays for easier plotting
  eigenvalues_means = np.array(eigenvalues_means)
  eigenvalues_stds = np.array(eigenvalues_stds)
  eigvec_dot_means = np.array(eigvec_dot_means)

  # Top left: Mean of eigenvalues
  ax = axes[0, 0]
  sns.heatmap(eigenvalues_means, fmt=".3f", ax=ax, cmap="viridis", xticklabels=range(k), yticklabels=orders)
  ax.set_title("Eigenvalues mean")
  ax.set_xlabel("Eigenvalue index")
  ax.set_ylabel("Order")

  # Top right: Standard deviation of eigenvalues
  ax = axes[0, 1]
  sns.heatmap(eigenvalues_stds, fmt=".3f", ax=ax, cmap="viridis", xticklabels=range(k), yticklabels=orders)
  ax.set_title("Eigenvalues standard deviation")
  ax.set_xlabel("Eigenvalue index")
  ax.set_ylabel("Order")

  # Bottom left: Mean deviation of eigenvector dot products
  ax = axes[1, 0]
  sns.heatmap(eigvec_dot_means, fmt=".3f", ax=ax, cmap="viridis", xticklabels=range(k), yticklabels=orders)
  ax.set_title("Eigenvectors dot mean deviation")
  ax.set_xlabel("Eigenvector index")
  ax.set_ylabel("Order")

  # Bottom right: Heatmap of dot matrix for the last order
  ax = axes[1, 1]
  sns.heatmap(dot_matrices[-1], ax=ax, cmap="viridis", cbar_kws={'label': 'Deviation'})
  ax.set_title(f"Eigenvector dot matrix (Order {orders[-1]})")
  ax.set_xlabel("Run index")
  ax.set_ylabel("Eigenvector index")

  # Adjust layout and show plot
  plt.tight_layout()
  plt.show()

# ===========================================================================================================
# DISCRETIZATION FUNCTIONS
# ===========================================================================================================

def check_discretization(omega, L, k, orders, N_values):
  # Initialize storage
  eigenvalue_errors = np.zeros((len(N_values), len(orders), k))
  eigenfunction_errors = np.zeros((len(N_values), len(orders), k))

  # Loop over discretization steps and orders
  for i, N in enumerate(N_values):
    ev_errors = energy_difference(omega, L, N, k, orders)  # Shape: len(orders) x k
    eigenvalue_errors[i, :, :] = ev_errors  # Store for current N and all orders

    ef_errors = wfc_difference(omega, L, N, k, orders)  # Shape: len(orders) x k
    eigenfunction_errors[i, :, :] = ef_errors  # Store for current N and all orders

  return eigenvalue_errors, eigenfunction_errors

# ===========================================================================================================

def plot_discretization_heatmaps(omega, L, k, orders, N_values):
  import seaborn as sns

  # Compute dx values
  dx_values = [2 * L / N for N in N_values]

  # Get errors
  eigenvalue_errors, eigenfunction_errors = check_discretization(omega, L, k, orders, N_values)

  # Create subplots for heatmaps
  fig, axes = plt.subplots(2, 3, figsize=(18, 12))

  # Prepare and plot eigenvalue errors (top row)
  for i, order in enumerate(orders):
    # Extract the data for the specific order
    data = eigenvalue_errors[:, i, :]  # Shape: (N_values, k)

    sns.heatmap(
      data.T,  # Transpose to put dx on x-axis and k on y-axis
      ax=axes[0, i],
      xticklabels=np.round(dx_values, 3),
      yticklabels=range(1, k + 1),  # k values
      cmap="cividis",
      annot=False,
      cbar=True,
      fmt=".2e"
    )
    axes[0, i].set_title(f"Eigenvalue Errors (Order {order})")
    axes[0, i].set_xlabel("Discretization Step (dx)")
    axes[0, i].set_ylabel("k (Eigenvalue Index)")

  # Prepare and plot eigenfunction errors (bottom row)
  for i, order in enumerate(orders):
    # Extract the data for the specific order
    data = eigenfunction_errors[:, i, :]  # Shape: (N_values, k)

    sns.heatmap(
      data.T,  # Transpose to put dx on x-axis and k on y-axis
      ax=axes[1, i],
      xticklabels=np.round(dx_values, 3),
      yticklabels=range(1, k + 1),  # k values
      cmap="plasma",
      annot=False,
      cbar=True,
      fmt=".2e"
    )
    axes[1, i].set_title(f"Eigenfunction Errors (Order {order})")
    axes[1, i].set_xlabel("Discretization Step (dx)")
    axes[1, i].set_ylabel("k (Eigenfunction Index)")

  # Adjust layout
  plt.tight_layout()
  plt.show()

# ===========================================================================================================
# EFFICIENCY FUNCTIONS
# ===========================================================================================================

def measure_efficiency(omega, L, orders, N_values, repetitions=1):
  times = np.zeros((len(N_values), len(orders), repetitions))

  for i, N in enumerate(N_values):
    for j, order in enumerate(orders):
      for r in range(repetitions):
        start_time = time.perf_counter()
                
        harmonic_oscillator_spectrum(omega, L, N, order)
                
        end_time = time.perf_counter()
                
        times[i, j, r] = end_time - start_time

  return times

# ===========================================================================================================

def plot_efficiency_heatmap(omega, L, orders, N_values, repetitions=1):
  # Average times over repetitions if applicable
  times = measure_efficiency(omega, L, orders,  N_values, repetitions)
  
  if repetitions > 1:
      avg_times = np.mean(times, axis=2)
  else:
      avg_times = times[:, :, 0]
  
  # Convert grid sizes to dx values for better representation
  dx_values = [2 * L / N for N in N_values]
  
  # Create the heatmap
  plt.figure(figsize=(10, 6))
  sns.heatmap(avg_times, xticklabels=orders, yticklabels=np.round(dx_values, 3), cmap="viridis", fmt=".2e", 
              cbar_kws={"label": "Time (seconds)"})
  plt.xlabel("Finite Difference Order")
  plt.ylabel("Discretization Step (dx)")
  plt.title("Computation Time Heatmap")
  plt.show()
  
# ===========================================================================================================