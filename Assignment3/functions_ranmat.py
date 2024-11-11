###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 3.2 - EIGENPROBLEM
# Assignment 3.3 - RANDOM MATRIX THEORY


# IMPORT ZONE:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from scipy.optimize import curve_fit

# =========================================================

def random_herm_spectrum(N, seed, verb = False):
  """
  random_herm_spectrum:
    Generates a random Hermitian matrix and computes its eigenvalues and eigenvectors.

  Parameters
  ----------
  N : int
    Size of the Hermitian matrix (NxN).
  seed : int
    Random seed used to generate the matrix.
  verb : bool, optional
    If True, prints the Hermitian matrix and the eigenvalues/eigenvectors. By default False.

  Returns
  -------
  eigenvalues: np.ndarray
    Array of eigenvalues in ascending order.
  eigenvectors: np.ndarray
    Matrix of eigenvectors (each column is an eigenvector). Order based on the eigenvalues.
  """
  np.random.seed(seed)
  A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
  A = np.tril(A) + np.tril(A, -1).conj().T  # Make the matrix Hermitian
  eigenvalues, eigenvectors = np.linalg.eigh(A)  # 'eigh' is used for Hermitian matrices
  spectrum = pd.DataFrame({'Eigenvalue' : eigenvalues, 'Eigenvectors' : [vec for vec in eigenvectors.T]})
  
  if verb==True:
    print(f'Random Hermitian Matrix:\n {A} \n')
    print(f'Eigenvalues with corresponding eigenvectors:\n {spectrum} \n')
  
  return eigenvalues, eigenvectors

# =========================================================

def random_diag_spectrum(N, seed, verb = False):
  """
  random_diag_spectrum:
    Generates a random diagonal matrix and computes its eigenvalues and eigenvectors.
    
  Parameters
  ----------
  N : int
    Size of the diagonal matrix (NxN).
  seed : int
    Random seed used to generate the matrix.
  verb : bool, optional
    If True, prints the diagonal matrix and the eigenvalues/eigenvectors. By default False.

  Returns
  -------
  eigenvalues: np.ndarray
    Array of eigenvalues in ascending order.
  eigenvectors: np.ndarray
    Matrix of eigenvectors (each column is an eigenvector). Order based on the eigenvalues.
  """
  np.random.seed(seed)
  diag = np.random.randn(N)
  A = np.diag(diag)
  
  eigenvalues, eigenvectors = np.linalg.eig(A)
  sorted_indices = np.argsort(eigenvalues)
  eigenvalues = eigenvalues[sorted_indices]
  eigenvectors = eigenvectors[:, sorted_indices]

  spectrum = pd.DataFrame({'Eigenvalue' : eigenvalues, 'Eigenvectors' : [vec for vec in eigenvectors]})
  
  if verb:
    print(f'Random Diagonal Matrix (real):\n {A} \n')
    print(f'Eigenvalues with corresponding eigenvectors:\n {spectrum} \n')
  
  return eigenvalues, eigenvectors

# =========================================================

def compute_norm_spacing(N, seed, mode = 'herm', trim = False):
  """
  compute_norm_spacing:
    Computes normalized spacings between eigenvalues

  Parameters
  ----------
  N : _type_
    Size of the matrix.
  seed : _type_
    Random seed used to generate the matrix.
  mode : str, optional
    Type of random matrix to be generated. By default 'herm'.
  trim : bool, optional
    If True, removes the first eigenvalue. By default False.

  Returns
  -------
  spacings: np.ndarray
    Vector of consequent spacings.
  """
  if mode == 'herm':
    eigval, _ = random_herm_spectrum(N, seed)
  elif mode == 'diag':
    eigval, _ = random_diag_spectrum(N, seed)
  
  if trim:
    eigval = eigval[1:]  # Discard the first eigenvalue
  
  differences = np.diff(eigval)
  average = np.mean(differences) 
  spacings = differences / average
  return spacings

# =========================================================

def compute_spacing_distr(N, seed, N_matrices, mode, trim = False):
  """
  Computes the spacing distribution of normalized eigenvalues across multiple matrices.

  Parameters
  ----------
  N : int
    Size of each random matrix (NxN).
  seed : int
    Initial seed value for random number generation, which is incremented for each matrix to ensure different random states.
  N_matrices : int
    Number of random matrices to generate.
  mode : str
    Type of random matrix to create.
  trim : bool, optional
    If True, removes the first eigenvalue. By default False.

  Returns
  -------
  spacing_distr: np.ndarray
    Eigenvalue spacing distribution across all generated matrices.
  """
  spacing_distr = []
  
  for i in range(N_matrices):
    seed = seed + N * i
    spacings = compute_norm_spacing(N, seed, mode=mode, trim=trim)
    spacing_distr.extend(spacings)
  
  spacing_distr = np.array(spacing_distr)
  return spacing_distr

# =========================================================

def fitting_function(s, a, alpha, b, beta):
  """
  fitting_function:
    Fuction used for fitting spacing distributions.

  Parameters
  ----------
  s : np.ndarray
    Distribution of spacings.
  a : float
    Parameter of the function.
  alpha : float
    Parameter of the function.
  b : float
    Parameter of the function.
  beta : float
    Parameter of the function.
    
  Returns
  -------
  f: np.ndarray
    Vector of function's values over the spacing distribution given in input. 
  """
  f = a * (s**alpha) * np.exp(b * (s**beta))
  return f

# =========================================================

def RMSE(data, expected):
  """
  RMSE:
    Computes root mean squared error for the given data.

  Parameters
  ----------
  data : np.ndarray
    Real data.
  expected : np.ndarray
    Expected data. 

  Returns
  -------
  rms_error:
    Root mean squared error.
  """
  rms_error = np.sqrt(np.mean((data - expected) ** 2))
  return rms_error

# =========================================================
  
def fitting_distribution(distr, bins=100, p0=[1, 1, -1, 1], verb=False):
  """
  fitting_distribution:
    Fits a spacing distribution using the previously defined 'fitting function' and returns the best-fit parameters, the covariance matrix, and the root mean square error (RMSE) if requested.

  Parameters
  ----------
  distr : np.ndarray
    Spacing distribution to be fitted.
  bins : int, optional
    Number of bins of the histogram. By default 100.
  p0 : list, optional
    Inital parameters of the fit. By default [1, 1, -1, 1].
  verb : bool, optional
    If True, prints the best-fit parameters, the covariance matrix and the RMSE. By default False.

  Returns
  -------
  params : np.ndarray
    Array of best-fit parameters [a, alpha, b, beta] from the fitting function.
  pcov : np.ndarray
    Covariance matrix of the parameter estimates.
  rms_error : float
    Root Mean Squared Error of the fit.
  """
  counts, bin_edges = np.histogram(distr, bins=bins, density=False)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  counts = counts / sum(counts)

  params, pcov = curve_fit(fitting_function, bin_centers, counts, p0=p0)
  expected = fitting_function(bin_centers, *params)
  rms_error = RMSE(counts, expected)
  
  if verb:
    print(f'BEST PARAMETERS:\n a: {params[0]:.4f}, alpha: {params[1]:.4f}, b: {params[2]:.4f}, beta: {params[3]:.4f}\n')
    print(f'COVARIANCE MATRIX:\n {pcov}\n')
    print(f'Root Mean Squared Error: {rms_error:.4f}')

  return params, pcov, rms_error

# =========================================================

def plot_fit(distr, bins=100, p0=[1, 1, -1, 1]):
  """
  plot_fit:
    Plots the histogram of a spacing distribution along with the fitted function.

  Parameters
  ----------
  distr : np.ndarray
    Array of spacing values to be analyzed.
  bins : int, optional
    Number of bins for the histogram. By default 100.
  p0 : list, optional
    Initial parameters for fitting the distribution. By default [1, 1, -1, 1].

  Returns
  -------
  None
  """
  spacings_range = (np.min(distr), np.max(distr))
  s_vals = np.linspace(spacings_range[0], spacings_range[1], 1000)

  best_params, _ , _ = fitting_distribution(distr, bins, p0)
  
  counts, bin_edges = np.histogram(distr, bins=bins, range=spacings_range)
  normalized_counts = counts / counts.sum()  # Normalize by the sum of counts
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  
  plt.figure(figsize=(8, 6))
  plt.bar(bin_centers, normalized_counts, width=(bin_edges[1] - bin_edges[0]), color='orange', edgecolor='orange', alpha=0.7, label='Normalized histogram')
  plt.plot(s_vals, fitting_function(s_vals, *best_params), 'r-', label='Fitted function', linewidth=1)
  plt.title('Histogram of spacing distribution')
  plt.xlabel('Spacing (s)')
  plt.ylabel('P(s)')
  plt.grid(visible=True, linestyle='--', alpha=0.5)
  plt.legend()
  plt.show()
  
# =========================================================