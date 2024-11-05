import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def random_herm_spectrum(N, seed, verb = False):
  
  """
  random_herm_spectrum:
    Generate a random Hermitian matrix and compute its eigenvalues and eigenvectors.

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
  spectrum = pd.DataFrame({'Eigenvalue' : eigenvalues, 'Eigenvectors' : [vec for vec in eigenvectors]})
  
  if verb==True:
    print(f'Random Hermitian Matrix:\n {A} \n')
    print(f'Eigenvalues with corresponding eigenvectors:\n {spectrum} \n')
  
  return eigenvalues, eigenvectors

def random_diag_spectrum(N, seed, verb = False):
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

def compute_norm_spacing(N, seed, mode = 'herm', trim = False):
  if mode == 'herm':
    eigval, _ = random_herm_spectrum(N, seed)
  elif mode == 'diag':
    eigval, _ = random_diag_spectrum(N, seed)
  else:
    None
  
  if trim:
    eigval = eigval[1:]  # Discard the first eigenvalue
  
  differences = np.diff(eigval)
  average = np.mean(differences) 
  spacings = differences / average
  return spacings

def compute_spacing_distr(N, seed, N_matrices, mode, trim = False):
  spacing_distr = []
  
  for i in range(N_matrices):
    seed = seed + N * i
    spacings = compute_norm_spacing(N, seed, mode=mode, trim=trim)
    spacing_distr.extend(spacings)
    
  return np.array(spacing_distr)

def fitting_function(s, a, alpha, b, beta):
  return a * (s**alpha) * np.exp(b * (s**beta))

def RMS(data, expected):
  return np.sqrt(np.mean((data - expected) ** 2))
  
def fitting_distribution(distr, bins=100, p0=[1, 1, -1, 1], verb=False):
  counts, bin_edges = np.histogram(distr, bins=bins, density=False)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  hist_density = counts / (np.sum(counts) * np.diff(bin_edges)[0])  # Normalize counts

  params, pcov = curve_fit(fitting_function, bin_centers, hist_density, p0=p0)
  expected = fitting_function(bin_centers, *params)
  rms_error = RMS(hist_density, expected)
  
  if verb:
    print(f'BEST PARAMETERS:\n a: {params[0]:.4f}, alpha: {params[1]:.4f}, b: {params[2]:.4f}, beta: {params[3]:.4f}\n')
    print(f'COVARIANCE MATRIX:\n {pcov}\n')
    print(f'Root Mean Squared Error: {rms_error:.4f}')

  return params, pcov, rms_error

def plot_fit(distr, bins=100, p0=[1, 1, -1, 1]):
  spacings_range = (np.min(distr), np.max(distr))
  s_vals = np.linspace(spacings_range[0], spacings_range[1], 1000)

  best_params, _ , _ = fitting_distribution(distr, bins, p0)
  plt.figure(figsize=(8, 6))
  plt.hist(distr, bins=bins, range=spacings_range, color='orange', edgecolor='orange', alpha=0.7, label='Normalized histogram', density=True)
  plt.plot(s_vals, fitting_function(s_vals, *best_params), 'r-', label='Fitted function', linewidth=1)
  plt.title('Histogram of spacing distribution')
  plt.xlabel('Spacing (s)')
  plt.ylabel('P(s)')
  plt.grid(visible=True, linestyle='--', alpha=0.5)
  plt.legend()
  plt.show()