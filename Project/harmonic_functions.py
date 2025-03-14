# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np

from scipy.linalg import eigh
from scipy.special import factorial
from scipy.sparse import diags


# ===========================================================================================================
# FINITE DIFFERENCE METHOD
# ===========================================================================================================

def kinetic_matrix(x, order=2):
  """
  kinetic_matrix:
    Computes the kinetic energy matrix for the finite difference method using sparse matrices.

  Parameters
  ----------
  x : np.ndarray
    Real space grid.
  order : int, optional
    Order of the finite difference approximation (2, 4, 6, 8). Default is 2.

  Returns
  -------
  K : scipy.sparse.csr_matrix
    The kinetic energy matrix in sparse format.
  """
  # Grid params
  dx = x[1] - x[0]
  factor = 1 / (dx**2)
  N = len(x)
  
  # Define diagonals and offsets for different finite difference orders
  if order == 2:
    diagonals = [
      factor * np.ones(N),                 # Main diagonal
      -factor / 2 * np.ones(N - 1),        # First upper diagonal
      -factor / 2 * np.ones(N - 1),        # First lower diagonal
    ]
    offsets = [0, 1, -1]
  
  elif order == 4:
    diagonals = [
      5 * factor / 4 * np.ones(N),         # Main diagonal
      -2 * factor / 3 * np.ones(N - 1),    # First upper diagonal
      -2 * factor / 3 * np.ones(N - 1),    # First lower diagonal
      factor / 24 * np.ones(N - 2),        # Second upper diagonal
      factor / 24 * np.ones(N - 2),        # Second lower diagonal
    ]
    offsets = [0, 1, -1, 2, -2]
  
  elif order == 6:
    diagonals = [
      49 * factor / 36 * np.ones(N),       # Main diagonal
      -3 * factor / 4 * np.ones(N - 1),    # First upper diagonal
      -3 * factor / 4 * np.ones(N - 1),    # First lower diagonal
      3 * factor / 40 * np.ones(N - 2),    # Second upper diagonal
      3 * factor / 40 * np.ones(N - 2),    # Second lower diagonal
      -factor / 180 * np.ones(N - 3),      # Third upper diagonal
      -factor / 180 * np.ones(N - 3),      # Third lower diagonal
    ]
    offsets = [0, 1, -1, 2, -2, 3, -3]
  
  elif order == 8:
    diagonals = [
      205 * factor / 144 * np.ones(N),     # Main diagonal
      -4 * factor / 5 * np.ones(N - 1),    # First upper diagonal
      -4 * factor / 5 * np.ones(N - 1),    # First lower diagonal
      factor / 10 * np.ones(N - 2),        # Second upper diagonal
      factor / 10 * np.ones(N - 2),        # Second lower diagonal
      -4 * factor / 315 * np.ones(N - 3),  # Third upper diagonal
      -4 * factor / 315 * np.ones(N - 3),  # Third lower diagonal
      factor / 1120 * np.ones(N - 4),      # Fourth upper diagonal
      factor / 1120 * np.ones(N - 4),      # Fourth lower diagonal
    ]
    offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4]
  
  else:
    raise ValueError(f"Unsupported order (order = {order}). \nPlease choose order = 2, 4, 6, or 8.")
  
  # Create the sparse matrix using scipy.sparse.diags
  K = diags(diagonals, offsets, format="csr")
  
  return K

# ===========================================================================================================

def hamiltonian(x, omega, order=2):
  """
  hamiltonian: 
    Constructs the Hamiltonian matrix for the harmonic oscillator
    (using the finite difference numerical method).

  Parameters
  ----------
  x : np.ndarray
    Real space grid.
  omega : float
    Angular frequency of the harmonic oscillator.
  order : int, optional
    Order of finite difference approximation. Default is 2.

  Returns
  -------
  H : np.ndarray
    The Hamiltonian matrix.
  """  
  # Construct the Hamiltonian matrix
  K = kinetic_matrix(x, order)
  
  V_diag = 0.5 * omega **2  * x**2
  V = np.diag(V_diag)

  H = K + V
  return H

# ===========================================================================================================

def harmonic_oscillator_spectrum(x, omega, order=2, n_max=1):
  """
  harmonic_oscillator_spectrum:
    Computes the eigenvalues and eigenfunctions of the harmonic oscillator
    (using the finite difference numerical method).

  Parameters
  ----------
  x : np.ndarray
    Real space grid.
  omega : float
    Angular frequency of the harmonic oscillator.
  order : int, optional
    Order of finite difference approximation. Default is 2.
  n_max : int, optional
    Number of wavefunctions to return. Default is 1 (ground state).

  Returns
  -------
  psi : np.ndarray
    First n_max normalized wavefunctions.
  """
  # Eigenvalues and eigenfunctions computation
  H = hamiltonian(x, omega, order)
  _, psi = eigh(H)
  
  # Eigenfunctions flipping (for consistence with analytical solutions)
  center_index = len(x) // 2

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
  psi = psi.T / np.sqrt(np.sum(np.abs(psi.T)**2, axis = 0))
  psi = psi[:n_max].astype(complex)
  
  return psi


# ===========================================================================================================
# EXACT DIAGONALIZATION METHOD
# ===========================================================================================================

def hermite(x, n):
  """
  hermite :
    Computes the Hermite polynomial of order 'n', 
    defined over the real space grid 'x'.

  Parameters
  ----------
  x : np.ndarray
    Real space grid.
  n : int
    Order of the polynomial.

  Returns
  -------
  pol = np.ndarray
    Hermite polynomial of order 'n'
  """
  if n < 0:
    raise ValueError(f"Invalid order of Hermite polynomial: n={n}, expected n>=0")
  
  herm_coeffs = np.zeros(n + 1)
  herm_coeffs[n] = 1
  pol = np.polynomial.hermite.hermval(x, herm_coeffs)
  
  return pol

# ===========================================================================================================

def harmonic_wfc(x, omega, n_max=1):
  """
  harmonic_wfc:
    Computes the first 'n_max' wavefunctions for an harmonic potential, 
    defined over the real space grid 'x'.
  
    V(x) = 0.5 * omega * x**2
        
  Parameters
  ----------
  x : np.ndarray
    Spatial grid used for discretization.
  omega : float
    Angular frequency of the harmonic oscillator.
  n_max : int, optional
    Maximum order of the wavefunction. By default 1 (ground state).

  Returns
  -------
  psi: np.ndarray
    First 'n_max' normalized wavefunctions.
  """
  wfcs = []
  
  for n in range(n_max):
    # Components of the analytical solution for stationary states.
    prefactor = 1 / np.sqrt(2**n * factorial(n)) * (omega / np.pi) ** 0.25
    exponential = np.exp(- (omega * x**2) / 2)

    # Complete wavefunction, with normalization
    psi = prefactor * exponential * hermite(x * np.sqrt(omega), n)
    psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2))
    
    wfcs.append(psi_normalized.astype(complex))
  
  wfcs = np.array(wfcs)
  
  return wfcs