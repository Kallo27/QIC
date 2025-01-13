###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 8 - RG and INFINITE DMRG


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import ising_model as im

# ===========================================================================================================

def magnetization(ground_state, dim):
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
  
  M_z = sp.csr_matrix((2**dim, 2**dim), dtype=complex)
  
  for i in range(dim):
    M_z_i = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_z, sp.identity(2**(dim - i - 1), format='csr')))
    M_z += M_z_i
    
  M_z /= dim
  
  M = ground_state.conj().transpose().dot(M_z.dot(ground_state))
  return M

# ===========================================================================================================

def plot_magnetization(N, max_dim, l_values, eigenvectors):
  """
  plot_magnetization :
    Plot the magnetization as a function of l.
  
  Parameters
  ----------
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
  Ms = []
  # Loop over the first k levels
  for l in l_values:
    M = magnetization(eigenvectors[(max_dim, l)], N)
    Ms.append(M)

  plt.plot(l_values, Ms, marker='^', linestyle='--', label = f"N={max_dim}", markersize=3)
  
  plt.axvline(x=1, linestyle="--", color="red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Magnetization')
  plt.title(f'Magnetization vs λ')
  plt.xscale('log')
  plt.legend(loc="lower left")
  plt.grid()
  plt.show()
  
# ===========================================================================================================

def mean_field(l_vals):
  val = {}
  
  for l in l_vals:
    if abs(l) <= 2:
      val[l] = - 1 - (l**2) / 4
    else:
      val[l] = - abs(l)
      
  return val