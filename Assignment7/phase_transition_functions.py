###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 7 - ISING MODEL


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import matplotlib.pyplot as plt
import scipy.sparse as sp
import ising_model as im


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
  ----------
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
    Plot the energy gap (between first excited state and ground state) 
    as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvecttors : np.ndarray
    Precomputed eigenvectors for every (N, l).
  
  Returns
  ----------
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

    plt.plot(l_values, Ms, marker='^', linestyle='--', label = f"N={N}", markersize=6)
  
  plt.axvline(x = 1, linestyle="--", color = "red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Magnetization')
  plt.title(f'Magnetization vs λ')
  plt.xscale('log')
  plt.legend(loc="lower left")
  plt.grid()
  plt.show()