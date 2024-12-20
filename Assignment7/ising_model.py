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

# ===========================================================================================================
# ISING MODEL
# ===========================================================================================================

def pauli_matrices():
  """
  pauli_matrices:
    Builds the Pauli matrices as sparse matrices.

  Returns
  -------
  s_x, s_y, s_z: tuple of sp.csr_matrix
    Pauli matrices for a 2x2 system in sparse format.
  """
  s_x = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
  s_y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
  s_z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
  return s_x, s_y, s_z

# ===========================================================================================================

def ising_hamiltonian(N, l):
  """
  ising_hamiltonian:
    Builds the Ising model Hamiltonian using sparse matrices.

  Parameters
  ----------
  N : int
    Number of spins.
  l : float
    Interaction strength.

  Returns
  -------
  H : sp.csr_matrix
    Sparse Ising Hamiltonian.
  """
  dim = 2 ** N
  H_nonint = sp.csr_matrix((dim, dim), dtype=complex)
  H_int = sp.csr_matrix((dim, dim), dtype=complex)
  
  s_x, _, s_z = pauli_matrices()
  
  for i in range(N):
    zterm = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_z, sp.identity(2**(N - i - 1), format='csr')))
    H_nonint += zterm
    
  for i in range(N - 1):
    xterm = sp.kron(sp.identity(2**i, format='csr'), sp.kron(s_x, sp.kron(s_x, sp.identity(2**(N - i - 2), format='csr'))))
    H_int += xterm
  
  H = H_int + l * H_nonint
  return H

# ===========================================================================================================
# EIGENVALUES
# ===========================================================================================================

def diagonalize_ising(N_values, l_values, k):
  """
  diagonalize_ising :
    Diagonalize the Ising Hamiltonian for different values of N and l using sparse methods.

  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.

  Returns
  -------
  eigenvalues, eigenvectors : tuple of dict
    Eigenvalues and eigenvectors of the Ising Hamiltonian for different
    values of N and l.
  """
  eigenvalues = {}
  eigenvectors = {}
  
  for N in N_values:
    print(f"Diagonalizing Ising Hamiltonian with N={N} ...")
    x = min(k, N - 1)
    
    for l in l_values:
      # Generate the sparse Ising Hamiltonian
      H = ising_hamiltonian(N, l)
      
      # Diagonalize the Hamiltonian
      
      eigval, eigvec = sp.linalg.eigsh(H, k=x, which='SA')  # Compute the smallest `k` eigenvalues
      eigvec = eigvec.T

      for i in range(x):
        eigvec[i] /= np.linalg.norm(eigvec[i])
        
      eigenvalues[(N, l)] = eigval
      eigenvectors[(N, l)] = eigvec
  
  return eigenvalues, eigenvectors


# ===========================================================================================================

def plot_eigenvalues(N_values, l_values, eigenvalues):
  """
  plot_eigenvalues :
    Plot the first k energy levels as a function of l for different N.
  
  Parameters
  ----------
  N_values : list of int
    Values of N, number of spins in the system.
  l_values : list of float
    Values of l, interaction strength.
  eigenvalues : list of float
    Precomputed eigenvalues for every (N, l).
  k : int
    Number of lowest energy levels to plot.
  
  Returns
  ----------
  None
  """  
  # Loop over the values of N (many plots)
  for N in N_values:
    plt.figure(figsize=(8, 5))
    
    # Compute the number of available eigenvalues
    k = len(eigenvalues[N, l_values[0]])
      
    # Loop over the first k levels
    for level in range(k):
      energies = []
      for l in l_values:
        energies.append(eigenvalues[(N, l)][level] / N)
          
      plt.plot(l_values, energies, label=f'Level {level + 1}')
    
    plt.axvline(x = -1, linestyle="--", color = "red", label="Critical point")
        
    # Plot formatting
    plt.xlabel('Interaction strength (λ)')
    plt.ylabel('Energy')
    plt.title(f'First {k} energy levels vs λ (N={N})')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

# ===========================================================================================================

def plot_energy_gaps(N_values, l_values, eigenvalues):
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
  
  Returns
  ----------
  None
  """  
  plt.figure(figsize=(8, 5))
    
  # Loop over the values of N (many plots)
  for N in N_values:    
    gaps = []
    # Loop over the first k levels
    for l in l_values:
      gap = (eigenvalues[(N, l)][1] - eigenvalues[(N, l)][0]) / N
      gaps.append(gap)

    plt.plot(l_values, gaps, label = f"level {N}")
  
  plt.axvline(x = -1, linestyle="--", color = "red", label="Critical point")
      
  # Plot formatting
  plt.xlabel('Interaction strength (λ)')
  plt.ylabel('Energy')
  plt.title(f'Normalized energy gap vs λ')
  plt.legend(loc="upper right")
  plt.grid()
  plt.show()