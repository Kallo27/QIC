###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 8 - RG and INFINITE DMRG


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import scipy.sparse as sp
import ising_model as im

# ===========================================================================================================

def initialize_operators(m, l):
  """
  initialize_operators :
    Initializes operators for the iDMRG algorithm

  Parameters
  ----------
  m : int
    Quenched size.
  l : float
    Interaction strength.

  Returns
  -------
  A_L_0, B_L_0, A_R_0, B_R_0, H_L_0, H_R_0 : tuple of csr matrices
    Initialized operators for the iDMRG algorithm.
  """
  s_x, _, _ = im.pauli_matrices()
  
  A_L_0 = A_R_0 = sp.identity(2**m, format='csr')
  B_L_0 = sp.kron(sp.identity(2**(m - 1), format='csr'), s_x)
  B_R_0 = sp.kron(s_x, sp.identity(2**(m - 1), format='csr'))
  
  H_L_0 = H_R_0 = im.ising_hamiltonian(m, l)
  
  return A_L_0, B_L_0, A_R_0, B_R_0, H_L_0, H_R_0

# ===========================================================================================================

def compute_H_LR(H_L, H_R, A_L, B_L, A_R, B_R, l):
  """
  compute_H_LR :
    Computes the left ad right operators of the enlarged system.

  Parameters
  ----------
  H_L : csr matrix
    Current left Hamiltonian.
  H_R : csr matrix
    Current right Hamiltonian.
  A_L : csr matrix
    Left non-interaction operator
  B_L : csr matrix
    Left interaction operator
  A_R : csr matrix
    Right non-interaction operator
  B_R : csr matrix
    Right interaction operator
  l : float
    Interaction strength.

  Returns
  -------
  H_L1, H_R1 : tuple of csr matrices
    Left and right operators of the enlarged system.
  """
  s_x, _, s_z = im.pauli_matrices()
  H_L1 = sp.kron(H_L, sp.identity(2, format='csr')) + sp.kron(A_L, l * s_z) + sp.kron(B_L, s_x)
  H_R1 = sp.kron(sp.identity(2, format='csr'), H_R) + sp.kron(l * s_z, A_R) + sp.kron(s_x, B_R)
  return H_L1, H_R1

# ===========================================================================================================

def update_operators(m):
  """
  update_operators:
    Updates operators before truncation.

  Parameters
  ----------
  m : int
    Quenched size.

  Returns
  -------
  A_L_new, B_L_new, A_R_new, B_R_new : tuple of csr_matrices
    Updated operators for the iDMRG algorithm.
  """
  s_x, _, _ = im.pauli_matrices()
  
  A_L_new = A_R_new = sp.identity(2**(m+1), format='csr')
  
  B_L_new = sp.kron(sp.identity(2**(m), format='csr'), s_x)
  B_R_new = sp.kron(s_x, sp.identity(2**(m), format='csr'))
  
  return A_L_new, B_L_new, A_R_new, B_R_new

# ===========================================================================================================

def compute_H_2m(H_L1, H_R1, m):
  """
  compute_H_2m : 
    Computes enlarged Hamiltonian for the iDMRG algorithm.

  Parameters
  ----------
  H_L1 : csr matrix
    Enlarged left operator.
  H_R1 : csr matrix
    Enlarged right operator.
  m : int
    Quenched size.

  Returns
  -------
  H_2m : csr matrix
      _description_
  """
  s_x, _, _ = im.pauli_matrices()
  I_m = sp.identity(2**(m), format='csr')
  I_m1 = sp.identity(2**(m+1), format='csr')
  
  H_int = sp.kron(s_x, s_x)
  H_LR = sp.kron(I_m, sp.kron(H_int, I_m))
  
  H_2m = (sp.kron(H_L1, I_m1) + sp.kron(I_m1, H_R1) + H_LR)
  
  return H_2m

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
  rdm : np.ndarray
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

  # Reshape the wavefunction into a sparse tensor (use csr_matrix for efficient sparse storage)
  psi_tensor = psi.reshape([D] * N)

  # Reorder the axes to group subsystem (first) and environment (second)
  all_indices = list(range(N))
  env_indices = [i for i in all_indices if i not in keep_indices]  # complement of keep_indices
  reordered_tensor = np.transpose(psi_tensor, axes=keep_indices + env_indices)

  # Partition into subsystem and environment (reshape back)
  psi_partitioned = reordered_tensor.reshape((subsystem_dim, env_dim))

  # Compute the reduced density matrix (use sparse matrix multiplication)
  rdm = psi_partitioned.dot(psi_partitioned.conj().T)

  return rdm

# ===========================================================================================================

def projector(rho_L, k):
  """
  projector :
    Builds (2**m, k) projector for the iDMRG algorithm.

  Parameters
  ----------
  rho_L : csr matrix
    Reduced density matrix
  k : int
    Number of eigenvectors to keep after truncation.

  Returns
  -------
  proj _ csr matrix
    (2**m, k) projector for the iDMRG algorithm.
  """
  if k > rho_L.shape[0]:
    raise ValueError(f"'k' must be <= the dimension of rho_L, got k={k} and dim={rho_L.shape[0]}")

  _, eigvecs = sp.linalg.eigsh(rho_L, k=k, which='LA')  # Compute the largest `k` eigenvalues
  proj = sp.csr_matrix(eigvecs)
  return proj

# ===========================================================================================================

def truncate_operators(P_L, P_R, A_L, B_L, A_R, B_R, H_L, H_R):
  """
  truncate_operators :
    Truncates operator using the (2**m, k) projector for the iDMRG algorithm.

  Parameters
  ----------
  P_L : csr matrix
    Left projector.
  P_R : csr matrix
    Right projector.
  A_L : csr matrix
    Left non-interaction operator
  B_L : csr matrix
    Left interaction operator
  A_R : csr matrix
    Right non-interaction operator
  B_R : csr matrix
    Right interaction operator
  H_L : csr matrix
    Current left Hamiltonian.
  H_R : csr matrix
    Current right Hamiltonian.

  Returns
  -------
  A_L_trunc, B_L_trunc, A_R_trunc, B_R_trunc, H_L_trunc, H_R_trunc : tuple of csr matrices
    Truncated operators of size (k, k)
  """
  P_L_dagger = P_L.conj().T
  P_R_dagger = P_R.conj().T
  
  A_L_trunc = P_L_dagger @ A_L @ P_L
  B_L_trunc = P_L_dagger @ B_L @ P_L
  H_L_trunc = P_L_dagger @ H_L @ P_L
  
  A_R_trunc = P_R_dagger @ A_R @ P_R
  B_R_trunc = P_R_dagger @ B_R @ P_R
  H_R_trunc = P_R_dagger @ H_R @ P_R

  return A_L_trunc, B_L_trunc, A_R_trunc, B_R_trunc, H_L_trunc, H_R_trunc

# ===========================================================================================================

def infinite_dmrg(l, m_0, m_max, threshold=1e-6, max_iter=100, verb=False):
  """
  infinite_dmrg : 
    Iteratively applies iDMRG to compute the ground-state energy density 
    and wavefunction for a system starting at size 2*m_0.
    
  Parameters
  ----------
  l : float
    Interaction strength.
  m_0 : int
    Quenched size.
  m_max : int
    Maximum number of states to keep after truncation.
  convergence_threshold : float, optional
      Convergence threshold for the iDMRG algorithm.
  max_iter : int, optional
    Maximum number of iteration allowed. Default is 100.
  verb : bool, optional
    Verbosity flag. Default is False

  Returns
  -------
  E_ground : float
      Ground state energy per site in the thermodynamic limit.
  """
  # Initialize operators and Hamiltonians
  A_L, B_L, A_R, B_R, H_L, H_R = initialize_operators(m_0, l)
  
  curr_dim = 2 * m_0
  prev_energy_density = np.inf
  
  for iteration in range(max_iter):
    # Step 1: Enlarge Hamiltonians
    H_L1, H_R1 = compute_H_LR(H_L, H_R, A_L, B_L, A_R, B_R, l)
    
    # Step 2: Combine into full system Hamiltonian
    H_2m = compute_H_2m(H_L1, H_R1, m_0)
    
    # Step 3: Compute the ground state and wavefunction
    E, psi = sp.linalg.eigsh(H_2m, k=1, which='SA')
    E_ground = E[0]
    psi_ground = psi[:, 0]
    
    # Step 4: Compute reduced density matrix
    N = int(np.log2(H_2m.shape[0]))
    D = 2  # Local Hilbert space dimension
    left_indices = list(range(0, N // 2))  # Keep left block sites
    rho_L = rdm(psi_ground, N, D, left_indices)
    
    right_indices = list(range(N // 2, N))  # Keep left block sites
    rho_R = rdm(psi_ground, N, D, right_indices)
    
    # Step 5: Construct the projector
    k = min(2 ** m_max, 2 ** m_0)  # Ensure k does not exceed the dimension
    P_L = projector(rho_L, k)
    P_R = projector(rho_R, k)
            
    # Step 6: Truncate operators and Hamiltonians
    A_L, B_L, A_R, B_R = update_operators(m_0)
    
    if m_0 != m_max:
      m_0 = m_max
    
    A_L, B_L, A_R, B_R, H_L, H_R = truncate_operators(P_L, P_R, A_L, B_L, A_R, B_R, H_L1, H_R1)
            
    # Step 7: Check convergence
    curr_dim += 2
    current_energy_density = E_ground / curr_dim
    delta = abs(current_energy_density - prev_energy_density)

    if delta < threshold:
      print(f"Converged after {iteration + 1} iterations.")
      break

    # Update for the next iteration
    prev_energy_density = current_energy_density
      
    if verb and iteration % 10 == 0:
      print(f"Starting iteration {iteration} ...")
          
  print(f"Reached N = {curr_dim} with precision: delta = {delta}")
  return current_energy_density, E_ground, psi_ground, curr_dim
  
# ===========================================================================================================

def update_hamiltonian(l_values, m_0, m_max, threshold, max_iter=100):
  """
  update_hamiltonian :
    Evaluates ground-state energy densities, eigenvalues, and wavefunctions 
    for different values of the interaction strength lambda. 

  Parameters
  ----------
  l_values : float
    Values of the interaction strength.
  m_0 : 
    Quenched size.
  m_max :
    Maximum size for truncation.
  threshold : float
    Convergence threshold for the iDMRG algorithm.
  max_iter : int, optional
    Maximum number of iteration allowed. Default is 100.

  Returns
  -------
  gs_density_dict, gs_energy_dict, gs_dict, dims: tuple of dict
    Dictionaries for different values of system size and l, which contain the current 
    energy densities, the ground energies and wavefunctions and the reached dimensions.
  """
  # Initialize dictionaries to store eigenvalues and eigenvectors
  gs_density_dict = {}
  gs_energy_dict = {}
  gs_dict = {}
  dims = {}
  
  print(f"Analysis with N={2*(max_iter) + 2*m_0}...")

  for l in l_values:      
    energy_density_ground, E_ground, psi_ground, reached_dim = infinite_dmrg(l, m_0, m_max, threshold, max_iter)  
    
    gs_density_dict[(reached_dim, l)] = energy_density_ground
    gs_energy_dict[(reached_dim, l)] = E_ground
    gs_dict[(reached_dim, l)] = psi_ground
    dims[(reached_dim, l)] = reached_dim
    
  print("-----------------------------------------")
    
  return gs_density_dict, gs_energy_dict, gs_dict, dims