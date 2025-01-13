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
# REAL SPACE RENORMALIZATION GROUP
# ===========================================================================================================

def initialize_A_B(N):
  s_x, _, _ = im.pauli_matrices()
  
  A_0 = sp.kron(sp.identity(2**(N - 1), format='csr'), s_x)
  B_0 = sp.kron(s_x, sp.identity(2**(N - 1), format='csr'))
  
  return A_0, B_0

# ===========================================================================================================

def compute_H_2N(N, H, A, B):  
  H_2N = sp.kron(H, sp.identity(2**(N), format='csr')) + sp.kron(sp.identity(2**(N), format='csr'), H) + sp.kron(A, B)
  return H_2N

# ===========================================================================================================

def projector(H, d_eff):
  _, eigvecs = sp.linalg.eigsh(H, k=d_eff, which='SA')  # Compute the smallest `k` eigenvalues
    
  proj = sp.csr_matrix(eigvecs)

  return proj

# ===========================================================================================================

def update_operators(N, H_2N, A, B):
  P = projector(H_2N, d_eff=2**N)
  
  P_dagger = P.conj().T
  I_N = sp.identity(2**N, format='csr')

  # Compute H_Nnn, Ann, Bnn, Pnn
  H_eff = P_dagger @ H_2N @ P
  A_eff = P_dagger @ sp.kron(I_N, A) @ P
  B_eff = P_dagger @ sp.kron(B, I_N) @ P
  
  return H_eff, A_eff, B_eff, P

# ===========================================================================================================
  
def real_space_rg(N, l, threshold, d_eff, max_iter=100, verb=False):
  prev_energy_density = np.inf
  H = im.ising_hamiltonian(N, l)
  A, B = initialize_A_B(N)
  curr_dim = N

  for iteration in range(max_iter):
    curr_dim *= 2
    H_2N = compute_H_2N(N, H, A, B)

    # Compute the current energy density and eigenvectors
    E, psi = sp.linalg.eigsh(H_2N, k=d_eff, which='SA')
    E_ground = E[0]
    psi_ground = psi[:, 0]
    
    current_energy_density = E_ground / curr_dim

    # Check for convergence 
    delta = abs(current_energy_density - prev_energy_density)

    if delta > threshold:
      H, A, B, P = update_operators(N, H_2N, A, B)
    else:
      print(f"Convergence achieved at iteration {iteration}")
      break

    # Update previous energy density for next iteration
    prev_energy_density = current_energy_density
    
    if verb and iteration % 10 == 0:
      print(f"Starting iteration {iteration} ...")

  
  print(f"Reached N = 2*{N} x 2**{iteration} = {curr_dim} with precision: delta = {delta}")
  return current_energy_density, E_ground, psi_ground, curr_dim

# ===========================================================================================================

def update_hamiltonian(N, l_values, threshold, max_iter=100):
  # Initialize dictionaries to store eigenvalues and eigenvectors
  gs_density_dict = {}
  gs_energy_dict = {}
  gs_dict = {}
  dims = {}
  
  print(f"Analysis with N={2**(max_iter)}...")

  for l in l_values:      
    d_eff = 2**N    
    energy_density_ground, E_ground, psi_ground, reached_dim = real_space_rg(N, l, threshold, d_eff, max_iter)  
    
    gs_density_dict[(reached_dim, l)] = energy_density_ground
    gs_energy_dict[(reached_dim, l)] = E_ground
    gs_dict[(reached_dim, l)] = psi_ground
    dims[(reached_dim, l)] = reached_dim
    
  print("-----------------------------------------")
    
  return gs_density_dict, gs_energy_dict, gs_dict, dims