###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 6 - DENSITY MATRICES


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import time
import matplotlib.pyplot as plt

from sys import getsizeof


# ===========================================================================================================
# STATE GENERATION
# ===========================================================================================================

def separable_state(N, D, predefined_states=None):
  """
  separable_state:
    Generates a separable pure state for an N-body non-interacting system.

  Parameters
  ----------
  N : int
    Number of subsystems.
  D : int
    Dimension of each subsystem.
  predefined_states : list of np.ndarray, optional
    List of N predefined state vectors for each subsystem (dimension D).
    If not provided, states are initialized randomly. Default is None.

  Returns
  -------
  total_state: np.ndarray
    Normalized separable pure state vector for the composite system.
  """
  # Initialize the states as predefined states or as empty list
  states = predefined_states if predefined_states else []

  if not predefined_states:
    # Generate N random state vectors of dimension D
    for _ in range(N):
      psi = np.random.rand(D) + 1j * np.random.rand(D)
      psi /= np.linalg.norm(psi)
      states.append(psi)
  else:
    # Ensure the provided states match the dimensions
    if len(predefined_states) != N or any(len(psi) != D for psi in predefined_states):
      raise ValueError("Predefined states must match the number and dimension of subsystems (N, D).")

  # Create the composite state
  total_state = states[0]
  for i in range(1, N):
    total_state = np.kron(total_state, states[i])

  return total_state

# ===========================================================================================================

def general_state(N, D, predefined_state=None):
  """
  general_state:
    Generates a general N-body pure wave function in H^(DN).

  Parameters
  ----------
  N : int
    Number of subsystems.
  D : int
    Dimension of each subsystem.
  predefined_state : np.ndarray, optional
    Predefined state vector of size D^N. Default is None

  Returns
  -------
  psi: numpy.ndarray
    Normalized state vector in H^(DN).
  """
  # Total dimension of the composite Hilbert space
  total_dim = D**N

  if predefined_state is not None:
    # Ensure the provided state match the dimension
    if len(predefined_state) != total_dim:
      raise ValueError(f"Predefined state must have length {total_dim}.")
    psi = predefined_state
  else:
    # Generate a normalized random complex wave function
    psi = np.random.rand(total_dim) + 1j * np.random.rand(total_dim)

  # Ensure the state is normalized
  psi /= np.linalg.norm(psi)

  return psi

# ===========================================================================================================
# EFFICIENCY
# ===========================================================================================================

def test_efficiency(N_values, D_values, repetitions=10):
  """
  test_efficiency:
    Tests the efficiency of separable and general state preparation.

  Parameters
  ----------
  N_values : list of int
    List of numbers of subsystems to test.
  D_values : list of int
    List of dimensions of subsystems to test.
  repetitions : int
    Number of times to repeat the generation (for averaging). Default is 10.

  Returns
  -------
  separable_times, general_times: tuple of np.ndarray
    Arrays (for separable and general states) of average times (N x D).
  """
  separable_times = np.zeros((len(N_values), len(D_values)))
  general_times = np.zeros((len(N_values), len(D_values)))
  separable_bytes = np.zeros((len(N_values), len(D_values)))
  general_bytes = np.zeros((len(N_values), len(D_values)))

  for i, N in enumerate(N_values):
    for j, D in enumerate(D_values):
      # Time the separable state generation
      sep_times = []
      sep_bytes = []
      for _ in range(repetitions):
        start = time.time()
        sep = separable_state(N, D)
        sep_times.append(time.time() - start)
        sep_bytes.append((sep.nbytes + getsizeof(sep)) / 1e6)

      # Time the general state generation
      gen_times = []
      gen_bytes = []
      for _ in range(repetitions):
        start = time.time()
        gen = general_state(N, D)
        gen_times.append(time.time() - start)
        gen_bytes.append((gen.nbytes + getsizeof(gen)) / 1e6)
      
      separable_times[i, j] = np.mean(sep_times)
      general_times[i, j] = np.mean(gen_times)
      separable_bytes[i, j] = np.mean(sep_bytes)
      general_bytes[i, j] = np.mean(gen_bytes)

  return separable_times, general_times, separable_bytes, general_bytes

# ===========================================================================================================

def plot_efficiency(N_values, D_values, repetitions=10):
  """
  Plots a combined heatmap for separable and general state efficiencies.

  Parameters
  ----------
  N_values : list of int
    List of numbers of subsystems tested.
  D_values : list of int
    List of dimensions of subsystems tested.
  repetitions : int
    Number of times to repeat the generation (for averaging). Default is 10.
    
  Returns
  -------
  None
  """
  # Generate efficiency data
  sep, gen, sep_b, gen_b = test_efficiency(N_values, D_values, repetitions)

  # Compute global min and max across both heatmaps
  global_min = min(sep.min(), gen.min())
  global_max = max(sep.max(), gen.max())
  global_min_bytes = min(sep_b.min(), gen_b.min())
  global_max_bytes = max(sep_b.max(), gen_b.max())

  fig, ax = plt.subplots(2, 2, figsize=(12, 12))

  # Plot heatmap for separable states
  im1 = ax[0, 0].imshow(sep, cmap="cividis", origin="lower", aspect="auto", vmin=global_min, vmax=global_max)
  ax[0, 0].set_title("Separable state")
  ax[0, 0].set_xticks(range(len(D_values)))
  ax[0, 0].set_yticks(range(len(N_values)))
  ax[0, 0].set_xticklabels(D_values)
  ax[0, 0].set_yticklabels(N_values)
  ax[0, 0].set_xlabel("D")
  ax[0, 0].set_ylabel("N")
  cbar1 = plt.colorbar(im1, ax=ax[0, 0])

  # Plot heatmap for general states
  im2 = ax[0, 1].imshow(gen, cmap="cividis", origin="lower", aspect="auto", vmin=global_min, vmax=global_max)
  ax[0, 1].set_title("General state")
  ax[0, 1].set_xticks(range(len(D_values)))
  ax[0, 1].set_yticks(range(len(N_values)))
  ax[0, 1].set_xticklabels(D_values)
  ax[0, 1].set_yticklabels(N_values)
  ax[0, 1].set_xlabel("D")
  ax[0, 1].set_ylabel("N")
  cbar2 = plt.colorbar(im2, ax=ax[0, 1])
  
  # Plot heatmap for separable states bytes
  im3 = ax[1, 0].imshow(sep_b, cmap="cividis", origin="lower", aspect="auto", vmin=global_min_bytes, vmax=global_max_bytes)
  ax[1, 0].set_title("Separable state")
  ax[1, 0].set_xticks(range(len(D_values)))
  ax[1, 0].set_yticks(range(len(N_values)))
  ax[1, 0].set_xticklabels(D_values)
  ax[1, 0].set_yticklabels(N_values)
  ax[1, 0].set_xlabel("D")
  ax[1, 0].set_ylabel("N")
  cbar3 = plt.colorbar(im3, ax=ax[1, 0])

  # Plot heatmap for general states bytes
  im4 = ax[1, 1].imshow(gen_b, cmap="cividis", origin="lower", aspect="auto", vmin=global_min_bytes, vmax=global_max_bytes)
  ax[1, 1].set_title("General state")
  ax[1, 1].set_xticks(range(len(D_values)))
  ax[1, 1].set_yticks(range(len(N_values)))
  ax[1, 1].set_xticklabels(D_values)
  ax[1, 1].set_yticklabels(N_values)
  ax[1, 1].set_xlabel("D")
  ax[1, 1].set_ylabel("N")
  cbar4 = plt.colorbar(im4, ax=ax[1, 1])
  
  fig.suptitle("Efficiency heatmaps", fontsize=16, y=0.92)
  fig.text(0.5, 0.49, "Memory usage heatmaps (in Mb)", fontsize=14, ha="center", va="center")

  #plt.tight_layout()
  plt.show()

# ===========================================================================================================
# DENSITY MATRIX
# ===========================================================================================================

def build_density_matrix(state):
  """
  build_density_matrix:
    Builds the density matrix of a given state.
    
  Parameters
  ----------
  state : np.ndarray
    General quantum state. 

  Returns
  -------
  density_matrix : np.ndarray
    Density matrix
  """
  density_matrix = np.outer(state, state.conj())
  return density_matrix

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
  # Check correct values for 'keep_indeces'
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

def trace_check(density_matrix):
  trace = round(np.trace(density_matrix), 8)
  result = "The trace is 1 (as expected)" if trace == 1 else f"The trace is: {trace} \n Something is wrong!!!"
  return result