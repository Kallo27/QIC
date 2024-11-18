import numpy as np
import debugger as db
import math
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from itertools import cycle


def kinetic_matrix(K, N, dx, order):
  # Constants
  hbar = 1.0  # Reduced Planck constant (set to 1 in atomic units)
  m = 1.0     # Mass of the particle (set to 1 in atomic units)
  
  factor = hbar**2 / (m * dx**2)
  
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
    K += np.diag(-factor / 120 * np.ones(N-3), k=3)  # Third upper diagonal
    K += np.diag(-factor / 120 * np.ones(N-3), k=-3)  # Third lower diagonal
    
  else:
    db.checkpoint(debug=True, msg1="APPROXIMATION ORDER", msg2="Unsupported order. Please choose order = 2, 4, or 6.", stop=True)
  
  return K

def harmonic_oscillator_spectrum(omega, L, N=1000, order=2):
  # Constants
  hbar = 1.0  # Reduced Planck constant (set to 1 in atomic units)
  m = 1.0     # Mass of the particle (set to 1 in atomic units)
  
  # Discretization
  x = np.linspace(-L, L, N)
  dx = x[1] - x[0]

  # Construct the Hamiltonian matrix
  N = N
  K = np.zeros((N, N))
  V = np.zeros((N, N))
  H = np.zeros((N, N))
  
  K = kinetic_matrix(K, N, dx, order)
  
  V_diag = 0.5 * m * omega**2 * x**2
  V = np.diag(V_diag)

  H = K + V

  energies, psi = eigh(H)
  norm = np.sqrt(np.sum(np.abs(psi)**2, axis=0))
  for i in range(len(energies)):
    psi[:, i] = psi[:, i] / norm
    
  center_index = N // 2  # assuming symmetric grid centered around x = 0

  for i in range(len(psi)):
    if i % 2 == 0:  # Even states
      # Ensure the wavefunction is positive at the center
      if ((i//2)%2==0 and psi[:, i][center_index] < 0) or ((i//2)%2!=0 and psi[:, i][center_index] > 0):
        psi[:, i] *= -1
    else :  # Odd states
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
    
  return energies, psi.T


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
  if n<0:
    db.checkpoint(debug=True, msg=f"The order of the Hermite polynomial is not valid (n={n}, expected n>=0)", stop=True)

  # Coefficients set to 0 except for the one of order n.
  herm_coeffs = np.zeros(n+1)
  herm_coeffs[n] = 1
  
  # Actual computation of the polinomial over the space grid.
  herm_pol = np.polynomial.hermite.hermval(x, herm_coeffs)
  return herm_pol

# ===========================================================================================================

def harmonic_en(omega=1.0, n=0):
  """
  harmonic_en:
      Energy levels for an harwmonic potential.

  Parameters
  ----------
  omega : float, optional
          Angular frequency of the harmonic potential. By default 1.0.
  n : int, optional
      Energy level. By default 0.

  Returns
  -------
  energy: float
          Energy of level 'n'.
      
  """
  # Constants set to 1 in atomic units.
  hbar = 1.0
  
  # Pre-condition: n>=0
  if n<0:
    db.checkpoint(debug=True, msg=f"The order of the energy level is not valid (n={n}, expected n>=0)", stop=True)
    
  # Complete wavefunction.
  energy = hbar * omega * (n + 1/2)
  return energy

# ===========================================================================================================

def harmonic_wfc(x, omega=1.0, n=0):
    """
    harmonic_wfc:
        Wavefunction of order 'n' for a harmonic potential, 
        defined over the real space grid 'x'.
        
        V(x) = 0.5 * m * omega * x**2
        
    Parameters
    ----------
    x : np.ndarray
        Real space grid.
    omega: float, optional
           Angular frequency of the harmonic potential. By default 1.0.
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
    
    # Components of the analytical solution for stationary states.
    prefactor = 1 / np.sqrt(2**n * math.factorial(n)) * ((m * omega) / (np.pi * hbar))**0.25
    x_coeff = np.sqrt(m * omega / hbar)
    exponential = np.exp(- (m * omega * x**2) / (2 * hbar))
    
    # Complete wavefunction.
    psi = prefactor * exponential * hermite(x_coeff * x, n)
    
    # Normalization condition.
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (x[1] - x[0]))  # Integrate using the grid spacing.
    psi_normalized = psi / norm
    
    return psi_normalized
  
# ===========================================================================================================
# PLOTTING FUNCTIONS
# ===========================================================================================================

def generate_colors(n):
  base_colors = ['blue', 'green', 'grey', 'black', 'purple']
  return [color for color, _ in zip(cycle(base_colors), range(n))]

# ===========================================================================================================

def plot_wf_en(omega, N, L, k, order=None):
  x = np.linspace(-L, L, N)
  if order is not None:
    energies, wavefunctions = harmonic_oscillator_spectrum(omega, L, N, order)
  else:
    energies = [harmonic_en(omega, k) for k in range(0, k)]
    wavefunctions = [harmonic_wfc(x, omega, k) for k in range(0, k)]

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

  colors = generate_colors(k)
  # Plot the wavefunctions in the first subplot
  for n in range(k):
    ax1.plot(x, wavefunctions[n], label=f"$\psi(x)$ of order {n}", color=colors[n], linewidth=1.5)
  
  ax1.set_xlabel("Position $x$")
  ax1.set_ylabel("Amplitude")
  ax1.set_title("Wavefunctions of Quantum Harmonic Oscillator")
  ax1.grid(True, linestyle='--', alpha=0.7)
  ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

  # Plot the potential and energy levels in the second subplot
  ax2.plot(x, 0.5 * omega**2 * x**2, label="Harmonic potential $V(x)$", color="red", linestyle="--", linewidth=1.5)
  for n in range(k):
    ax2.axhline(energies[n], label=f"Energy of order {n}", color=colors[n], linestyle="-.", linewidth=1, alpha=0.8)
   
  ax2.set_xlabel("Position $x$")
  ax2.set_ylabel("Energy")
  ax2.set_title("Energy Levels and Harmonic Potential")
  ax2.grid(True, linestyle='--', alpha=0.7)
  ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

  fig.tight_layout()
  plt.show()