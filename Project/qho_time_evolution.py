# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import harmonic_functions as hfu

from scipy.linalg import sqrtm

# ===========================================================================================================
# PARAM CLASS
# ===========================================================================================================

class Param:
  """
  Param: 
    Container for holding all simulation parameters.
  """
  def __init__(self,
               x_min: float,
               x_max: float,
               num_x: int,
               tsim: float,
               num_t: int,
               tc: float,
               num_tc:int,
               im_time: bool = False) -> None:
    """
    __init__ : 
      Initialize simulation parameters.

    Parameters
    ----------
    x_min : float
      Minimum spatial value.
    x_max : float
      Maximum spatial value.
    num_x : int
      Number of spatial grid points.
    tsim : float
      Total simulation time.
    num_t : int
      Number of time steps.
    im_time : bool, optional
      Whether to use imaginary time evolution. Default is False.
    """
    # Initialization
    self.x_min = x_min
    self.x_max = x_max
    self.num_x = num_x
    self.tsim = tsim
    self.num_t = num_t
    self.tc = tc
    self.num_tc = num_tc
    self.im_time = im_time

    # Infinitesimal quantities (space, time, momentum)
    self.dx = (x_max - x_min) / num_x
    self.dt = tsim / num_t
    self.dk = 2 * np.pi / (x_max - x_min)
    
    # Check consistency in time-grids
    assert (np.allclose(self.dt, tc / num_tc))

    # Spatial grid
    self.x = np.linspace(x_min + 0.5 * self.dx, x_max - 0.5 * self.dx, num_x)

    # Momentum grid -> For FFT, frequencies are in this order
    self.k = np.fft.fftfreq(num_x, d=self.dx) * 2 * np.pi

    # validation check
    self._validate()

  def _validate(self) -> None:
    """
    _validate :
      Check for common errors in parameter initialization.
    """
    if self.num_x <= 0 or self.num_t <= 0:
      raise ValueError("num_x and num_t must be both positive integers. \nGot num_x={num_x} and num_t={num_t}.", stop=True)
    if self.x_max <= 0 or self.tsim <= 0:
      raise ValueError("xmax and tsim must be both positive integers. \nGot xmax={xmax} and tsim={tsim}.", stop=True)



# ===========================================================================================================
# AUXILIARY FUNCTIONS FOR OPERATOR CLASS
# ===========================================================================================================

def energy_basis(psi, x, num_wfcs, omega, delta_x):
  """
  Project the wavefunction `psi` onto the energy eigenbasis (Hermite functions).
  """
  # Initialize coefficients in energy basis
  coeff = np.zeros((num_wfcs, num_wfcs), dtype=complex)
  true_psi = hfu.harmonic_wfc(x - delta_x, omega, num_wfcs)
  
  # Loop over energy levels and compute overlap
  for n in range(num_wfcs):
    for i in range(num_wfcs):
      coeff[i, n] = true_psi[i] @ np.conj(psi[n])
  return np.round(np.real_if_close(coeff), 4)

def density_matrix(probabilities, wavefunctions, x, num_wfcs, omega, delta_x):
  rho = 0
  rho = np.zeros((num_wfcs, num_wfcs), dtype=complex)
  coefficients = energy_basis(wavefunctions, x, num_wfcs, omega, delta_x)
  
  for i in range(num_wfcs):
    rho += probabilities[i] * np.outer(coefficients[i], np.conj(coefficients[i]))
    
  rho /= np.trace(rho)
  rho = np.nan_to_num(rho, nan=0.0)

  return np.round(np.real_if_close(rho), 4)

def calculate_energies(wavefunctions, potential, par: Param):
  energies = []
  
  for wfc in wavefunctions:
    # Creating momentum and conjugate wavefunctions.
    wfc_k = np.fft.fft(wfc)
    wfc_c = np.conj(wfc)
    
    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k)
    energy_r = wfc_c * potential * wfc
    
    # Integrating over all space
    energy_final = sum(energy_k + energy_r).real
    
    # Store the energy
    energies.append(energy_final)
  
  return energies

def calculate_probabilities(energies, T):
  # Store probabilities for mixed state density matrix
  probs = [np.exp(-energies[i] / T) for i in range(len(energies))]
  probs /= np.sum(probs)
    
  return probs


# ===========================================================================================================
# OPERATORS CLASS
# ===========================================================================================================

class Operators:
  """
  Container for holding operators and wavefunction coefficients.
  """
  def __init__(self, 
               res: int,
               omega: float = 1.0,
               num_wfcs: int = 1,
               T: float = 10e-6,
               r_t: list[float] = None,
               par: Param = None) -> None:

    # Initialize empty complex arrays for potential, propagators, and wavefunction
    self.V = np.empty(res, dtype=complex)  # Potential operator
    self.R = np.empty(res, dtype=complex)  # Real-space propagator
    self.K = np.empty(res, dtype=complex)  # Momentum-space propagator
    self.wfcs = np.empty((num_wfcs+1, res), dtype=complex)  # Wavefunction coefficients
    self.shifted_wfcs = np.empty((num_wfcs+1, res), dtype=complex)  # Shifted wavefunction coefficients
    
    # Store finite temperature
    self.T = T
    
    # Energy list for  (finite temperature case)
    self.energies = []
    self.probabilities = []
    
    # Density matrices
    self.rho = []
    self.shifted_rho = []
    
    # Infidelity
    self.average_infidelity = 0
    
    # Store time-dependent drive (default to no potential if None)
    self.r_t = r_t if r_t is not None else np.zeros(par.num_t)
    
    # Store angular frequency
    self.omega = omega
    
    # Store maximum order of the Hermite polynomial
    self.num_wfcs = num_wfcs

    # Initialize potential and wavefunction if a Param instance is provided
    if par is not None:
      self._initialize_operators(par)


  def reinitialize_operators(self, par: Param, pulse) -> None:
    """Reinitialize the operators with a new Param object."""
    # Reinitialize potential, propagators, and wavefunctions
    self.V = np.empty(self.V.shape, dtype=complex)
    self.R = np.empty(self.R.shape, dtype=complex)
    self.K = np.empty(self.K.shape, dtype=complex)
    self.wfcs = np.empty((self.num_wfcs, self.V.shape[0]), dtype=complex)
    self.shifted_wfcs = np.zeros((self.num_wfcs, self.V.shape[0]))

    # Reset density matrices and energy lists
    self.energies = []
    self.probabilities = []
    self.rho = []
    self.shifted_rho = []

    # Store new time-dependent drive
    self.r_t = pulse

    # Call the initialization function with the new parameter object
    self._initialize_operators(par)

  def _initialize_operators(self, par: Param) -> None:
    # Initial and final time-dependent offset
    r0 = self.r_t[0]
    rf = self.r_t[-1]

    # Quadratic potential with offset
    self.V = 0.5 * (par.x - r0) ** 2 * self.omega **2

    # Wavefunctions and shifted wavefunctions based on analytical solution for the harmonic oscillator
    self.wfcs = hfu.harmonic_wfc(par.x - r0, self.omega, self.num_wfcs)
    self.shifted_wfcs = hfu.harmonic_wfc(par.x - rf, self.omega, self.num_wfcs)
    
    # Coefficient for imaginary or real time evolution
    coeff = 1 if par.im_time else 1j

    # Momentum and real-space propagators
    self.K = np.exp(-0.5 * (par.k ** 2) * par.dt * coeff)
    self.R = np.exp(-0.5 * self.V * par.dt * coeff)
    
    # Energies and probabilities for mixed state density matrix
    self.energies = calculate_energies(self.wfcs, self.V, par)
    self.probabilities = calculate_probabilities(self.energies, self.T)
    
    # Density matrix
    self.rho = density_matrix(self.probabilities, self.wfcs, par.x, self.num_wfcs, self.omega, r0)
    self.shifted_rho = density_matrix(self.probabilities, self.shifted_wfcs, par.x, self.num_wfcs, self.omega, rf)


  def get_wavefunction(self, n):
    """Retrieve the nth wavefunction."""
    if n >= self.num_wfcs:
      raise ValueError(f"Requested wavefunction n={n} exceeds num_wfcs={self.num_wfcs}. Indexing starts from 0.")
    return self.wfcs[n]

  def get_shifted_wavefunction(self, n):
    """Retrieve the nth wavefunction."""
    if n >= self.num_wfcs:
      raise ValueError(f"Requested wavefunction n={n} exceeds num_wfcs={self.num_wfcs}. Indexing starts from 0.")
    return self.shifted_wfcs[n]

      
  def infidelity(self):
    overlap = np.trace(sqrtm(sqrtm(self.rho) @ self.shifted_rho @ sqrtm(self.rho)))
    infidelity = 1 - abs(overlap) ** 2
    #infidelity = np.maximum(0, infidelity)
    infidelity = np.clip(infidelity, 1e-3, 1)
    return infidelity
  
  
  def split_op(self, par: Param, fixed_potential: bool = False):
    # Set coefficient for real or imaginary time evolution
    coeff = 1 if par.im_time else 1j
    
    timesteps = par.num_t if not fixed_potential else par.num_tc
    drive = self.r_t if not fixed_potential else np.full(par.num_tc, self.r_t[-1])

    infidelities = []
    
    # Loop over the number of timesteps    
    for i in range(timesteps):
      # If not fixed, update the time-dependent potential
      self.V = 0.5 * (par.x - drive[i]) ** 2 * self.omega ** 2

      # Update the real-space propagator
      self.R = np.exp(-0.5 * self.V * par.dt * coeff)

      # Loop over all wavefunctions
      for n in range(self.num_wfcs):
        # Half-step in real space
        self.wfcs[n] *= self.R

        # Full step in momentum space
        self.wfcs[n] = np.fft.fft(self.wfcs[n])
        self.wfcs[n] *= self.K
        self.wfcs[n] = np.fft.ifft(self.wfcs[n])

        # Final half-step in real space
        self.wfcs[n] *= self.R
      
      if fixed_potential:
        # Compute updated quantities
        self.energies = calculate_energies(self.wfcs, self.V, par)
        self.probabilities = calculate_probabilities(self.energies, self.T)
        self.rho = density_matrix(self.probabilities, self.wfcs, par.x, self.num_wfcs, self.omega, drive[i])
        infidelities.append(self.infidelity())
      
    self.average_infidelity = np.mean(infidelities) if fixed_potential else 0

  def time_evolution(self, par: Param, fixed_potential: bool = False):
    # Apply split operator to wfcs
    self.split_op(par, fixed_potential)
    rf = self.r_t[-1] 
    self.shifted_rho = density_matrix(self.probabilities, self.shifted_wfcs, par.x, self.num_wfcs, self.omega, rf)
    
    if not fixed_potential:
      # Compute updated quantities
      self.energies = calculate_energies(self.wfcs, self.V, par)
      self.probabilities = calculate_probabilities(self.energies, self.T)
      self.rho = density_matrix(self.probabilities, self.wfcs, par.x, self.num_wfcs, self.omega, rf)