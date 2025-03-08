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
    tc : float
      Additional in-place simulation time.
    num_tc : int
      Number of additional time steps.
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
    
    # Store total evolution time
    self.total_time = 0
    
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
  energy_basis :
    Project the wavefunction `psi` onto the energy eigenbasis (Hermite functions).

  Parameters
  ----------
  psi : np.ndarray
    Wavefunctions to be projected.
  x : np.ndarray
    Spatial grid points where the wavefunctions are evaluated.
  num_wfcs : int
    Number of eigenfunctions used in the projection.
  omega : float
    Characteristic frequency of the harmonic oscillator.
  delta_x : float
    Spatial displacement applied to the eigenfunctions.

  Returns
  -------
  coeff : np.ndarray
    Projection coefficients of `psi` onto the energy eigenbasis.
  """
  # Initialize coefficients in energy basis
  coeff = np.zeros((num_wfcs, num_wfcs), dtype=complex)
  true_psi = hfu.harmonic_wfc(x - delta_x, omega, num_wfcs)
  
  # Loop over energy levels and compute overlap
  for n in range(num_wfcs):
    for i in range(num_wfcs):
      coeff[i, n] = true_psi[i] @ np.conj(psi[n])
  
  coeff = np.round(np.real_if_close(coeff), 4)
  
  return coeff

# ===========================================================================================================

def density_matrix(probabilities, wavefunctions, x, num_wfcs, omega, delta_x):
  """
  density_matrix :
    Compute the density matrix of a quantum system in the energy eigenbasis.

  Parameters
  ----------
  probabilities : np.ndarray
    Occupation probabilities of each wavefunction.
  wavefunctions : np.ndarray
    Set of wavefunctions representing the system.
  x : np.ndarray
    Spatial grid points where the wavefunctions are evaluated.
  num_wfcs : int
    Number of wavefunctions considered in the density matrix.
  omega : float
    Characteristic frequency of the harmonic oscillator.
  delta_x : float
    Displacement applied to the eigenfunctions.

  Returns
  -------
  rho : np.ndarray
    The computed density matrix in the energy eigenbasis.
  """
  rho = 0
  rho = np.zeros((num_wfcs, num_wfcs), dtype=complex)
  coefficients = energy_basis(wavefunctions, x, num_wfcs, omega, delta_x)
  
  for i in range(num_wfcs):
    rho += probabilities[i] * np.outer(coefficients[i], np.conj(coefficients[i]))
    
  rho /= np.trace(rho)
  rho = np.nan_to_num(rho, nan=0.0)
  rho = np.round(np.real_if_close(rho), 4)
  
  return rho

# ===========================================================================================================

def calculate_energies(wavefunctions, potential, par: Param):
  """
  calculate_energies :
    Compute the total energy of a quantum state given its wavefunction.

  Parameters
  ----------
  wavefunctions : np.ndarray
    Set of wavefunctions for which the energy is computed.
  potential : np.ndarray
    Potential energy function evaluated at the grid points.
  par : Param
    A parameter object containing the simultaions constants.

  Returns
  -------
  energies : list of float
    The computed energies for each wavefunction.
  """
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

# ===========================================================================================================

def calculate_probabilities(energies, T):
  """
  calculate_probabilities :
    Compute the occupation probabilities of quantum states using the Boltzmann distribution.

  Parameters
  ----------
  energies : list of float
    Energy levels of the quantum system.
  T : float
    Temperature used in the Boltzmann factor.

  Returns
  -------
  probs : np.ndarray
    Normalized occupation probabilities of the states.
  """
  # Store probabilities for mixed state density matrix
  probs = [np.exp(-energies[i] / T) for i in range(len(energies))]
  probs /= np.sum(probs)
    
  return probs

# ===========================================================================================================

def position_statistics(x, wavefunctions):
  """
  position_statistics :
    Compute the expectation values and standard deviation of position.

  Parameters
  ----------
  x : np.ndarray
    Spatial grid points where the wavefunctions are evaluated.
  wavefunctions : np.ndarray
    Set of wavefunctions for which position statistics are computed.

  Returns
  -------
  avg_positions, sigma_x : tuple of np.ndarray
    The expectation value ⟨x⟩ for each wavefunction and 
    the standard deviation of position, σ_x = sqrt(⟨x²⟩ - ⟨x⟩²).
  """
  avg_positions = []
  avg_positions_squared = []

  for wfc in wavefunctions:
    density = np.abs(wfc) ** 2
    avg_x = np.sum(x * density)
    avg_x2 = np.sum((x ** 2) * density)

    avg_positions.append(avg_x)
    avg_positions_squared.append(avg_x2)

  avg_positions = np.array(avg_positions)
  avg_positions_squared = np.array(avg_positions_squared)

  # Compute standard deviation: σ_x = sqrt(⟨x²⟩ - ⟨x⟩²)
  sigma_x = np.sqrt(avg_positions_squared - avg_positions ** 2)

  return avg_positions, sigma_x


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
    """
    __init__ :
      Initialize the Operators class.
    
    Parameters
    ----------
    res : int
      Number of spatial grid points.
    omega : float, optional
      Angular frequency of the system. By default 1.0.
    num_wfcs : int, optional
      Number of wavefunctions to store. By default 1.
    T : float, optional
      Temperature for thermal-state calculations. By default 10e-6.
    r_t : list[float], optional
      Time-dependent drive. By default None.
    par : Param, optional
      Parameter object containing simulation parameters. By default None.
    """
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
    
    # Average position
    self.avg_pos = []
    self.sigma_x = []
    self.total_drive = []
    
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
    """
    reinitialize_operators : 
      Reinitialize the operators with a new Param object and pulse sequence.
    
    Parameters
    ----------
    par : Param
      New parameter object for the simulation.
    pulse : list[float]
      New time-dependent drive sequence.
    """    
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
    """
    _initialize_operators : 
      Initialize the system's operators based on the given parameters.
    
    Parameters
    ----------
    par : Param
      Parameter object containing simulation parameters.
    """
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
    """
    get_wavefunction : 
      Retrieve the nth wavefunction.
    
    Parameters
    ----------
    n : int
      Index of the wavefunction to retrieve.
    
    Returns
    -------
    wfc : np.ndarray
      The nth wavefunction.
    """
    if n >= self.num_wfcs:
      raise ValueError(f"Requested wavefunction n={n} exceeds num_wfcs={self.num_wfcs}. Indexing starts from 0.")
    wfc = self.wfcs[n]
    
    return wfc

  def get_shifted_wavefunction(self, n):
    """
    get_shifted_wavefunction :
      Retrieve the nth shifted wavefunction.
    
    Parameters
    ----------
    n : int
      Index of the shifted wavefunction to retrieve.
    
    Returns
    -------
    wfc : np.ndarray
      The nth shifted wavefunction.
    """
    if n >= self.num_wfcs:
      raise ValueError(f"Requested wavefunction n={n} exceeds num_wfcs={self.num_wfcs}. Indexing starts from 0.")
    wfc = self.shifted_wfcs[n]

    return wfc
      
  def infidelity(self):
    """
    infidelity :
      Compute the infidelity between the initial and final density matrices.
    
    Returns
    -------
    infidelity : float
      Infidelity measure between the initial and final state.
    """
    overlap = np.trace(sqrtm(sqrtm(self.rho) @ self.shifted_rho @ sqrtm(self.rho)))
    infidelity = 1 - abs(overlap) ** 2
    infidelity = np.clip(infidelity, 1e-3, 1)
    
    return infidelity
  
  
  def split_op(self, par: Param, fixed_potential: bool = False, compute_statistics: bool = False):
    """
    split_op:
      Performs the time evolution with the split-operator approximation.

    Parameters
    ----------
    par : Param
      Param instance containing the parameters of the simulation.
    fixed_potential : bool, optional
      If true, performs evolution with a static potential. By default False.
    compute_statistics : bool, optional
      If true, computes positions statistics. By default False.
    """
    # Set coefficient for real or imaginary time evolution
    coeff = 1 if par.im_time else 1j
    
    timesteps = par.num_t if not fixed_potential else par.num_tc
    par.total_time += timesteps * par.dt
    drive = self.r_t if not fixed_potential else np.full(par.num_tc, self.r_t[-1])
    self.total_drive.extend(drive)

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
      
      self.energies = calculate_energies(self.wfcs, self.V, par)
      self.probabilities = calculate_probabilities(self.energies, self.T)
      self.rho = density_matrix(self.probabilities, self.wfcs, par.x, self.num_wfcs, self.omega, drive[i])
      
      if compute_statistics:  
        positions, sigma = position_statistics(par.x, self.wfcs)
        self.avg_pos.append(positions)
        self.sigma_x.append(sigma)
      
      if fixed_potential:
        infidelities.append(self.infidelity())
    

    self.average_infidelity = np.mean(infidelities) if fixed_potential else 0

  def time_evolution(self, par: Param, fixed_potential: bool = False, compute_statistics: bool = False):
    """
    time_evolution
      Performs time evolutions and computes the final density matrix.

    Parameters
    ----------
    par : Param
      Param instance containing the parameters of the simulation.
    fixed_potential : bool, optional
      If true, performs evolution with a static potential. By default False.
    compute_statistics : bool, optional
      If true, computes positions statistics. By default False.
    """
    # Apply split operator to wfcs
    self.split_op(par, fixed_potential, compute_statistics)
    rf = self.r_t[-1] 
    self.shifted_rho = density_matrix(self.probabilities, self.shifted_wfcs, par.x, self.num_wfcs, self.omega, rf) 