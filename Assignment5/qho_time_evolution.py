###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 5 - TIME-DEPENDENT SCHROEDINGER EQUATION


# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import debugger as db
import auxiliary_functions as aux
import matplotlib.ticker as ticker

from matplotlib import animation


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
    self.im_time = im_time

    # Infinitesimal quantities (space, time, momentum)
    self.dx = (x_max - x_min) / num_x
    self.dt = tsim / num_t
    self.dk = 2 * np.pi / (x_max - x_min)

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
      db.checkpoint(debug=True, msg1="INITIALIZATION", msg2="ValueError: num_x and num_t must be positive integers.", stop=True)
    if self.x_max <= 0 or self.tsim <= 0:
      db.checkpoint(debug=True, msg1="INITIALIZATION", msg2="ValueError: xmax and tsim must be positive values.", stop=True)
      
      
# ===========================================================================================================
# OPERATORS CLASS
# ===========================================================================================================

class Operators:
  """
  Container for holding operators and wavefunction coefficients.
  """
  def __init__(self, 
               res: int, 
               voffset: float = 0, 
               wfcoffset: float = 0,
               omega: float = 1.0,
               order: int = 2,
               n: int = 0, 
               q0_func=None, 
               par: Param = None) -> None:
    """
    __init__: 
      Initialize operator arrays and configure time-dependent potential 
      and wavefunction.

    Parameters
    ----------
    res : int
      Resolution of the spatial domain (number of grid point).
    voffset : float, optional
      Offset of the quadratic potential in real space. Default is 0.
    wfcoffset : float, optional
      Offset of the wavefunction in real space. Default is 0.
    omega : float
      Angular frequency of the harmonic oscillator. Default is 1.0.
    order : int
      Order of the finite difference approximation. Default is 2.
    n : int, optional
      Order of the Hermite polynomial. Default is 0.
    q0_func : callable, optional
      A function q0_func(t) defining the time-dependent offset q0(t). 
      Default is None.
    par : Param, optional
      Instance of the Param class containing simulation parameters, 
      used to initialize grid-related quantities. Default is None.
    """
    # Initialize empty complex arrays for potential, propagators, and wavefunction
    self.V = np.empty(res, dtype=complex)  # Potential operator
    self.R = np.empty(res, dtype=complex)  # Real-space propagator
    self.K = np.empty(res, dtype=complex)  # Momentum-space propagator
    self.wfc = np.empty(res, dtype=complex)  # Wavefunction coefficients

    # Energy history list to track energy over time
    self.energy_history = []
    
    # Store time-dependent offset function (default to no potential if None)
    self.q0_func = q0_func or (lambda t: 0)
    
    # Store angular frequency
    self.omega = omega

    # Initialize potential and wavefunction if a Param instance is provided
    if par is not None:
      self._initialize_operators(par, voffset, wfcoffset, order, n)
      self.calculate_energy(par)

  def _initialize_operators(self, par: Param, voffset: float, wfcoffset: float, order: int, n: int) -> None:
    """
    _initialize_operators: 
      Initialize operators and wavefunction based on the provided parameters.

    Parameters
    ----------
    par : Param
      Simulation parameter instance containing grid and time information.
    voffset : float
      Offset of the quadratic potential in real space.
    wfcoffset : float
      Offset of the wavefunction in real space.
    order: int
      Order of the finite difference approximation
    n : int
      Order of the Hermite polynomial.
      
    Returns
    -----------
    None (acts in place).
    """
    # Initial time-dependent offset (at t=0)
    q0 = self.q0_func(0)

    # Quadratic potential with offset
    self.V = 0.5 * (par.x - voffset - q0) ** 2 * self.omega **2

    # Wavefunction based on a harmonic oscillator eigenstate
    #self.wfc = aux.harmonic_oscillator_spectrum(par.x - wfcoffset, self.omega, order, n).astype(complex)
    self.wfc = aux.harmonic_oscillator_spectrum(par.x - wfcoffset, self.omega, order, n).astype(complex)

    # Coefficient for imaginary or real time evolution
    coeff = 1 if par.im_time else 1j

    # Momentum and real-space propagators
    self.K = np.exp(-0.5 * (par.k ** 2) * par.dt * coeff)
    self.R = np.exp(-0.5 * self.V * par.dt * coeff)

  def calculate_energy(self, par: Param) -> float:
    """
    calculate_energy:
      Calculate the energy <Psi|H|Psi>.

    Parameters
    ----------
    par : Param
      Parameters of the simulation.

    Returns
    -------
    None (acts in place).
    """
    # Creating real, momentum, and conjugate wavefunctions.
    wfc_r = self.wfc
    wfc_k = np.fft.fft(wfc_r)
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k)
    energy_r = wfc_c * self.V * wfc_r

    # Integrating over all space (discrete sum weighted by grid spacing)
    energy_final = (sum(energy_k + energy_r).real) * par.dx

    # Store the energy in the history
    self.energy_history.append(energy_final)
    
# ===========================================================================================================
    
def split_op(par: Param, opr: Operators) -> None:
  """
  split_op :
    Split operator method for time evolution.

  Parameters
  ----------
  par : Param
    Parameters of the simulation.
  opr : Operators
    Operators of the simulation.

  Returns
  -------
  densities, potential, avg_position: tuple of np.ndarray
    Densities, potential and average position arrays, each with 100
    elements for visualization purposes.
  """
  # Initialize storage
  densities = np.zeros((100, 2 * par.num_x)) # 1st half -> wfc in real space; 2nd half -> wfc in momentum space
  potential = np.zeros((100, par.num_x))
  avg_position = np.zeros(100)
  
  # Initialize jj for gif visualization
  jj = 0
  
  # Loop over the number of timesteps
  for i in range(par.num_t):
    # Update the time-dependent potential V(x, t)
    q0 = opr.q0_func(i * par.dt)
    opr.V = 0.5 * (par.x - q0) ** 2 * opr.omega ** 2
    
    # Update the real space propagator
    coeff = 1 if par.im_time else 1j
    opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)

    # Half-step in real space
    opr.wfc *= opr.R

    # Full step momentum space
    opr.wfc = np.fft.fft(opr.wfc)
    opr.wfc *= opr.K
    opr.wfc = np.fft.ifft(opr.wfc)

    # Final half-step in real space
    opr.wfc *= opr.R

    # Density for plotting and potential
    density = np.abs(opr.wfc) ** 2

    # Renormalization
    if par.im_time:
      renorm_factor = np.sum(density * par.dx)
      if renorm_factor != 0.0:
        opr.wfc /= np.sqrt(renorm_factor)
        density = np.abs(opr.wfc) ** 2
      else:
        db.checkpoint(debug=True, msg1=f"RENORMALIZATION WARNING! Renorm factor too small at timestep {i}: {renorm_factor}", stop=False)

    # Saves exactly 100 snapshots
    if i % (par.num_t // 100) == 0 and jj < 100:
      # Save wfc in real and momentum space
      densities[jj, 0:par.num_x] = np.real(density)
      densities[jj, par.num_x:2 * par.num_x] = np.abs(np.fft.fft(opr.wfc)) ** 2
      
      # Save potential
      potential[jj, :] = opr.V
      
      # Save average position
      avg_position[jj] = np.sum(par.x * density) * par.dx

      # Update jj
      jj += 1

  return densities, potential, avg_position

# ===========================================================================================================
# VISUALIZATION
# ===========================================================================================================

def plot_average_position(par, avg_position):
  """
  plot_average_position :
    Plot the average position of the particle over time.

  Parameters
  ----------
  par : Param
    Parameters of the simulation.
  avg_position : np.ndarray
    Array of average position values.

  Returns
  -------
  None
  """
  # Compute time array (100 elements)
  time = np.linspace(0, par.tsim, 100)
  
  # Plot
  plt.figure(figsize=(8, 6))
  plt.plot(time, avg_position, label="Average position $\langle x(t) \\rangle$")
  plt.xlabel("Time (t)")
  plt.ylabel("Position (x)")
  plt.title("Average position of the particle over time")

  # Show
  plt.legend()
  plt.grid(True)
  plt.show()
  
# ===========================================================================================================

def gif_real_animation(par, density, potential, avg_position, filename='qho_time_evolution.gif'):
  """
  gif_real_animation:
    Creates an animated GIF showing the wave function, potential, and average position
    at each timestep.

  Parameters
  ----------
  par : Param
    Parameters of the simulation (contains spatial grid, etc.)
  density : numpy.ndarray
    Array of the wave function density at each timestep.
  potential : numpy.ndarray
    Array of the potential at each timestep.
  avg_position : numpy.ndarray
    Array of the average position of the particle at each timestep.
  filename : str, optional
    Name of the output GIF file. Default is 'qho_time_evolution.gif'.
  """
  # Set up the figure and axis for the animation
  fig, ax = plt.subplots()
  ax.set_xlim(par.x_min, par.x_max)
  ax.set_ylim(0, 1)
  ax.set_xlabel("Position (x)")
  ax.set_ylabel("Probability density |ψ(x)|^2")

  # Lines for the wave function, potential, and average position
  line_wfc, = ax.plot([], [], lw=2, label="|ψ(x)|^2", color='blue')
  line_pot, = ax.plot([], [], lw=2, label="V(x)", color='gray', linestyle='--')
  line_avg_pos, = ax.plot([], [], lw=1, label="Average position", color='red', linestyle='-.')

  ax.legend(loc="upper left")

  # Initialization function for the animation
  def init_line():
    """
    init_line:
      Initialization function for the animation.
    """
    line_wfc.set_data([], [])
    line_pot.set_data([], [])
    line_avg_pos.set_data([], [])
    return line_wfc, line_pot, line_avg_pos

  def animate(i):
    """
    animate:
      Animation function to update the plot at each frame.
    """
    # Extract density, potential and average position for frame i
    y_wfc = density[i, :par.num_x]
    y_pot = potential[i, :]
    avg_pos = avg_position[i]

    # Set data for the wave function and potential
    line_wfc.set_data(par.x, y_wfc)
    line_pot.set_data(par.x, y_pot)
    
    # Update the average position line
    line_avg_pos.set_data([avg_pos, avg_pos], [0, 1.5])  # Vertical line from y=0 to y=1.5

    return line_wfc, line_pot, line_avg_pos

  # Create the animation and save as a GIF
  anim = animation.FuncAnimation(fig, animate, init_func=init_line, frames=100, interval=10, blit=True)
  writer = animation.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
  anim.save(filename, writer=writer)

  # Close the figure to prevent it from displaying in the notebook
  plt.close(fig)
    
  # Final print statement
  print(f"Animation saved as '{filename}'")
  
# ===========================================================================================================

def gif_momentum_animation(par, density, filename='momentum_space.gif'):
  """
  gif_momentum_animation:
    Creates an animated GIF showing the wave function in momentum space at each timestep.

  Parameters
  ----------
  par : Param
    Parameters of the simulation (contains momentum grid, etc.).
  density : numpy.ndarray
    Array containing the wave function density in both real and momentum space
    at each timestep.
  filename : str, optional
    Name of the output GIF file. Default is 'momentum_space.gif'.
  """
  # Set up the figure and axis for the momentum space animation
  fig, ax = plt.subplots()
  ax.set_xlim(-10, 10)  # Set the x-axis limit for momentum space
  ax.set_ylim(density[:, par.num_x:2 * par.num_x].min(), density[:, par.num_x:2 * par.num_x].max())  # Dynamic y-axis limit
  ax.set_xlabel("Momentum (k)")
  ax.set_ylabel("Amplitude |ψ(k)|^2")

  # Enable scientific notation for the y-axis
  formatter = ticker.ScalarFormatter(useMathText=True)
  formatter.set_scientific(True)
  formatter.set_powerlimits((-2, 2))
  ax.yaxis.set_major_formatter(formatter)
  
  # Initialize the line for momentum space
  line, = ax.plot([], [], lw=1.5, label="|ψ(k)|^2", color='blue')
  ax.legend()

  def init_line():
    """
    init_line:
      Initialization function for the animation.
    """
    line.set_data([], [])
    return line,

  def animate(i):
    """
    animate:
      Animation function to update the plot at each frame.
    """
    # Extract momentum space density and sort for proper plotting
    x = par.k
    y = density[i, par.num_x:2 * par.num_x]
    order = np.argsort(x)  # Sort momentum values
    line.set_data(x[order], y[order])
    return line,

  # Create the animation and save as a GIF
  anim = animation.FuncAnimation(fig, animate, init_func=init_line, frames=100, interval=20, blit=True)
  writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  anim.save(filename, writer=writer)

  # Close the figure to prevent it from displaying in the notebook
  plt.close(fig)

  # Final print statement
  print(f"Animation saved as '{filename}'")
