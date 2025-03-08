# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================================================
# WAVEFUNCTIONS PLOTTING
# ===========================================================================================================

def plot_position_statistics(par, opr, num_wfc):
  """
  plot_position_statistics :
    Plot the average position of the particle over time with ± σ_x uncertainty, 
    and an additional subplot for the drive function.

  Parameters
  ----------
  par : Param
    Param instance containing the parameters of the simulation.
  opr : Operators
    Operators instance containing the wavefunctions.
  num_wfc : int
    Wavefunction index to plot.
  """
  # Check if avg_pos has elements
  if not opr.avg_pos or len(opr.avg_pos) == 0:
    raise ValueError("Error: 'opr.avg_pos' is empty. Cannot plot position statistics.")
  
  # Compute time array and drive array
  time = np.linspace(0, par.total_time, len(opr.total_drive))
  drive = opr.total_drive  # Drive function over time

  # Extract average position and standard deviation
  positions = np.array(opr.avg_pos)  # Shape: (num_time_steps, num_wfcs)
  sigmas = np.array(opr.sigma_x)     # Shape: (num_time_steps, num_wfcs)

  avg_position = positions[:, num_wfc]
  sigma_x = sigmas[:, num_wfc]

  # Create figure with two subplots
  fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

  # Plot average position and uncertainty
  axs[0].plot(time, avg_position, label=r"Average position $\langle x(t) \rangle$", color="blue")
  axs[0].fill_between(time, avg_position - sigma_x, avg_position + sigma_x, color="orange", alpha=0.4, label=r"$\langle x \rangle \pm \sigma_x$")
  axs[0].set_ylabel("Position (x)")
  axs[0].set_title("Average Position Over Time")
  axs[0].legend()
  axs[0].grid(True)

  # Plot drive function
  axs[1].plot(time, drive, label="Drive function", color="red")
  axs[1].set_xlabel("Time (t)")
  axs[1].set_ylabel("Drive")
  axs[1].set_title("Drive Function Over Time")
  axs[1].legend()
  axs[1].grid(True)

  # Determine the common y-axis limits
  y_min = min(np.min(avg_position - sigma_x), np.min(drive))
  y_max = max(np.max(avg_position + sigma_x), np.max(drive))

  # Set the same y-axis limits for both subplots
  axs[0].set_ylim(y_min - 0.5, y_max + 0.5)
  axs[1].set_ylim(y_min - 0.5, y_max + 0.5)
  
  # Adjust layout and show plot
  plt.tight_layout()
  plt.show()

# ===========================================================================================================

def plot_wavefunctions(par, opr, shifted=False):
  """
  plot_wavefunctions :
    Plots wavefunctions.

  Parameters
  ----------
  par : _type_
    Param instance containing the parameters of the simulation.
  opr : _type_
    Operators instance containing the wavefunctions.
  shifted : bool, optional
    If true, plots shifted wavefunctions. By default False.
  """
  wfcs = opr.wfcs if not shifted else opr.shifted_wfcs
  for i in range(opr.num_wfcs):
    plt.plot(par.x, abs(wfcs[i])**2)
  
  plt.axvline(opr.r_t[0], color='blue', linestyle='--', label="Starting point")
  plt.axvline(opr.r_t[-1], color='red', linestyle='--', label="Ending point")
  plt.xlim(opr.r_t[-1] - 7, opr.r_t[-1] + 7)
  
  plt.grid()
  plt.legend()
  plt.show()
  
# ===========================================================================================================

def plot_optimization_process(fomlist, timegrid, pulse):
  """
  plot_optimization_process :
    Plots the optimization process showing the figure of merit (FoM) over iterations 
    and the pulse amplitude over time in a single figure with two subplots.

  Parameters
  ----------
  fomlist : list
    List of figure-of-merit (FoM) values over optimization iterations.
  timegrid : np.ndarray
    Time grid for the pulse.
  pulse : np.ndarray
    Pulse amplitude over time.
  """
  fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=False)

  # First subplot: FoM over function evaluations
  iterations = range(1, len(fomlist) + 1)
  axs[0].plot(iterations, np.asarray(fomlist), color='orange', linewidth=1.5, zorder=10)
  axs[0].set_yscale("log")
  axs[0].set_ylim(0.0008, 1.1)
  axs[0].grid(True, which="both")
  axs[0].set_xlabel('Iteration')
  axs[0].set_ylabel('FoM')
  axs[0].set_title("Optimization progress", fontsize=14)

  # Second subplot: Pulse amplitude over time
  axs[1].step(timegrid, pulse * 1e-4, color='darkred', linewidth=1.5, zorder=10)
  axs[1].grid(True, which="both")
  axs[1].set_xlabel('Time')
  axs[1].set_ylabel('Amplitude')
  axs[1].set_title("Optimized pulse", fontsize=14)

  # Adjust layout and show plot
  plt.tight_layout()
  plt.show()

    
# ===========================================================================================================
# TEMPERATURE ANALYSIS
# ===========================================================================================================

def load_data_temp(filename):
  """
  load_data_temp : 
    Load data from a text file.
  
  Parameters
  ----------
    filename : str
      The name of the file containing the data.
  
  Returns
  -------
    data_dict : dict
      A dictionary where keys are unique T values and values are arrays with tsim and avg_inf.
  """
  data = np.loadtxt(filename)
  data_dict = {}
  
  for T, tsim, avg_inf in data:
    if T not in data_dict:
      data_dict[T] = []
    data_dict[T].append((tsim, avg_inf))
  
  for T in data_dict:
    data_dict[T] = np.array(data_dict[T])
  
  return data_dict

# ===========================================================================================================

def plot_data_temp(data_dict):
  """
  Plot avg_inf as a function of tsim for different values of T.
  
  Args:
    data_dict (dict): Dictionary containing temperature data.
  """
  plt.figure(figsize=(8, 6))
  
  for T, values in sorted(data_dict.items()):
    tsim, avg_inf = values[:, 0], values[:, 1]
    plt.plot(tsim, avg_inf, marker='o', linestyle='-', label=f'T={T}')
  
  plt.axhline(y=0.01, c='b', linestyle='--')
  plt.xlabel("Simulation time")
  plt.ylabel("J_avg")
  plt.title("Figure of merit for different temperatures")
  plt.legend()
  plt.yscale('log')
  plt.grid()
  plt.show()