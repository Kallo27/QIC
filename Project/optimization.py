# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import re
import os
import zipfile
import numpy as np

from quocslib.utils.AbstractFoM import AbstractFoM
from qho_time_evolution import Operators, Param


# ===========================================================================================================
# OPTIMALCONTROL CLASS
# ===========================================================================================================

class OptimalControl(AbstractFoM):
  """
  Class for optimal control.
  """
  def __init__(self, operator: Operators, par: Param):
    """
    __init__  :
      Initializes the OptimalControl instance.

    Parameters
    ----------
    operator : Operators
      Operators instance that will handle the system's operations and evolution.
    par : Param
      Param instance containing the parameters for the simulation.
    """
    self.opr = operator  # Pass in the Operators instance
    self.par = par       # Pass in the Param instance
    self.evolution_is_computed = False
    
  def t_ev(self, pulses_list, time_grids_list, parameter_list):
    """
    t_ev : 
      Performs time evolution of the system for the given pulses and time grids.

    Parameters
    ----------
    pulses_list : list
      List of pulses that will be applied during the evolution.
    time_grids_list : list
      List of time grids corresponding to the time steps for the evolution.
    parameter_list : list
      List of parameters for the time evolution, such as pulse characteristics.
    """
    time_grid = time_grids_list[0]
    self.par.dt = time_grid[-1] / len(time_grid)
    self.par.num_t = len(time_grid)
    
    self.opr.reinitialize_operators(self.par, pulses_list[0])
    
    # Perform the time evolution for all wavefunctions
    self.opr.time_evolution(self.par, fixed_potential=False)
    self.opr.time_evolution(self.par, fixed_potential=True)

    print(self.opr.energies)
    
    # Save flag
    self.evolution_is_computed = True
  
  def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
    """
    get_FoM :
      Computes and returns the figure of merit (FoM) for the given pulses, parameters, and time grids.

    Parameters
    ----------
    pulses : list, optional
      List of pulses to be used for the evolution. By default [].
    parameters : list, optional
      List of parameters to be used for the evolution. By default [].
    timegrids : list, optional
      List of time grids for the evolution. By default [].

    Returns
    -------
    FoM_dict : dict
      Dictionary containing the computed figure of merit (FoM).
    """

    if pulses is None:
      raise ValueError("Missing 'pulses' in get_FoM arguments")
    
    # Perform the time evolution
    if not self.evolution_is_computed:
      self.t_ev(pulses, timegrids, parameters)
    self.evolution_is_computed = False
    
    # Compute figure of merit
    FoM = self.opr.average_infidelity
    FoM_dict = {"FoM": FoM}
    
    return FoM_dict


# ===========================================================================================================
# LOADING FUNCTIONS
# ===========================================================================================================

def load_fom(timestamp):
  """
  load_fom :
    Loads FoM list from a specific optimization run.

  Parameters
  ----------
  timestamp : str
    Timestamp that identifies the optimization run.

  Returns
  -------
  fomlist
    List of FoM from a specific optimization run.
  """
  file_path = f'./QuOCS_Results/{timestamp}_OptimalControldCRAB/{timestamp}_logging.log'
  
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

  with open(file_path, 'r') as file:
    fomlist = []
    for line in file:
      # Search for lines that contain "FoM" and extract the FoM value
      match = re.search(r'FoM:\s([0-9\.]+)', line)
      if match:
        fomlist.append(float(match.group(1)))

  return fomlist

# ===========================================================================================================

def load_best_results(timestamp):
  """
  load_best_results :
    Loads best results from a specific optimization run.

  Parameters
  ----------
  timestamp : str
    Timestamp that identifies the optimization run.

  Returns
  -------
  timegrid, pulse : tuple of np.ndarray
    Timegrid and best pulse for the selected optimization run.
  """
  file_path = f'./QuOCS_Results/{timestamp}_OptimalControldCRAB/{timestamp}_best_controls.npz'
  
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

  with zipfile.ZipFile(file_path, 'r') as zip_ref:
    # List all files inside the archive
    file_names = zip_ref.namelist()
    print("Files inside the archive:", file_names)

    with zip_ref.open('time_grid_for_Pulse_1.npy') as time_file:
      timegrid = np.load(time_file)
    with zip_ref.open('Pulse_1.npy') as file:
      pulse = np.load(file)
    
  return timegrid, pulse