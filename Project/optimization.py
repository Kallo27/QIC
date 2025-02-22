import re
import os
import zipfile
import numpy as np

from quocslib.utils.AbstractFoM import AbstractFoM
from qho_time_evolution import Operators, Param

class OptimalControl(AbstractFoM):
  def __init__(self, operator: Operators, par: Param):
    self.opr = operator  # Pass in the Operators instance
    self.par = par       # Pass in the Param instance
    self.evolution_is_computed = False
        
  def t_ev(self, pulses_list, time_grids_list, parameter_list):
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
    if pulses is None:
      raise ValueError("Missing 'pulses' in get_FoM arguments")
    
    # Perform the time evolution
    if not self.evolution_is_computed:
      self.t_ev(pulses, timegrids, parameters)
    self.evolution_is_computed = False
    
    # Compute figure of merit
    FoM = self.opr.average_infidelity
    
    return {"FoM": FoM}



def load_fom(timestamp):
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

  return fomlist  # Don't forget to return the list!

def load_best_results(timestamp):
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