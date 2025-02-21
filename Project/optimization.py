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