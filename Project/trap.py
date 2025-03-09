# ===========================================================================================================
# IMPORT ZONE
# ===========================================================================================================

import numpy as np
from pytrans.electrode import DCElectrode, RFElectrode
from pytrans.abstract_model import AbstractTrapModel
from rectset import rectangle_electrode as rect, pseudopotential as ps

from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation

import json

# ===========================================================================================================
# TRAP GEOMETRY PARAMETERS
# ===========================================================================================================

um = 1e-6
rf_width = 120 * um
rf_sep = 60 * um

# 5 segmented DC electrodes
n_dc_lines = 5
dc_width = 100 * um

dc_length = 1300 * um
trap_length = 3000 * um

filename = "surface_trap_geometry.json"

# Build corners and edges for the electrodes
corners = {}
corners['DCintop'] = (-trap_length / 2, trap_length / 2, 0, rf_sep / 2)
corners['DCinbot'] = (-trap_length / 2, trap_length / 2, -rf_sep / 2, 0)

dc_edges = [
    (dc_width * (j - n_dc_lines / 2), dc_width * (j + 1 - n_dc_lines / 2))
    for j in range(n_dc_lines)
]

corners.update({
    f"DCtop{j + 1}": dc_edges[j] + ((rf_sep / 2 + rf_width), (rf_sep / 2 + rf_width + dc_length))
    for j in range(n_dc_lines)
})

corners.update({
    f"DCbot{j + 1}": dc_edges[j] + (-(rf_sep / 2 + rf_width + dc_length), -(rf_sep / 2 + rf_width))
    for j in range(n_dc_lines)
})

_DC_ELECTRODES = [
    "DCintop", "DCinbot",
    "DCtop1", "DCtop2", "DCtop3", "DCtop4", "DCtop5",
    "DCbot1", "DCbot2", "DCbot3", "DCbot4", "DCbot5",
]


# ===========================================================================================================
# SURFACE TRAP CLASS
# ===========================================================================================================

class SurfaceTrapDCElectrode(DCElectrode):
  """
  Represents a rectungular DC electrode in a surface ion trap.
  """
  def __init__(self, name):
    """
    __init__ :
      Initializes the DC electrode with predefined corner positions.

    Parameters
    ----------
    name : str
      The name of the electrode, used to retrieve its corner positions.
    """
    self._corners = corners[name]
    super().__init__()

  def _unit_potential(self, x, y, z):
    """
    _unit_potential : 
      Computes the potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    potential: float
      Electrostatic potential at (x, y, z).
    """
    potential = rect.rect_el_potential(x, y, z, *self._corners)
    return potential 
  
  def _unit_gradient(self, x, y, z):
    """
    _unit_gradient : 
      Computes the gradient of the potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    gradient : tuple of float
      Gradient components (∂V/∂x, ∂V/∂y, ∂V/∂z).
    """
    gradient = rect.rect_el_gradient(x, y, z, *self._corners)
    return gradient

  def _unit_hessian(self, x, y, z):
    """
    _unit_hessian : 
      Computes the Hessian matrix of the potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    hessian: tuple of float
      Second derivatives (∂²V/∂x², ∂²V/∂y², ∂²V/∂z², ...).
    """
    hessian = rect.rect_el_hessian(x, y, z, *self._corners)
    return hessian

# ===========================================================================================================

class SurfaceTrapRFElectrode(RFElectrode):
  """
  Represents an RF electrode in a surface ion trap.
  """
  def _unit_potential(self, x, y, z):
    """
    _unit_potential : 
      Computes the pseudo-potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    pseudopotential : float
      Pseudo-potential at (x, y, z).
    """
    pseudopotential = ps.pseudo_potential(x, y, z, rf_sep, rf_width)
    return pseudopotential

  def _unit_gradient(self, x, y, z):
    """
    _unit_gradient : 
      Computes the gradient of the pseudo-potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    ps_gradient : tuple of float
      Gradient components (∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z).
    """
    ps_gradient = ps.pseudo_gradient(x, y, z, rf_sep, rf_width)
    return ps_gradient

  def _unit_hessian(self, x, y, z):
    """
    _unit_hessian : 
      Computes the Hessian matrix of the pseudo-potential at a given point.

    Parameters
    ----------
    x, y, z : float
      Spatial coordinates.

    Returns
    -------
    ps_hessian : tuple of float
      Second derivatives (∂²Φ/∂x², ∂²Φ/∂y², ∂²Φ/∂z², ...).
    """
    ps_hessian = ps.pseudo_hessian(x, y, z, rf_sep, rf_width)
    return ps_hessian

# ===========================================================================================================

class SurfaceTrap(AbstractTrapModel):
  """
  Represent a SurfaceTrap model, with DC and RF electrodes (5 wire).
  """
  _dc_electrodes = {name: SurfaceTrapDCElectrode(name) for name in _DC_ELECTRODES}
  _rf_electrodes = {"RF": SurfaceTrapRFElectrode()}
  _rf_voltage = 40
  _rf_freq_mhz = 20

  # Extra attributes and methods to enrich the model
  w_ele = dc_width
  x = np.arange(-250, 251, 0.5) * 1e-6
  y0 = 0.0
  z0 = ps.rf_null_z(rf_sep, rf_width)
  rf_null_coords = (None, y0, z0)
  dt = 392e-9

  @classmethod
  def x_ele(cls, j):
    "Returns centered position in element j."
    return cls.w_ele * (j - 3)


def scale_pulse(pulse, x_start, x_end, method="linear", k=3):
  """
  scale_pulse :
    Scale an input array `pulse` to the range [x_start, x_end].

  Parameters
  ----------
  pulse : np.ndarray
    Input pulse array to scale.
  x_start : float
    Start of the target range.
  x_end : float
    End of the target range.
  method : str, optional
    "linear" for linear scaling, "sigmoid" for sigmoid-based scaling. By default "linear".
  k : int, optional
    Smoothness parameter for the sigmoid function. By default 3.

  Returns
  -------
  final_pulse : np.ndarray
    Scaled array in the range [x_start, x_end].
  """
  if method == "linear":
    pulse_min = np.min(pulse)
    pulse_max = np.max(pulse)
    pulse_scaled = (pulse - pulse_min) / (pulse_max - pulse_min)  # Normalize to [0, 1]
  
  elif method == "sigmoid":
    pulse_norm = (pulse - np.mean(pulse)) / np.std(pulse)  # Normalize to mean 0, std 1
    pulse_scaled = 1 / (1 + np.exp(-k * pulse_norm))  # Apply sigmoid
  
  else:
    raise ValueError(f"Invalid method={method}. Choose 'linear' or 'sigmoid'.")
  
  final_pulse = x_start + pulse_scaled * (x_end - x_start)
  
  return final_pulse


# ===========================================================================================================
# VISUALIZATION FUNCTIONS
# ===========================================================================================================

plt.rcParams['font.size'] = 9
electrodes_data = './surface_trap_geometry.json'

with open(electrodes_data, 'r') as f:
  electrodes = json.load(f)

all_patches = {}
for name, ele in electrodes.items():
  path = Path.make_compound_path(*[Path(ring) for ring in ele])
  patch = PathPatch(path)
  all_patches[name] = patch

labels_x = np.asarray(dc_edges).mean(axis=1) * 1e6
el_y = (rf_sep / 2 + rf_width) * 1e6
dy = 50

labels_positions = [(300, 50)] + [(300, -50)] + \
    [(labels_x[j], el_y + dy) for j in range(n_dc_lines)] + \
    [(labels_x[j], -el_y - dy) for j in range(n_dc_lines)]

# ===========================================================================================================

def find_ylim(a, r=0.05):
  """
  find_ylim : 
    Computes the y-axis limits for plotting by adding a margin.

  Parameters
  ----------
  a : array-like
    Input data to determine limits.
  r : float, optional
    Fractional margin to add to the range, by default 0.05

  Returns
  -------
  limits = tuple
    Lower and upper y-axis limits.
  """
  _min = np.min(a)
  _max = np.max(a)
  ptp = _max - _min
  limits = _min - r * ptp, _max + r * ptp
  
  return limits

# ===========================================================================================================

def make_plot_on_trap_layout(n: int = 1, fig=None, figsize1=(6, 3.2)):
  """
  make_plot_on_trap_layout :
    Creates a figure layout for plotting trap data.

  Parameters
  ----------
  n : int, optional
    Number of subfigures, by default 1
  fig : matplotlib.figure.Figure, optional
    Existing figure to use, by default None
  figsize1 : tuple, optional
    Base figure size, by default (6, 3.2)

  Returns
  -------
  fig, axes : tuple
    Figure and axes for the plots.
  """
  if fig is None:
    fig = plt.figure(figsize=(figsize1[0] * n, figsize1[1]), dpi=100)
  
  _subplots_kw = dict(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1], top=0.95, bottom=0.14))
  if n > 1:
    subfigs = fig.subfigures(1, n)
    axes = [subfigs[j].subplots(**_subplots_kw) for j in range(n)]
  else:
    axes = fig.subplots(**_subplots_kw)
  
  return fig, axes

# ===========================================================================================================

def _setup_plot_on_trap(trap: SurfaceTrap, axes=None, vmin=-10, vmax=10, cmap='RdBu_r',
                        edgecolor='k', linewidth=0.5, fontsize=7, title=''):
  """
  _setup_plot_on_trap :
    Initializes a plot for visualizing a trap layout.

  Parameters
  ----------
  trap : SurfaceTrap
    Trap object containing electrode and potential information.
  axes : tuple, optional
    Axes for plotting. By default None.
  vmin : int, optional
    Minimum voltage for color scale. By default -10.
  vmax : int, optional
    Maximum voltage for color scale. By default 10.
  cmap : str, optional
    Colormap for visualization. By default 'RdBu_r'.
  edgecolor : str, optional
    Color of electrode edges. By default 'k'.
  linewidth : float, optional
    Width of electrode edges. By default 0.5.
  fontsize : int, optional
    Font size for labels. By default 7.
  title : str, optional
    Title for the figure. By default ''.

  Returns
  -------
  fig, axes, artists : tuple
    Figure, axes and plotted elements.
  """
  if axes is None:
    fig, axes = make_plot_on_trap_layout(1)  # Create figure if not provided
  else:
    fig = axes[0].figure

  fig.suptitle(title)
  ax0, ax = axes  

  nv = trap.n_electrodes  
  vzeros = np.zeros((nv,))  

  # Get electrode patches and non-electrode background patches
  patches = [all_patches[name] for name in trap._all_electrodes]
  patches_blank = set(all_patches.values()) - set(patches)

  # Set colormap for voltage visualization
  cmap = plt.colormaps[cmap]  
  norm = Normalize(vmin=vmin, vmax=vmax)  
  colors = cmap(norm(vzeros))  

  # Plot electrodes with initial zero voltage
  pc = PatchCollection(patches, facecolor=colors, edgecolor=edgecolor, 
                       linewidth=linewidth, cmap=cmap, norm=norm)
  ax.add_collection(pc)  

  # Add text labels to electrodes
  poss = [labels_positions[j] for j in trap.electrode_all_indices]  
  labels = [ax.text(*pos, f"{0:+.2f}", ha='center', va='center', fontsize=fontsize) 
            for pos in poss] if fontsize > 0 else []

  # Outline non-electrode areas
  pc_rf = PatchCollection(patches_blank, facecolor='none', edgecolor=edgecolor, 
                          linewidth=linewidth, cmap=cmap, norm=norm)
  ax.add_collection(pc_rf)  

  ttime = ax.text(0.9, 0.85, '', transform=ax0.transAxes)  

  ax.set(
        xlim=(-350, 350),
        ylim=(-400, 400),
        xticks=np.arange(-300, 301, 100),
        yticks=np.arange(-400, 401, 400),
        xlabel='x [um]',
        ylabel='y [um]',
  )

  # Plot initial potential as a zero baseline
  p0 = np.zeros_like(trap.x)  
  pot, = ax0.plot(trap.x * 1e6, p0)  

  # Clean up subplot appearance
  ax0.spines['right'].set_visible(False)  
  ax0.spines['top'].set_visible(False)  
  ax0.set(ylabel='$\\phi$ [eV]')  

  fig.align_ylabels()  

  artists = (pc, pot, ttime, tuple(labels))
  return fig, axes, artists

# ===========================================================================================================

def plot_voltages_on_trap(trap: SurfaceTrap, voltages: ArrayLike, axes=None, vmin=-10, vmax=10, cmap='RdBu_r',
                          edgecolor='k', linewidth=0.5, fontsize=7, title=''):
  """
  plot_voltages_on_trap :
    Visualizes the applied voltages on a surface trap.

  Parameters
  ----------
  trap : SurfaceTrap
    Trap object containing electrode and potential information.
  voltages : ArrayLike
    Array of voltages applied to the electrodes.
  axes : tuple, optional
    Axes for plotting. By default None.
  vmin : int, optional
    Minimum voltage for color scale. By default -10.
  vmax : int, optional
    Maximum voltage for color scale. By default 10.
  cmap : str, optional
    Colormap for visualization. By default 'RdBu_r'.
  edgecolor : str, optional
    Color of electrode edges. By default 'k'.
  linewidth : float, optional
    Width of electrode edges. By default 0.5.
  fontsize : int, optional
    Font size for labels. By default 7.
  title : str, optional
    Title for the figure. By default ''.

  Returns
  -------
  fig, axes : tuple
    Figure and axes with the plotted voltages.
  """
  # Setting up the plot with given parameters
  fig, axes, artists = _setup_plot_on_trap(trap, axes, vmin, vmax, cmap,
                                           edgecolor, linewidth, fontsize, title)

  # Unpacking the axes and artists for easier reference
  ax0, ax = axes
  pc, pot, ttime, labels = artists

  # Calculating the potential at the given voltages and trap coordinates
  potential = trap.potential(voltages, trap.x, 0, trap.z0, 1, pseudo=False)

  # Get the colormap and normalization for the voltage range
  cmap = plt.colormaps[cmap]
  norm = Normalize(vmin=vmin, vmax=vmax)

  # Assign colors based on the voltage values
  colors = cmap(norm(voltages))
  pc.set_facecolor(colors)

  # Update the potential plot with the calculated potential values
  pot.set_ydata(potential)
  pot.axes.relim()
  pot.axes.autoscale_view()

  # Update text labels with voltages and adjust color for readability
  for volt, t in zip(voltages, labels):
    t.set_text(f"{volt:+.2f}")
    t.set_color('k' if abs(volt) < 8 else 'w')

  # Return the updated figure and axes
  return fig, axes

# ===========================================================================================================

def animate_waveform_on_trap(trap: SurfaceTrap, waveform: ArrayLike, axes=None, vmin=-10, vmax=10, cmap='RdBu_r',
                             edgecolor='k', linewidth=0.5, fontsize=7, title='',
                             frames=None, animate_kw=dict()):
  """
  animate_waveform_on_trap :
    Creates a gif of a waveform applied to a surface trap.

  Parameters
  ----------
  trap : SurfaceTrap
    Trap object containing electrode and potential information.
  waveform : ArrayLike
    Time-dependent voltages applied to the electrodes.
  axes : tuple, optional
    Axes for plotting. By default None.
  vmin : int, optional
    Minimum voltage for color scale. By default -10.
  vmax : int, optional
    Maximum voltage for color scale. By default 10.
  cmap : str, optional
    Colormap for visualization. By default 'RdBu_r'.
  edgecolor : str, optional
    Color of electrode edges. By default 'k'.
  linewidth : float, optional
    Width of electrode edges. By default 0.5.
  fontsize : int, optional
    Font size for labels. By default 7.
  title : str, optional
    Title for the figure. By default ''.
  frames : int, optional
    Number of frames in the animation. By default None.
  animate_kw : dict, optional
    Additional keyword arguments for animation. By default dict().

  Returns
  -------
  ani : matplotlib.animation.FuncAnimation
      Animation object for waveform evolution.
  """
  fig, axes, artists = _setup_plot_on_trap(trap, axes, vmin, vmax, cmap,
                                             edgecolor, linewidth, fontsize, title)
  ax0, ax = axes
  pc, pot, ttime, labels = artists

  potentials = trap.potential(waveform, trap.x, 0, trap.z0, 1, pseudo=False)
  cmap = plt.colormaps[cmap]
  norm = Normalize(vmin=vmin, vmax=vmax)

  def init():
    ax0.set_ylim(find_ylim(potentials))
    return (pc, pot, ttime,) + labels

  def update(j):
    colors = cmap(norm(waveform[j]))
    pc.set_facecolor(colors)
    pot.set_ydata(potentials[j])
    pot.axes.relim()
    pot.axes.autoscale_view()
  
    for v, t in zip(waveform[j], labels):
      t.set_text(f"{v:+.2f}")
      t.set_color('k' if abs(v) < 8 else 'w')
    ttime.set_text(str(j))
  
    return (pc, pot, ttime,) + labels

  kw = dict(blit=True, interval=20, repeat_delay=2000)
  kw.update(animate_kw)

  frames = range(len(waveform)) if frames is None else frames
  ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, **kw)
  
  return ani
