import numpy as np

T = 15

def shape_function(t):
  return 3 * (1 / (1 + np.exp(-(t - T / 2))))

def scaling_function(t):
	return 3 * t / T

def initial_guess_function(t):
  return 3 * t / T
