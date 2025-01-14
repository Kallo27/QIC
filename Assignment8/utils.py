###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 8 - RG and INFINITE DMRG


def mean_field(l_vals):
  """
  mean_field :
    Computes the mean field approximation as function of lambda.

  Parameters
  ----------
  l_vals : list of float
    Values of lambda.

  Returns
  -------
  val : list of float
    Mean field values.
  """
  val = {}
  
  for l in l_vals:
    if abs(l) <= 2:
      val[l] = - 1 - (l**2) / 4
    else:
      val[l] = - abs(l)
      
  return val