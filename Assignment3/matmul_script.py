###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 3.1 - SCALING OF THE MATRIX-MATRIX MULTIPLICATION
# Consider the code developed in the Exercise 3 from Assignment 1 (matrix-matrix multiplication):
# (a) Write a python script that changes N between two values 'N_min' and 'N_max', and launches the program.
# (b) Store the results of the execution time in different files depending on the multiplication method used.
# (c) Fit the scaling of the execution time for different methods as a function of the input size. Consider
# the largest possible difference between 'N_min' and 'N_max'.
# (d) Plot results for different multiplication methods

"""
USAGE:
  python script.py Nmin Nmax m
    - Nmin: The minimum matrix size.
    - Nmax: The maximum matrix size.
    - m: The number of matrix sizes to test between Nmin and Nmax.

EXAMPLES:
  To run the script with Nmin=100, Nmax=1000, and m=10:
    python script.py 100 1000 10
    
"""

# ============================================================================================


# IMPORT ZONE
import subprocess
import sys
import os

# ============================================================================================

# FUNCTIONS

def compile_module(module_file):
  """
  compile_module:
    Compiles a Fortran module using gfortran.
  
  Parameters
  ----------
  module_file : str
    The name of the Fortran module file to compile.
  
  Returns
  -------
  bool
    True if the compilation is successful, False otherwise.
  """
  try:
    print(f'Compiling {module_file}...')
    # Runs module files needed for the compilation of the fortran program.
    subprocess.run(['gfortran', module_file, '-c'], check=True)
    print('Compilation successful!')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

# ============================================================================================

def compile_fortran(source_file, exec_name, *object_files):
  """
  Compiles the main Fortran source file along with any provided object files.

  Parameters
  ----------
  source_file : str
    The main Fortran source file to compile.
  exec_name : str
    The name of the executable to generate.
  object_files : list of str
    List of object files for additional modules.

  Returns
  -------
  bool
    True if the compilation is successful, False otherwise.
  """
  try:
    print(f'Compiling {source_file} with additional modules: {object_files}...')
    # Compiles the fortran program (optimization -O3) + the object files of the modules
    command = ['gfortran', '-O3', '-o' , exec_name, source_file] + list(object_files)
    subprocess.run(command, check=True)
    print('Compilation successful!')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

# ============================================================================================

def run_executable(exec_name, input_data):
  """
  Runs the Fortran executable with the specified input data and captures output.
  
  Parameters
  ----------
  exec_name : str
    The name of the compiled executable to run.
  input_data : str
    The input data to pass to the executable.

  Returns
  -------
  str or None
    The captured output from the program if successful, or None if an error occurs.
  """
  try:
    print(f'Running {exec_name} with input: {input_data.strip()}')
    # Runs the executable, with inputs from command line
    result = subprocess.run([f'./{exec_name}'], input=input_data, text=True, capture_output=True, shell=True, check=True)
    print('Execution successful.')
    return result.stdout
  except subprocess.CalledProcessError as e:
    print(f'Error during execution: {e}')
    return None

# ============================================================================================

def save_to_file(filename, data):
  """
  Writes the provided data to a file.

  Parameters
  ----------
  filename : str
    The name of the file to save the data to.
  data : str
    The data to be written to the file.

  Returns
  -------
  None
  """
  with open(filename, 'a') as file:
    file.write(data)

# ============================================================================================

def main(Nmin, Nmax, m):
  """
  Loops through m evenly spaced values between Nmin and Nmax, launching 
  the Fortran program with each value of N, capturing the output, and 
  extracting the elapsed times for different multiplication methods. The 
  results are saved to separate files (one for each method).
  
  Parameters
  ----------
  Nmin : int
    The minimum value of matrix size N.
  Nmax : int
    The maximum value of matrix size N.
  m : int
    The number of different N values to test between Nmin and Nmax.

  Returns
  -------
  None
  """
  if not os.path.exists('Data'):
    os.makedirs('Data')
    
  for i in range(m):
    n = Nmin + i * (Nmax - Nmin) / (m - 1)
    output = run_executable(executable_name, f'{int(n)}\n')
    if output:
      elapsed_time1 = None
      elapsed_time2 = None
      elapsed_time3 = None

      # Collects outputs and saves in different vectors for the three methods.
      for line in output.splitlines():
        if "RC" in line:
          elapsed_time1 = line + "\n"
        elif "CR" in line:
          elapsed_time2 = line + "\n"
        elif "I" in line:
          elapsed_time3 = line + "\n"

      # Saves the vectors in different files for the three methods.
      if elapsed_time1:
        save_to_file(f'Data/rowbycolumn.txt', elapsed_time1)
      if elapsed_time2:
        save_to_file(f'Data/columnbyrow.txt', elapsed_time2)
      if elapsed_time3:
        save_to_file(f'Data/intrinsic.txt', elapsed_time3)

# ============================================================================================

# MAIN (when runned as 'main')
if __name__ == '__main__':
  # Pre-conditions: checks the correct number of inputs.
  if len(sys.argv) != 4:
    print('Usage: python script.py Nmin Nmax m')
    sys.exit(1)

  # Pre-conditions: checks if the value of N are not positive or if Nmax < Nmin.
  try:
    Nmin = int(sys.argv[1])
    Nmax = int(sys.argv[2])
    m = int(sys.argv[3])

    if Nmin <= 0 or Nmax <= 0 or m <= 0:
      print('All inputs must be greater than 0.')
      sys.exit(1)
    if Nmin >= Nmax:
      print('Nmin must be less than Nmax.')
      sys.exit(1)

  except ValueError:
    print('Please enter valid integers for Nmin, Nmax, and m.')
    sys.exit(1)

  fortran_file = 'matmul_docum.f90'
  executable_name = 'matmul_docum.x'
  modules = ['debugger.f90', 'matmul_timing.f90']
  object_files = ['debugger.o', 'matmul_timing.o']

  if compile_module(modules[0]) and compile_module(modules[1]):
    if compile_fortran(fortran_file, executable_name, *object_files):
      main(Nmin, Nmax, m)
  
  print("Finished saving on files")
  
# ============================================================================================