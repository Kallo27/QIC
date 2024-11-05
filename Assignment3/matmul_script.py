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
  FUNCTIONS:

  compile_module(module_file)
          Inputs   | module_file (str): The name of the Fortran module file to compile.
          Purpose  | Compiles a Fortran module using gfortran.
          Outputs  | Returns True if the compilation is successful, False otherwise.

  compile_fortran(source_file, exec_name, *object_files)
          Inputs   | source_file (str): The main Fortran source file to compile.
                   | exec_name (str): The name of the executable to generate.
                   | object_files (list of str): List of object files for additional modules.
          Purpose  | Compiles the main Fortran source file along with any provided object 
                   | files, using gfortran to create an executable.
          Outputs  | Returns True if the compilation is successful, False otherwise.

  run_executable(exec_name, input_data)
          Inputs   | exec_name (str): The name of the compiled executable to run.
                   | input_data (str): The input data to pass to the executable.
          Purpose  | Runs the Fortran executable with the specified input data, capturing 
                   | the output. If the execution fails, an error is displayed.
          Outputs  | Returns the captured output from the program if successful, or None 
                   | if an error occurs.

  save_to_file(filename, data)
          Inputs   | filename (str): The name of the file to save the data to.
                   | data (str): The data to be written to the file.
          Purpose  | Writes the provided data to a file; if successful, a confirmation 
                   | message is printed.
          Outputs  | None

  main(Nmin, Nmax, m)
          Inputs   | Nmin (int): The minimum value of matrix size N.
                   | Nmax (int): The maximum value of matrix size N.
                   | m (int): The number of different N values to test between Nmin and Nmax.
          Purpose  | Loops through m evenly spaced values between Nmin and Nmax, launching 
                   | the Fortran program with each value of N, capturing the output, and 
                   | extracting the elapsed times for different multiplication methods. The 
                   | results are saved to separate files (one for each method).
          Outputs  | None

  PRECONDITIONS AND VALIDATIONS:

  - The script validates that all inputs (Nmin, Nmax, and m) are positive integers and that 
    Nmin is less than Nmax, otherwise throws an error.

  USAGE:
    python script.py Nmin Nmax m

    - Nmin: The minimum matrix size.
    - Nmax: The maximum matrix size.
    - m: The number of matrix sizes to test between Nmin and Nmax.

  EXAMPLES:
    To run the script with Nmin=100, Nmax=1000, and m=10:
    python script.py 100 1000 10
    
"""

# IMPORT ZONE
import subprocess
import sys
import os

# ============================================================================================

# FUNCTIONS

# Compile the Fortran code
def compile_module(module_file):
  try:
    print(f'Compiling {module_file}...')
    subprocess.run(['gfortran', module_file, '-c'], check=True)
    print('Compilation successful!')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

# Compile a fortran program with object files,
def compile_fortran(source_file, exec_name, *object_files):
  try:
    print(f'Compiling {source_file} with additional modules: {object_files}...')
    
    # Create the command with source file and object files
    command = ['gfortran', '-O3', '-o' , exec_name, source_file] + list(object_files)
    subprocess.run(command, check=True)
    print('Compilation successful!')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

# Run the executable with input data
def run_executable(exec_name, input_data):
  try:
    print(f'Running {exec_name} with input: {input_data.strip()}')
    result = subprocess.run([f'./{exec_name}'], input=input_data, text=True, capture_output=True, shell=True, check=True)
    print('Execution successful.')
    
    # Return the output from the program
    return result.stdout
  except subprocess.CalledProcessError as e:
    print(f'Error during execution: {e}')
    return None

# Function to save data to a file
def save_to_file(filename, data):
  with open(filename, 'a') as file:
    file.write(data)

# Main function to run the program for different values of n
def main(Nmin, Nmax, m):
  # Check if directory Data exists, otherwise mkdir.
  if not os.path.exists('Data'):
    os.makedirs('Data')
    
  # Calculate the fraction increments
  for i in range(m):
    # Calculate n based on the current division
    n = Nmin + i * (Nmax - Nmin) / (m - 1)
    output = run_executable(executable_name, f'{int(n)}\n')
    if output:
      # Extract and save elapsed times from the output
      elapsed_time1 = None
      elapsed_time2 = None
      elapsed_time3 = None

      for line in output.splitlines():
        if "RC" in line:
          elapsed_time1 = line + "\n"
        elif "CR" in line:
          elapsed_time2 = line + "\n"
        elif "I" in line:
          elapsed_time3 = line + "\n"

      # Save each elapsed time to separate files
      if elapsed_time1:
        save_to_file(f'Data/rowbycolumn.txt', elapsed_time1)
      if elapsed_time2:
        save_to_file(f'Data/columnbyrow.txt', elapsed_time2)
      if elapsed_time3:
        save_to_file(f'Data/intrinsic.txt', elapsed_time3)

# ============================================================================================

# MAIN
if __name__ == '__main__':
  # Check command line arguments
  if len(sys.argv) != 4:
    print('Usage: python script.py Nmin Nmax m')
    sys.exit(1)

  try:
    Nmin = int(sys.argv[1])
    Nmax = int(sys.argv[2])
    m = int(sys.argv[3])

    # Validate inputs
    if Nmin <= 0 or Nmax <= 0 or m <= 0:
      print('All inputs must be greater than 0.')
      sys.exit(1)
    if Nmin >= Nmax:
      print('Nmin must be less than Nmax.')
      sys.exit(1)

  except ValueError:
    print('Please enter valid integers for Nmin, Nmax, and m.')
    sys.exit(1)

  # Define Fortran files and executable name
  fortran_file = 'matmul_docum.f90'
  executable_name = 'matmul_docum.x'
  modules = ['debugger.f90', 'matmul_timing.f90']
  object_files = ['debugger.o', 'matmul_timing.o']

  # Compile the Fortran modules and program
  if compile_module(modules[0]) and compile_module(modules[1]):
    if compile_fortran(fortran_file, executable_name, *object_files):
      # Run the main function with given Nmin, Nmax, and m
      main(Nmin, Nmax, m)
  
  print("Finished saving on files")