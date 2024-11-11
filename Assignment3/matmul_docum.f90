!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 3.1 - SCALING OF THE MATRIX-MATRIX MULTIPLICATION
! Consider the code developed in the Exercise 3 from Assignment 1 (matrix-matrix multiplication):
! (a) Write a python script that changes N between two values 'N_min' and 'N_max', and launches the program.
! (b) Store the results of the execution time in different files depending on the multiplication method used.
! (c) Fit the scaling of the execution time for different methods as a function of the input size. Consider
! the largest possible difference between 'N_min' and 'N_max'.
! (d) Plot results for different multiplication methods


program matrix_multiplication_timing
  ! Using debugger module and matmul_timing.
  use debugger
  use matmul_timing

  ! No implicitly declared variable allowed.
  implicit none

  ! Variable declaration.
  integer :: n

  ! Initial input: size of the square matrix (number of rows).
  read(*,*) n

  ! Call to the timing function from the matmul_timing module.
  call timing(n)

end program matrix_multiplication_timing
