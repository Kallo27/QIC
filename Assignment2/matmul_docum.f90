!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 2.2 - DOCUMENTATION
! Documentation: Rewrite Exercise 3 from Assignment 1 including:
! (a) Documentation.
! (b) Comments.
! (c) Pre- and post- conditions.
! (d) Error handling.
! (e) Checkpoints



program matrix_multiplication_timing
  ! Using debugger module and matmul_timing.
  use debugger
  use matmul_timing

  ! No implicitly declared variable allowed.
  implicit none

  ! Variable declaration.
  integer :: n

  ! Initial input: size of the square matrix (number of rows).
  print *, "Enter the maximum size of the matrices (n x n):"
  read *, n

  ! Call to the timing function from the matmul_timing module.
  call timing(n)

end program matrix_multiplication_timing
