!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.3.b - TESTING PERFORMANCE
! Use the FORTRAN intrinsic function.

! Program Overview:
! - This program implements matrix-matrix multiplication using the Fortran
! 'matmul' intrinsic function.
! - The size of the matrices is defined as a constant parameter, set to 512.
! - Matrices A and B are initialized with all elements set to 1.0, while 
!   matrix C is initialized using the matmul function directly.
! - After the multiplication, the program outputs a message indicating that 
!   the matrix multiplication process is completed.

program matrix_multiplication_intrinsic
    implicit none
    integer, parameter :: n = 512
    real :: A(n, n), B(n, n), C(n, n)

    A = 1.0
    B = 1.0

    C = matmul(A, B)

    print *, "Matrix multiplication using intrinsic function completed."
end program matrix_multiplication_intrinsic
