!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.3.a - TESTING PERFORMANCE
! Write explicitly the matrix-matrix multiplication loop in two different orders.

! Program Overview:
! - This program explicitly implementing two different loop orders to compute the 
! product of two square matrices.
! - The size of the matrices is defined as a constant parameter, set to 512.
! - Matrices A and B are initialized with all elements set to 1.0, while 
!   matrix C is initialized with all elements set to 0.0.
! - The program performs matrix multiplication in two ways:
!   1. Row-Column Order: The first loop structure multiplies matrix A and B,
!      storing the result in C.
!   2. Column-Row Order: The second multiplication resets C and computes 
!      the product with a different loop order.
! - After both multiplications, the program outputs a message indicating that 
!   the matrix multiplication process is completed.

program matrix_multiplication
    implicit none
    integer, parameter :: n = 512
    real :: A(n, n), B(n, n), C(n, n)
    integer :: i, j, k

    A = 1.0
    B = 1.0
    C = 0.0

    ! Order 1: C = A * B (standard row-column order)
    do i = 1, n
        do j = 1, n
            do k = 1, n
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    C = 0.0 ! Reset C for the second order multiplication

    ! Order 2: C = A * B (column-row order)
    do j = 1, n
        do i = 1, n
            do k = 1, n
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    print *, "Matrix multiplication completed."

end program matrix_multiplication
