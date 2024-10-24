!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.3.a - TESTING PERFORMANCE
! Increase the matrix size and track the code performance using FORTRAN basic 
! date and time routines (i.e. CPU TIME)

! Program Overview:
! - This program performs matrix multiplication using three different methods:
!   1. Standard row-column order.
!   2. Column-row order.
!   3. FORTRAN intrinsic function 'matmul'.
! - It allows the user to input the size of the matrices (n x n) at runtime.
! - The program tracks the elapsed time for each multiplication method using the 
!   FORTRAN intrinsic subroutine 'cpu_time', which measures CPU time.
! - After each multiplication, the program resets the result matrix C to zero 
!   before performing the next multiplication.


program matrix_multiplication_timing
    implicit none
    integer :: n
    real, allocatable :: A(:, :), B(:, :), C(:, :)
    integer :: i, j, k
    real :: start_time, end_time, elapsed_time


    print *, "Enter the size of the matrices (n x n):"
    read *, n

    ! Allocate memory for the matrices
    allocate(A(n, n), B(n, n), C(n, n))

    ! Initialize matrices A and B
    A = 1.0
    B = 1.0
    C = 0.0

    call cpu_time(start_time)

    ! Order 1: C = A * B (standard row-column order)
    do i = 1, n
        do j = 1, n
            do k = 1, n
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    call cpu_time(end_time)

    elapsed_time = end_time - start_time
    print *, "Elapsed time for matrix multiplication (order 1): ", elapsed_time, "s"


    C = 0.0

    call cpu_time(start_time)

    ! Order 2: C = A * B (column-row order)
    do j = 1, n
        do i = 1, n
            do k = 1, n
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    call cpu_time(end_time)

    elapsed_time = end_time - start_time
    print *, "Elapsed time for matrix multiplication (order 2): ", elapsed_time,  "s"


    C = 0.0

    call cpu_time(start_time)

    ! Intrinsic function
    C = matmul(A, B)

    call cpu_time(end_time)

    elapsed_time = end_time - start_time
    print *, "Elapsed time for matrix multiplication (intrinsic-function): ", elapsed_time, "s"

end program matrix_multiplication_timing
