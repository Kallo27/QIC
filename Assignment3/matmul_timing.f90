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

module matmul_timing
  !  ===================================================================== 
  !  
  !    This module implements a matrix multiplication function, 
  !    which performs the multiplication in three different ways: row by
  !    column, column by row and using the intrinsic 'matmul' function.
  ! 
  !  =====================================================================  
  !  
  !  SUBROUTINES:
  !
  !  rowbycolumn(A, B, C)
  !
  !           Inputs  | A(:,:) (real): first matrix (m x n) 
  !                   | B(:,:) (real): second matrix (n x l)
  !                   | C(:,:) (real): result matrix (m x l). If not 
  !                   |                initialized to all zeros, a warning 
  !                   |                occurs, then it's set to 0.0 to 
  !                   |                proceed.
  !                   |
  !                   | Correct dimensions of the input matrices are 
  !                   | checked before the multiplication; in case of
  !                   | mismatch, an error occurs. 
  !
  !
  !  columnbyrow(A, B, C)
  !
  !           Inputs  | A(:,:) (real): first matrix (m x n) 
  !                   | B(:,:) (real): second matrix (n x l)
  !                   | C(:,:) (real): result matrix (m x l). If not 
  !                   |                initialized to all zeros, a warning 
  !                   |                occurs, then it's set to 0.0 to 
  !                   |                proceed.
  !                   |
  !                   | Correct dimensions of the input matrices are 
  !                   | checked before the multiplication; in case of
  !                   | mismatch, an error occurs. 
  !
  !
  !  matmul_timing(n) 
  ! 
  !           Inputs  | n>0 (integer): dimension of the square matrix. If
  !                   |                n <= 0, then an error occurs, which 
  !                   |                stops the rubroutine execution.   
  !
  !  =====================================================================  


  ! Using debugger module for debugging.
  use debugger

  ! No implicitly declared variable
  implicit none

contains
  subroutine rowbycolumn(A, B, C)
    ! Variable definition.
    real :: A(:, :),  B(:, :), C(:, :)
    integer :: i, j, k
    
    ! Pre-conditions: ensure that the input matrices have compatible dimensions.
    ! (1) Check if the number of columns of B is equal to the number of rows of A.
    if (size(B, 2) .ne. size(A, 1)) then
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong input shapes for matrix product')
      stop  
    end if

    ! (2) Check if the number of rows of A is equal to the number of rows of c.
    if (size(A, 1) .ne. size(C, 1)) then
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong target shape (1) for matrix product') 
      stop 
    end if 

    ! (3) Check if the number of columns of B is equal to the number of columns of C.
    if (size(B, 2) .ne. size(C, 2)) then 
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong target shape (2) for matrix product') 
      stop     
    end if 


    ! Pre-condition: check if all elements of C are equal to 0.0.
    if (.not. all(C == 0.0)) then
      call checkpoint(debug = .TRUE., msg = 'WARNING!!! Product matrix not empty, set C=0.0.')
      C = 0.0
    end if

    ! Actual multiplication
    do i = 1, size(C, 1)
      do j = 1, size(C, 2)
        do k = 1, size(A, 2)
          C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
      end do
    end do

  end subroutine rowbycolumn

  ! ===================================================================== 

  subroutine columnbyrow(A, B, C)
    ! Variable definition.
    real :: A(:, :),  B(:, :), C(:, :)
    integer :: i, j, k
    
    ! Pre-conditions: ensure that the input matrices have compatible dimensions.
    ! (1) Check if the number of columns of B is equal to the number of rows of A.
    if (size(B, 2) .ne. size(A, 1)) then
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong input shapes for matrix product')
      stop  
    end if

    ! (2) Check if the number of rows of A is equal to the number of rows of c.
    if (size(A, 1) .ne. size(C, 1)) then
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong target shape (1) for matrix product') 
      stop 
    end if 

    ! (3) Check if the number of columns of B is equal to the number of columns of C.
    if (size(B, 2) .ne. size(C, 2)) then 
      call checkpoint(debug = .TRUE., msg = 'ERROR!!! Wrong target shape (2) for matrix product') 
      stop     
    end if 


    ! Pre-condition: check if all elements of C are equal to 0.0.
    if (.not. all(C == 0.0)) then
      call checkpoint(debug = .TRUE., msg = 'WARNING!!! Product matrix not empty, set C=0.0.')
      C = 0.0
    end if

    ! Actual multiplication
    do j = 1, size(C, 1)
      do i = 1, size(C, 2)
        do k = 1, size(A, 2)
          C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
      end do
    end do

  end subroutine columnbyrow

  ! ===================================================================== 

  subroutine timing(n)
    ! Variable definition.
    integer :: n
    real, allocatable :: A(:, :), B(:, :), C(:, :)
    real :: start_time, end_time, elapsed_time1, elapsed_time2, elapsed_time3

    ! Pre-condition: validate the input for matrix size (n>0), otherwise throw an error.
    if (n <= 0) then
      call checkpoint(debug=.TRUE., msg="ERROR!!! Matrix size must be a positive integer.")
      stop
    end if

    ! Allocate memory for the matrices.
      allocate(A(n, n), B(n, n), C(n, n))

    ! Pre-condition: check if memory allocation worked, otherwise throw an error.
    if (.not. allocated(A) .or. .not. allocated(B) .or. .not. allocated(C)) then
      call checkpoint(debug=.TRUE., msg="ERROR!!! Memory allocation failed.")
      stop
    end if


    !-------------------------------------------------------
    ! Initialize matrices A and B with 1.0s and set C to zeros.
    A = 1.0
    B = 1.0
    C = 0.0
      
    ! Call 'cpu_time' to start time measurement.
    call cpu_time(start_time)
    call rowbycolumn(A, B, C)
    call cpu_time(end_time)

    ! Compute elapsed time and write to rowbycolumn file.
      elapsed_time1 = end_time - start_time
    !-------------------------------------------------------

    !-------------------------------------------------------
    ! Reset matrix C to zero before the next multiplication.
    C = 0.0

    ! Call 'cpu_time' to start time measurement.
    call cpu_time(start_time)
    call columnbyrow(A, B, C)
    call cpu_time(end_time)

    ! Compute elapsed time and write to columnbyrow file
    elapsed_time2 = end_time - start_time
    !-------------------------------------------------------

    ! ------------------------------------------------------
    ! Reset matrix C to zero before the next multiplication.
    C = 0.0

    ! Call 'cpu_time' to start time measurement.
    call cpu_time(start_time)
    C = matmul(A, B)
    call cpu_time(end_time)

    ! Compute elapsed time and write to intrinsic file.
    elapsed_time3 = end_time - start_time
    !-------------------------------------------------------


    ! Deallocate matrices for the current loop iteration.
    deallocate(A, B, C)

    ! Post-condition: check if deallocation worked, otherwise throw an error.
    if (allocated(A) .or. allocated(B) .or. allocated(C)) then
      call checkpoint(debug=.TRUE., msg="WARNING!!! Matrices were not fully deallocated.")
      stop
    end if

    ! Output the elapsed times in a structured format.
    print *, "RC", ",", n, ",", elapsed_time1
    print *, "CR", ",", n, ",", elapsed_time1
    print *, "I", ",", n, ",", elapsed_time3

    ! Final print statement -> ensures program ended without errors.
    print *, "Matrix multiplication tests completed successfully."
  end subroutine timing

end module matmul_timing