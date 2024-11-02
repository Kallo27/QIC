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


module matmul_timing
  !  ===================================================================== 
  !  
  !    This module implements a matrix multiplication function, 
  !    which performs the multiplication in three different ways: row by
  !    column, column by row and using the intrinsic 'matmul' function.
  !    For each of them, the timing of the operation is printed to
  !    screen, to compare the different performances. Each multiplication
  !    is performed with 10 fractions of n, to see the how the performance
  !    scales with increasing sizes of the matrix.
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
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong input shapes for matrix product')
      stop  
    end if

    ! (2) Check if the number of rows of A is equal to the number of rows of c.
    if (size(A, 1) .ne. size(C, 1)) then
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong target shape (1) for matrix product') 
      stop 
    end if 

    ! (3) Check if the number of columns of B is equal to the number of columns of C.
    if (size(B, 2) .ne. size(C, 2)) then 
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong target shape (2) for matrix product') 
      stop     
    end if 


    ! Pre-condition: check if all elements of C are equal to 0.0.
    if (.not. all(C == 0.0)) then
      call checkpoint(debug = .TRUE., verb=1, msg = 'WARNING!!! Product matrix not empty, set C=0.0.')
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
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong input shapes for matrix product')
      stop  
    end if

    ! (2) Check if the number of rows of A is equal to the number of rows of c.
    if (size(A, 1) .ne. size(C, 1)) then
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong target shape (1) for matrix product') 
      stop 
    end if 

    ! (3) Check if the number of columns of B is equal to the number of columns of C.
    if (size(B, 2) .ne. size(C, 2)) then 
      call checkpoint(debug = .TRUE., verb=1, msg = 'ERROR!!! Wrong target shape (2) for matrix product') 
      stop     
    end if 


    ! Pre-condition: check if all elements of C are equal to 0.0.
    if (.not. all(C == 0.0)) then
      call checkpoint(debug = .TRUE., verb=1, msg = 'WARNING!!! Product matrix not empty, set C=0.0.')
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
    integer :: n, fraction, m
    real, allocatable :: A(:, :), B(:, :), C(:, :)
    real :: start_time, end_time, elapsed_time
    integer :: ios
    integer :: rowbycolumn_file, columnbyrow_file, intrinsic_file

    ! Pre-condition: validate the input for matrix size (n>0), otherwise throw an error.
    if (n <= 0) then
      call checkpoint(debug=.TRUE., verb=1, msg="ERROR!!! Matrix size must be a positive integer.")
      stop
    end if

    ! Open file for writing (replace mode). If unable to open, throw an error.
    open(unit=rowbycolumn_file, file='rowbycolumn.txt', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
      call checkpoint(debug=.TRUE., verb=1, msg="ERROR!!! Unable to open rowbycolumn log file.")
      stop
    end if

    ! Open file for writing (replace mode). If unable to open, throw an error.
    open(unit=columnbyrow_file, file='columnbyrow.txt', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
      call checkpoint(debug=.TRUE., verb=1, msg="ERROR!!! Unable to open columnbyrow log file.")
      stop
    end if

    ! Open file for writing (replace mode). If unable to open, throw an error.
    open(unit=intrinsic_file, file='intrinsic.txt', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
      call checkpoint(debug=.TRUE., verb=1, msg="ERROR!!! Unable to open intrinsic log file.")
      stop
    end if

    ! Loop 10 times to test performance with smaller matrix sizes (using fractional sizes up to n).
    do fraction = 1, 10
      m = max(1, fraction * n / 10)
      print *, "Percentage:", fraction*10, "%"

      ! Allocate memory for the matrices based on the current size 'm'.
      allocate(A(m, m), B(m, m), C(m, m))

      ! Pre-condition: check if memory allocation worked, otherwise throw an error.
      if (.not. allocated(A) .or. .not. allocated(B) .or. .not. allocated(C)) then
        call checkpoint(debug=.TRUE., verb=1, msg="ERROR!!! Memory allocation failed.")
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
      elapsed_time = end_time - start_time
      write(rowbycolumn_file, '(I5, F10.6)') m, elapsed_time
      !-------------------------------------------------------

      !-------------------------------------------------------
      ! Reset matrix C to zero before the next multiplication.
      C = 0.0

      ! Call 'cpu_time' to start time measurement.
      call cpu_time(start_time)
      call columnbyrow(A, B, C)
      call cpu_time(end_time)

      ! Compute elapsed time and write to columnbyrow file
      elapsed_time = end_time - start_time
      write(columnbyrow_file, '(I5, F10.6)') m, elapsed_time 
      !-------------------------------------------------------

      ! ------------------------------------------------------
      ! Reset matrix C to zero before the next multiplication.
      C = 0.0

      ! Call 'cpu_time' to start time measurement.
      call cpu_time(start_time)
      C = matmul(A, B)
      call cpu_time(end_time)

      ! Compute elapsed time and write to intrinsic file.
      elapsed_time = end_time - start_time
      write(intrinsic_file, '(I5, F10.6)') m, elapsed_time
      !-------------------------------------------------------


      ! Deallocate matrices for the current loop iteration.
      deallocate(A, B, C)

      ! Post-condition: check if deallocation worked, otherwise throw an error.
      if (allocated(A) .or. allocated(B) .or. allocated(C)) then
        call checkpoint(debug=.TRUE., verb=1, msg="WARNING!!! Matrices were not fully deallocated.")
        stop
      end if

      print *, "-----------------------------------------------------"
    end do

    ! Close the log files.
    close(rowbycolumn_file)
    close(columnbyrow_file)
    close(intrinsic_file)

    ! Final print statement -> ensures program ended without errors.
    print *, "Matrix multiplication tests completed successfully."
  end subroutine timing

end module matmul_timing