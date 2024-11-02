!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 2.3 - DERIVED TYPES
! In Fortran90 write a MODULE containing a double complex matrix derived 
! TYPE that includes the components: Matrix elements, Matrix Dimensions, 
! Matrix Trace, and Matrix Adjoint.
!
! (a) Define the correspondent TYPE.
! (b) Define a function/subroutine that initializes this new TYPE.
! (c) Define the functions/subroutines Trace and Adjoint.
! (d) Define the correspondent Interfaces of the previous points.
! (e) Define a subroutine that writes on file the Matrix TYPE in a readable form.
! (f) Include everything in a test program.


module complex_matrix
  !  ===================================================================== 
  !  
  !    This module defines and implements operations for a double complex 
  !    matrix type, which includes:
  !    - Matrix elements
  !    - Matrix dimensions
  !    - Matrix trace
  !    - Matrix adjoint
  ! 
  !  =====================================================================  
  !  
  !  SUBROUTINES:
  !
  !  initialize_matrix(matrix, input_elements)
  !
  !           Inputs  | matrix (ComplexMatrix, out): The matrix object 
  !                   |                to initialize.
  !                   | input_elements(:,:) (double complex, in): Elements 
  !                   |                used for initialization.
  !
  !
  !  write_matrix_to_file(matrix, filename)
  !
  !           Inputs  | matrix (ComplexMatrix, out): The matrix object 
  !                   |                to be saved to file.
  !                   | filename (character(len=*), in): Name of the 
  !                   |                file where to save.
  !
  !
  !  FUNCTIONS:
  !
  !  calculate_trace(matrix)
  !
  !           Inputs  | matrix (ComplexMatrix, in): The matrix whose 
  !                   |                trace is to be computed.
  !                   |
  !           Outputs | trace_value (double complex): The computed trace.
  !
  !
  !  calculate_adjoint(matrix)
  !
  !           Inputs  | matrix (ComplexMatrix, in): The matrix to compute 
  !                   |                the adjoint for.
  !                   
  !           Outputs | adjoint_mat(:,:) (double complex, allocatable): 
  !                   |                 The computed adjoint.
  !
  !  =====================================================================

  
  ! Use debugger module.
  use debugger

  ! No implicitly declared variable
  implicit none
  
  ! Definition of the derived type for a double complex matrix.
  type :: ComplexMatrix
    double complex, allocatable :: elements(:, :) ! Matrix elements
    integer, dimension(2) :: size                 ! Matrix size
    double complex :: trace                       ! Matrix trace
    double complex, allocatable :: adjoint(:, :)  ! Matrix adjoint
  end type ComplexMatrix
  
    ! Interfaces for the trace and adjoint operators.
    interface operator(.Tr.)
      module procedure calculate_trace
    end interface

    interface operator(.Adj.)
    module procedure calculate_adjoint
    end interface

  contains
  
    ! Subroutine to initialize the ComplexMatrix type.
    subroutine initialize_matrix(matrix, input_elements)
      ! Variable definition
      implicit none
      type(ComplexMatrix), intent(out) :: matrix
      double complex, allocatable, intent(in) :: input_elements(:, :)
  
      ! Check dimensions of input_elements
      if (size(input_elements, 1) <= 0 .or. size(input_elements, 2) <= 0) then
        call checkpoint(debug = .TRUE., verb=1, msg="ERROR: input_elements must have positive dimensions.")
        stop
      end if

      ! Set matrix dimensions and allocate memory (adjoint has transposed dimensions).
      matrix%size(1) = size(input_elements, 1)
      matrix%size(2) = size(input_elements, 2)
      allocate(matrix%elements(matrix%size(1), matrix%size(2)))
      allocate(matrix%adjoint(matrix%size(2), matrix%size(1)))
  
      ! Initialize matrix elements.
      matrix%elements = input_elements
  
      ! Compute the trace of the matrix.
      matrix%trace = calculate_trace(matrix)

      ! Compute the adjoint of the matrix
      matrix%adjoint = calculate_adjoint(matrix)
    end subroutine initialize_matrix

    ! ===================================================================== 

    ! Function to calculate the trace of the matrix.
    function calculate_trace(matrix) result(trace_value)
      ! Variable definition
      type(ComplexMatrix), intent(in) :: matrix
      double complex :: trace_value
      integer :: i 
      
      ! Check if the matrix is square before computing the trace.
      if (matrix%size(1) /= matrix%size(2)) then
        call checkpoint(debug = .TRUE., verb=1, msg="ERROR!!! Trace can only be computed for square matrices.")
        stop
      end if

      ! Initialize the trace as a double complex number
      trace_value = (0.0, 0.0)

      ! Calculate the trace of the matrix.
      do i = 1, min(matrix%size(1), matrix%size(2))
        trace_value = trace_value + matrix%elements(i, i)
      end do
    end function calculate_trace

    ! ===================================================================== 
  
    ! Function to calculate the adjoint of the matrix.
    function calculate_adjoint(matrix) result(adjoint_mat)
      type(ComplexMatrix), intent(in) :: matrix
      double complex, allocatable :: adjoint_mat(:,:)

      ! Allocate the adjoint matrix with transposed dimensions
      allocate(adjoint_mat(matrix%size(2), matrix%size(1)))
    
      ! Compute the adjoint (conjugate transpose) of the matrix
      adjoint_mat = transpose(conjg(matrix%elements))
    end function calculate_adjoint

    ! ===================================================================== 

    ! Subroutine to write matrix to file
    subroutine write_matrix_to_file(matrix, filename)
      ! Variable definition
      type(ComplexMatrix), intent(in) :: matrix
      character(len=*), intent(in) :: filename
      integer :: ios, unit_num
      integer :: i, j
  
      ! Open the output file
      open(unit=unit_num, file=filename, status='replace', action='write', iostat=ios)
      if (ios /= 0) then
        call checkpoint(debug=.TRUE., verb=1, msg="ERROR: Unable to open file " // trim(filename))
        stop
      end if
  
      ! Write matrix dimensions
      write(unit_num, *) "Matrix Dimensions: ", matrix%size(1), " x ", matrix%size(2)
      
      ! Write the matrix elements
      write(unit_num, *) "Matrix Elements:"
      do i = 1, matrix%size(1)
        do j = 1, matrix%size(2)
          write(unit_num, '(F10.4, " + i*", F10.4)', advance='no') real(matrix%elements(i, j)), aimag(matrix%elements(i, j))

          ! Add a comma between elements on the same row.
          if (j < matrix%size(2)) then
            write(unit_num, '(A)', advance='no') ', '
          end if
        end do

        ! New line after each row, for readibility.
        write(unit_num, *) '' 
      end do
  
      ! Write the trace
      write(unit_num, *) "Matrix Trace: ", matrix%trace
  
      ! Write the adjoint matrix
      write(unit_num, *) "Adjoint Matrix:"
      do i = 1, matrix%size(1)
        do j = 1, matrix%size(2)
          write(unit_num, '(F10.4, " + i*", F10.4)', advance='no') real(matrix%adjoint(i, j)), aimag(matrix%adjoint(i, j))

          ! Add a comma between elements on the same row.
          if (j < matrix%size(2)) then
            write(unit_num, '(A)', advance='no') ', '
          end if
        end do

        ! New line after each row, for readibility.
        write(unit_num, *) '' 
      end do
  
      ! Close the output file
      close(unit_num)
  
      print *, "Matrix written to file: ", filename
    end subroutine write_matrix_to_file
    
  end module complex_matrix
  