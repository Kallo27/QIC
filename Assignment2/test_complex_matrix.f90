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


program test_complex_matrix
  !  ===================================================================== 
  !  
  !    Test program for the ComplexMatrix module.
  !  
  !  ===================================================================== 

  use complex_matrix
  implicit none
    
  ! Variable definition.
  type(ComplexMatrix) :: my_matrix
  double complex, allocatable :: input_elements(:,:)
  character(len=100) :: output_file
  integer :: i, j
  integer, dimension(2) :: size

  size(1) = 4
  size(2) = 4

  ! Allocate input elements for initialization.
  allocate(input_elements(size(1), size(2)))
    
  ! Fill the input elements with some values.
  ! dcmplx builds complex elements (a + i*b).
  do i = 1, size(1)
    do j = 1, size(2)
      input_elements(i, j) = dcmplx(real(i + j), real(i - j))
    end do
  end do

  ! Initialize the matrix with the input elements.
  call initialize_matrix(my_matrix, input_elements)

  ! Specify the output file name.
  output_file = "complex_matrix_output.txt"

  ! Write the matrix details to a file.
  call write_matrix_to_file(my_matrix, output_file)

  ! Final print statement.
  print *, "Finished test."

  ! Clean up
  deallocate(input_elements)
  deallocate(my_matrix%elements)
  deallocate(my_matrix%adjoint)

end program test_complex_matrix