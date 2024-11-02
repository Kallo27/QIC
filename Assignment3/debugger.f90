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

module debugger
  implicit none
contains
  subroutine checkpoint(debug, msg)
    logical, intent(in) :: debug
    character(len=*), optional :: msg
        
    if (debug) then
      if (present(msg)) then
        print *, 'Checkpoint: ', msg
      else
        print *, "Checkpoint reached."
      end if
    end if
  end subroutine checkpoint
end module debugger
