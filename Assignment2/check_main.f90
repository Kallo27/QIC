!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 2.1 - CHECKPOINT
! Write a subroutine to be used as a checkpoint for debugging.
! (a) Include a control on a logical variable (Debug=.TRUE. or .FALSE.)
! (b) Include an additional (optional) string to be printed.
! (c) Include additional (optional) variables to be printed.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Program Overview:
! - This program, `check_main`, demonstrates the use of a `checkpoint` subroutine for 
!   debugging purposes in a Fortran program.
!
! - Program Logic:
!     - The program initializes integer variables 'a' and 'b', and calculates their 
!       sum 'result'. Then, using formatted `write` statements, it converts 'result' 
!       to a string for output.
!     - Several checkpoints with different verbosity levels and conditions track the 
!       program state:
!         - First, it prints the result when 'debug' is `.TRUE.` and 'verb=1', not
!           printing any optional variable.
!         - A logic check verifies if 'result' is greater than 20, printing a detailed 
!           checkpoint message if true ('verb=2'), printing also 'var1'.
!         - Finally, a checkpoint with 'verb=3' is called, in order to show a more
!           detailed message and also the values of all optional variables.
!         - It then toggles `debug` to `.FALSE.` to show that no checkpoint messages 
!           appear when debugging is disabled.
!
! Compiler instructions:
! - First, run 'gfortran -c debug.f90' in order to produce the 'debug.o' file.
! - Then, run 'gfortran -o check_main.x debug.o check_main.f90' to compile the 
!   'check_main.f90' together with the debug.o file, producing the check_main.x executable.
! - Run the executable with './check_main.x'.

program check_main
  use debugger
  implicit none
    
  integer :: a, b, result    
  logical :: debug_mode
  character(len=20) :: result_str

  a = 10
  b = 20
  result = a + b
  write(result_str, '(I0)') result


  debug_mode = .TRUE.

  call checkpoint(debug_mode, verb=1, msg='Addition: a + b = ' // trim(adjustl(result_str)), var1 = 13*3.8, var2 = 0.4, var3 = 6.9)
  
  if (result > 20) then
    call checkpoint(debug_mode, verb=2, msg = 'Logical check: result is greater than 20.', var1 = 1.0, var2 = 9.76, var3 = 6.9)
  else
    call checkpoint(debug_mode, verb=2, msg = 'Logical check: result is not greater than 20.', var1 = 0.0, var2 = 8.765, var3 = 6.9)
  end if

  call checkpoint(debug_mode, verb=3, msg='Addition: a + b = ' // trim(adjustl(result_str)), var1 = 13*3.8, var2 = 0.4, var3 = 6.9)

  print *, 'Debug flag = .TRUE.'

  
  debug_mode = .FALSE.

  call checkpoint(debug_mode, verb=1, msg='Addition: a + b = ' // trim(adjustl(result_str)))
  
  if (result > 20) then
    call checkpoint(debug_mode, verb=2, msg = 'Logical check: result is greater than 20.')
  else
    call checkpoint(debug_mode, verb=2, msg = 'Logical check: result is not greater than 20.')
  end if

  print *, 'Debug flag = .FALSE.'

end program check_main
