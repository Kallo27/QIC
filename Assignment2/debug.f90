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
! - This program provides a checkpoint subroutine within a debugging module 
!   to help monitor the state of a program at specific points. This is useful 
!   for tracking and diagnosing issues in complex computations.
!
! - The subroutine "checkpoint" provides the following functionalities:
!     1. Control by a logical 'debug' flag: when 'debug' is set to `.TRUE.`, the 
!        checkpoint message is printed, while when 'debug' is `.FALSE.`, the message 
!        is suppressed.
!     2. Verbosity levels (controlled by the integer 'verb': 1, 2 or 3):
!        - Level 1: Basic checkpoint message.
!        - Level 2: More detailed checkpoint message.
!        - Level 3: Full verbosity message for maximum detail.
!        - Default case: Prints an error (when we choose a value different than 1, 2, 3).
!     3. Optional message printing:
!        - The 'msg' argument is optional and, if provided, allows for a custom
!          message to be printed at the checkpoint.
!     4. Optional variables:
!        - The 'var1', 'var2' and 'var3' arguments are optional: if provided, they can be 
!          printed at the checkpoint (depending on the chosen verbose option).
!     NB: `intent(in)` indicates that the variable is an input to the subroutine and 
!         cannot be modified within it (in this case 'debug' and 'verb').
!
! - This modularized structure allows for reusing `checkpoint` in any part 
!   of the program that requires debugging output, improving code clarity 
!   and maintainability (we can call it many times in the same code).

module debugger
  implicit none
contains
  subroutine checkpoint(debug, verb, msg, var1, var2, var3)
    logical, intent(in) :: debug
    integer, intent(in) :: verb
    character(len=*), optional :: msg
    real(4), intent(in), optional :: var1, var2, var3
        
    if (debug) then
      select case (verb)
        case (1)
          if (present(msg)) then
            print *, 'Checkpoint: ', msg
          else
            print *, "Checkpoint reached."
          end if
        case (2)
          if (present(msg)) then
            print *, 'Detailed Checkpoint: ', msg
            if (present(var1)) then
              print *, "Var1: ", var1
            end if
          else
            print *, "Detailed checkpoint reached."
            if (present(var1)) then
              print *, "Var1: ", var1
            end if          
          end if
        case (3)
          if (present(msg)) then
            print *, 'Full verbosity checkpoint: ', msg
            if (present(var1)) then
              print *, "Var1: ", var1
            end if 
            if (present(var2)) then
              print *, "Var2: ", var2
            end if 
            if (present(var3)) then
              print *, "Var3: ", var3
            end if 
          else
            print *, "Full verbosity checkpoint reached."
            if (present(var1)) then
              print *, "Var1: ", var1
            end if 
            if (present(var2)) then
              print *, "Var2: ", var2
            end if 
            if (present(var3)) then
              print *, "Var3: ", var3
            end if 
          end if
        case default
            print *, "Unknown verbosity level. Please choose a value between 1, 2, 3."
      end select
    end if
  end subroutine checkpoint
end module debugger
