!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.1.b - SETUP
! Submit a test job

! This simple program reads messages from an existing file called 'error.txt'
! (assuming it already exists) and prints the contents to the console.

! Program Overview:
! - The program reads all the lines from 'error.txt', in read-only mode (status='old'),
! using an infinte loop (closes when reaches error or end of the file).
! - It stores each line in a character variable 'message' (up to 1000 characters).
! - After reading each line, the program trims any trailing whitespace and prints
! the line to the screen.

! NB: 'implicit none' is a statement in Fortran that tells the compiler not to implicitly
! declare the variables, forcing the programmer to declare all variables explicitly, improving
! code safety and readability.

program test
   implicit none

   integer :: ii, ios
   character(len=1000) :: message

   open (1, file='error.txt', status='old')

   ii = 0
   do
      read (1, '(A)', iostat=ios) message

      if (ios /= 0) exit

      print *, trim(message)
      ii = ii + 1
   end do

   close (1)
   print *, "Total lines read: ", ii

end program test
