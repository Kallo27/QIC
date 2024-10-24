!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.1 - SETUP
! Open a code editor and write your first program in FORTRAN

! This simple program 'hello' prints "Hello, Fortran!" to the output.
! We use * as the format specifier, which tells Fortran to use the default
! (list-directed) formatting (Fortran handles it automatically).

! In order to run this program we first need to compile it, using:
! gfortran -o hello hello.f90
! This command will create an executable named hello (second argument passed
! to the gfortran compiler), which we need to execute using:
! ./hello

program hello
   print *, "Hello, Fortran!"
end program hello
