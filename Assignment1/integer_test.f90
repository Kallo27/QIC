!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.2.a - NUMBER PRECISION
! Sum 2.000.000 and 1 with INTEGER*2 and INTEGER*4

! Program Overview:
! - The program defines two variables, x2 and x4, sepcifying their respective type
! (integer 2, range from -32,768 to 32,767, and integer 4, range from -2,147,483,648
! to 2,147,483,647) and intializes both of them to 2 million.
! - Then prints both variables increased by one: x2 will overflow while x4 will work
! correctly (due to their available precision range).

! NB: The compiler already knows that x2 + 1 will overflow, thus we need to compile
! using a flag which tells it to don't care about it, something like:
! gfortran -o integer_test integer_test.f90 -fno-range-check
! where '-fno-range-check' is the flag added (it will return a random value).

program integer_test
   implicit none
   integer*2 :: x2
   integer*4 :: x4

   x2 = 2000000
   x4 = 2000000

   print *, "INTEGER*2: ", x2 + 1
   print *, "INTEGER*4: ", x4 + 1

end program integer_test
