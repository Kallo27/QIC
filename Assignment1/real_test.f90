!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 1.2.b - NUMBER PRECISION
! Sum 𝜋*10e32 and √2*10e21 with single and double precision.

! Program Overview:
! - The program defines two variables for single precision (real) and double precision
! (double precision) to store the results of the calculations, together with 2 pairs of
! variables to store single and double precision values for pi and sqrt(2).
! - It calculates the values of π multiplied by 10^32 and √2 multiplied by 10^21 in both
! single and double precision formats.
! - It prints both results to show the difference in precision between the two types.

! NB: We can see here that we use 10e21 and then 10d21 -> The letter in the middle signifies
! that the number is either in single (e) or in double (d) precision format; in particular,
! it indicates that the subsequent exponent is for the same precision floating-point value.

program real_test
   implicit none
   real :: pi_s, sqrt2_s, result_single_precision
   double precision :: pi_d, sqrt2_d, result_double_precision

   pi_s = 3.1415927*1.0e32
   sqrt2_s = sqrt(2.0)*1.0e21

   pi_d = 3.141592653589793*1.0d32
   sqrt2_d = sqrt(2.0d0)*1.0d21

   result_single_precision = pi_s + sqrt2_s
   result_double_precision = pi_d + sqrt2_d

   print *, "Single Precision Result: ", result_single_precision
   print *, "Double Precision Result: ", result_double_precision

end program real_test
