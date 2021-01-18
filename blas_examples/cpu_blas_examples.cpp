// In this exmaple I am trying to implement the BLAS versions for
// GEMM ops.
// dgemm: https://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
#include <stdio.h>

#include <ATen/native/CPUBlas.h>

int main(){
  double *a, *x, *y;
  double alpha, beta;
  int m, n, lda, incx, incy, i;

  m = 4;   // Size of column (number of rows)
  n = 4;   // Size of row (number of columns)
  lda = 4; // Leading dimension; eg. for a 5x4 matrix is 5
  incx = 1;
  incy = 1;
  alpha = 1;
  beta = 0;

  // now we define the matrices
  a = (double *) malloc (sizeof(double) * m * n);
  x = (double *) malloc (sizeof(double) * n);
  y = (double *) malloc (sizeof(double) * n);

  /* The elements of the first column */
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;
  /* The elements of the second column */
  a[m] = 1;
  a[m+1] = 1;
  a[m+2] = 1;
  a[m+3] = 1;
  /* The elements of the third column */
  a[m*2] = 3;
  a[m*2+1] = 4;
  a[m*2+2] = 5;
  a[m*2+3] = 6;
  /* The elements of the fourth column */
  a[m*3] = 5;
  a[m*3+1] = 6;
  a[m*3+2] = 7;
  a[m*3+3] = 8;
  /* The elements of x and y */
  x[0] = 1;
  x[1] = 2;
  x[2] = 1;
  x[3] = 1;
  y[0] = 0;
  y[1] = 0;
  y[2] = 0;
  y[3] = 0;

  // this gives 0x7fe7f5d7b830 as this is a pointer we get the address
  std::cout << a << "\n";
  // use *a to get value for first and *a+1 to get value for second
  std::cout << *a << " " << *a+1 << "\n";

  
}