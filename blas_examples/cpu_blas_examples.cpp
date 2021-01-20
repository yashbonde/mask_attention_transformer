// In this exmaple I am trying to implement the BLAS versions for
// GEMM ops.
// dgemm: https://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html

#include <stdio.h>
#include <stdlib.h>
#include <ATen/native/CPUBlas.h>

// my implementation of: https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_14.c
#define min( i, j ) ( (i)<(j) ? (i): (j) )
#define mc 256
#define kc 128

void random_matrix( int m, int n, double *a) {
  for (int i=0; i<m; i++ ){
    for (int j=0; j<n; j++ ){
      // note in memory we are storing all these values as a contiguous array
      // so we need to get the proper indices
      a[i*n +j] = 2.0 * drand48() - 1.0;
  }}
}

void print_matrix(int m, int n, double *a) {
  for (int i=0; i<m; i++ ){
    for (int j=0; j<n; j++ ){
      std::cout << a[i*n + j] << " ";}
    std::cout << "\n";}
}

void print_number_of_values(int m, int n, double *a) {
  int cntr = 0;
  for (int i=0; i<m; i++ ){
    for (int j=0; j<n; j++ ){
      cntr += 1;
  }}
  std::cout << "number of tokens: " << cntr;
}

void my_gemm(
  // char transa, char transb,
  int m, int n, int k,
  // int alpha = 1, int beta = 0,
  double *a, int lda,
  double *b, int ldb,
  double *c, int ldc
  ) {
  // the arguments are same as for the GEMM operations

}

int main(){
  // In this demo we are calculating C = A x B
  int m, n, k;
  m = 40;
  n = 50;
  k = 60;

  double *a = (double *) malloc (sizeof(double) * m * k); // A has shape m x k
  double *b = (double *) malloc (sizeof(double) * k * n); // B has shape k x n
  random_matrix(m, k, a);
  random_matrix(k, n, b);

  std::cout << "---- Matrix A ";
  print_number_of_values(m,k,a);
  std::cout << "\n---- Matrix B ";
  print_number_of_values(k, n, b);

  // print if needed
  if (false){
    std::cout << "---- Matrix A\n";
    print_matrix(m, k, a);
    std::cout << "---- Matrix B\n";
    print_matrix(k, n, b);
    std::cout << "---------------------\n";
  }

  free (a);
  free (b);
}