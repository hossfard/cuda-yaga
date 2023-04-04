#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "dgemm.h"



cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k,
     double const* alpha,
     double const* A_mat, int lda,
     double const* B_mat, int ldb,
     double const *beta,
     double *C_mat, int ldc){
   return cublasDgemm(handle,
             transa, transb,
             m, n, k, alpha,
             A_mat, lda,
             B_mat, ldb,
             beta,  C_mat, ldc);
}


cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k,
     float const* alpha,
     float const* A_mat, int lda,
     float const* B_mat, int ldb,
     float const* beta,
     float *C_mat, int ldc){
   return cublasSgemm(handle,
             transa, transb,
             m, n, k, alpha,
             A_mat, lda,
             B_mat, ldb,
             beta,  C_mat, ldc);
}
