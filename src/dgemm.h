#ifndef _DGEMM_H_
#define _DGEMM_H_


#include <vector>
#include <string>
#include <cublas_v2.h>
#include "matrix.h"
#include "timer.h"
#include "utils.h"
#include "darray.h"



struct gemm_results{
  std::vector<double> flops;
  std::vector<std::string> time_points;
};


cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k,
     float const* alpha,
     float const* A_mat, int lda,
     float const* B_mat, int ldb,
     float const* beta,
     float *C_mat, int ldc);


cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k,
     double const* alpha,
     double const* A_mat, int lda,
     double const* B_mat, int ldb,
     double const *beta,
     double *C_mat, int ldc);


/** Run A*B on a gpu
 *
 * @param A input matrix A
 * @param B input matrix B
 * @param inter_count number of iterations to perform dgemm
 * @param rep_count number of times to perform dgemm to compute flops
 * @param device_id GPU device id, indexed at 0, to run on
 * @return estimated terraflops and corresponding local time
 *
 */
template <typename T>
gemm_results
run_gemm(matrix<T> const& A, matrix<T> const& B,
          int iter_count, int rep_count, int dev_id){
   check_err(cudaSetDevice(dev_id));
   cublasHandle_t handle;
   cublasCreate(&handle);

   cublasOperation_t const trans_a = CUBLAS_OP_N;
   cublasOperation_t const trans_b = CUBLAS_OP_T;

   int const m = A.row_count();
   int const n = A.col_count();
   int const k = B.row_count();
   int const lda = m;
   int const ldb = n;
   int const ldc = k;

   matrix<T> C(m, k);
   std::vector<T> mat_zeros(m*n);
   std::vector<T> mat_out(m*n);
   std::vector<T> const one = {{1.0}};

   auto alpha = to_device(one);
   auto beta = to_device(one);
   auto A_mat = to_device(A.data());
   auto B_mat = to_device(B.data());
   auto C_mat = to_device(C.data());
   gemm_results ret;

   cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
   for (int batch_idx=0; batch_idx<iter_count+1; ++batch_idx){
     timer t;
     t.tick();
     for (int i=0; i<rep_count; ++i){
       auto stat = gemm(handle,
                     trans_a, trans_b,
                     m, n, k,
                     alpha, A_mat, lda,
                     B_mat, ldb,
                     beta,  C_mat, ldc);
     }
     check_err(cudaDeviceSynchronize());

     // Ignore the first set
     if (batch_idx == 0){
       continue;
     }

     double const dur = t.tock<std::chrono::milliseconds>();
     double const op_count = rep_count*2.0*m*n*k;
     // ops/ms * 1000ms/1s * 1 gops/1e9 * 1 tops/1e3gops  ops
     double const tflops = op_count/dur/1.0e9;
     ret.flops.push_back(tflops);
     ret.time_points.push_back(now_str());
     printf("%d   %3d   %4.2f\n", dev_id, batch_idx, tflops);
   }

   return ret;
}




#endif /* _DGEMM_H_ */
