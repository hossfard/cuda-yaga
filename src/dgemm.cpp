#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "dgemm.h"
#include "darray.h"
#include "timer.h"



dgemm_results
run_dgemm(matrixd const& A, matrixd const& B,
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

   matrixd C(m, k);

   darray<double> alpha(1);
   darray<double> beta(1);
   darray<double> A_mat(m*n);
   darray<double> B_mat(n*k);
   darray<double> C_mat(m*k);

   double const alpha_val = 1.0;
   double const beta_val = 1.0;
   check_err(
      cudaMemcpy(alpha.data(), &alpha_val, 1*sizeof(double), cudaMemcpyHostToDevice)
   );
   check_err(
      cudaMemcpy(beta.data(), &beta_val, 1*sizeof(double), cudaMemcpyHostToDevice)
   );

   std::vector<double> mat_zeros(m*n);
   std::vector<double> mat_out(m*n);
   std::vector<double> const one = {{1.0}};

   check_err(
      cudaMemcpy(A_mat.data(), &A.data()[0], m*n*sizeof(double), cudaMemcpyHostToDevice));
   check_err(
      cudaMemcpy(B_mat.data(), &B.data()[0], m*n*sizeof(double), cudaMemcpyHostToDevice));
   check_err(
      cudaMemcpy(C_mat.data(), &mat_zeros[0], m*n*sizeof(double), cudaMemcpyHostToDevice));

   dgemm_results ret;
   cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

   for (int batch_idx=0; batch_idx<iter_count+1; ++batch_idx){
     timer t;
     t.tick();
     for (int i=0; i<rep_count; ++i){
       auto stat = cublasDgemm(
          handle,
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
