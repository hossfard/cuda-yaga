#ifndef ARGS_H_
#define ARGS_H_


#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>


enum class precision_t{
  fp32, fp64
};


struct args{
  // Matrix dimensions
  int m = 0;
  int n = 0;
  int k = 0;
  precision_t dtype = precision_t::fp32;

  std::vector<int> device_ids;

  // number of times to perform
  int iter_count = 1;

  // number of times to repeat dgemm while measuring time
  int rep_count = 10;

  std::string output_fn;

  static
  bool
  validate(args const& input){
     int deviceCount = -1;
     check_err(cudaGetDeviceCount(&deviceCount));
     auto maxDeviceId = *(std::max_element(
        input.device_ids.begin(), input.device_ids.end()));

     return (input.m > 1)
       && (input.n > 0)
       && (input.k > 0)
       && (input.iter_count > 0)
       && (input.rep_count > 0)
       && (maxDeviceId < deviceCount);
  }

};



#endif /* ARGS_H_ */
