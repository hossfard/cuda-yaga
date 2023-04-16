#ifndef ARGS_H_
#define ARGS_H_


#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>


enum class precision_t{
  fp32, fp64
};


inline
std::string
to_string(precision_t dtype){
   switch (dtype){
      case precision_t::fp32:
        return "fp32";
      case precision_t::fp64:
        return "fp64";
      default:
        return "";
   }
}


struct args{
  // Matrix dimensions
  int m = 0;
  int n = 0;
  int k = 0;
  precision_t dtype = precision_t::fp32;

  std::vector<int> device_ids;

  // number of times to perform
  int iter_count = 1;

  // number of times to repeat gemm while measuring time
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

  static
  void
  usage(std::string const& exec, std::ostream &stream){
     stream << "Usage: "
            << exec << " args\n";
     stream << "args:\n"
            << "   -m <int> Row count of matrix A\n"
            << "   -n <int> Column count of matrix A\n"
            << "   -k <int> Column count of matrix B\n"
            << "   -r <int> gemm repetition count to measure flops\n"
            << "   -p <str> ('fp32', 'fp64') data type - defaults to fp32\n"
            << "   -d <str> comma sep list of device ids, indexed at zero, to run concurrently run gemm on\n"
            << "   -i <int> Number of iterations to of gemm to perform\n"
            << "   [-o] <str> optional filename to write all data. If not .csv, will write in json\n";
  }

  static
  args
  parse(int argc, char *argv[]){
     args ret;
     std::string const exec(argv[0]);

     while (--argc > 0){
       std::string arg((++argv)[0]);
       if (arg == "-m"){
         std::string const val((++argv)[0]);
         ret.m = std::stoi(val);
         --argc;
       }
       else if (arg == "-n"){
         std::string const val((++argv)[0]);
         ret.n = std::stoi(val);
         --argc;
       }
       else if (arg == "-k"){
         std::string const val((++argv)[0]);
         ret.k = std::stoi(val);
         --argc;
       }
       else if (arg == "-d"){
         std::string const val((++argv)[0]);
         auto vals = split(val, ',');
         std::for_each(vals.begin(), vals.end(), [&ret](std::string const& v){
            ret.device_ids.push_back(std::stoi(v));
         });
         --argc;
       }
       else if (arg == "-p"){
         std::string const val((++argv)[0]);
         if (val == "fp32"){
           ret.dtype = precision_t::fp32;
         }
         else if (val == "fp64"){
           ret.dtype = precision_t::fp64;
         }
         --argc;
       }
       else if (arg == "-i"){
         std::string const val((++argv)[0]);
         ret.iter_count = std::stoi(val);
         --argc;
       }
       else if (arg == "-r"){
         // number of times to repeat while measuring duration
         std::string const val((++argv)[0]);
         ret.rep_count = std::stoi(val);
         --argc;
       }
       else if (arg == "-o"){
         ret.output_fn = (++argv)[0];
         --argc;
       }
       else{
         std::cerr << "Invalid argument: " << arg << std::endl;
         usage(exec, std::cerr);
         return ret;
       }
     }

     if (ret.device_ids.size() == 0){
       ret.device_ids = {0};
     }

     return ret;
  }


};



#endif /* ARGS_H_ */
