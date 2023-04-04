#include <iostream>
#include <future>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <random>
#include "matrix.h"
#include "dgemm.h"
#include "utils.h"
#include "args.h"
#include "serialize.h"



void
print_usage(std::string const& exec, std::ostream &stream){
  stream << "Usage: "
         << exec << " args\n";
  stream << "args:\n"
         << "   -m <int> Row count of matrix A\n"
         << "   -n <int> Column count of matrix A\n"
         << "   -k <int> Column count of matrix B\n"
         << "   -r <int> dgemm repetition count to measure flops\n"
         << "   -d <str> comma sep list of device ids, indexed at zero, to run concurrently run dgemm on\n"
         << "   -i <int> Number of iterations to of dgemm to perform\n"
         << "   [-o] <str> optional filename to write all data. If not .csv, will write in json\n";
}


args
parse_args(int argc, char *argv[]){
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
       print_usage(exec, std::cerr);
       return ret;
     }
   }

   if (ret.device_ids.size() == 0){
     ret.device_ids = {0};
   }

   return ret;
}


template <typename Q>
void task_sync(Q &queue){
   for (auto it=queue.begin(); it!=queue.end(); ++it){
     it->wait();
   }
}


template <typename T>
matrix<T>
pseudo_random_matrix(int m, int n, int seed=0){
   matrix<T> mat(m, n);
   std::mt19937 mt;
   mt.seed(seed);
   std::uniform_real_distribution<T> unum(-1, 1);
   for (int i=0; i<m; ++i){
     for (int j=0; j<n; ++j){
       mat(i, j) = unum(mt);
     }
   }
   return mat;
}


void
print_summary(
      std::unordered_map<int, gemm_results> const& rates,
      std::vector<int> const& devices,
      FILE * out){

   fprintf(out, "\n\n%5s | %10s | %10s | %10s | %10s\n",
           "DEV", "MIN", "MAX", "AVERAGE", "STD Dev"
   );
   fprintf(out,"-----------------------------------------------------------\n");
   for (auto dev_id : devices){
     auto stats = basic_stats(rates.at(dev_id).flops);
     fprintf(out,"%5d | %10g | %10g | %10g | %10g \n",
             dev_id, stats.min, stats.max, stats.mean, std::sqrt(stats.variance)
     );
   }
}


int
main(int argc, char *argv[]){
   auto const arg_list = parse_args(argc, argv);

   if (!args::validate(arg_list)){
     print_usage(argv[0], std::cerr);
     return 1;
   }

   using T = double;

   matrix<T> A = pseudo_random_matrix<T>(arg_list.m, arg_list.n, 0);
   matrix<T> B = pseudo_random_matrix<T>(arg_list.m, arg_list.n, 1);

   std::vector<std::future<gemm_results>> tasks;
   for (auto it=arg_list.device_ids.begin(); it!=arg_list.device_ids.end(); ++it){
         tasks.push_back(
            std::async(
               run_gemm<T>,
               A, B, arg_list.iter_count, arg_list.rep_count, *it)
         );
   }

   task_sync(tasks);

   std::unordered_map<int, gemm_results > rates;
   for (size_t i=0; i<tasks.size(); ++i){
     rates[arg_list.device_ids[i]] = tasks[i].get();
   }
   print_summary(rates, arg_list.device_ids, stdout);

   if (arg_list.output_fn != ""){
      std::ofstream out(arg_list.output_fn, std::ios::out);
      auto const ext = to_lower(extension(arg_list.output_fn));
      if (ext == "csv"){
        serialize_csv(rates, arg_list, out);
      }
      else{
        serialize(rates, arg_list, out);
      }
   }

   return 0;
}
