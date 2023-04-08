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


template <typename T>
std::vector<std::future<gemm_results>>
dispatch_gemms(args const& arg_list){
   matrix<T> const A = pseudo_random_matrix<T>(arg_list.m, arg_list.n, 0);
   matrix<T> const B = pseudo_random_matrix<T>(arg_list.m, arg_list.n, 0);

   std::vector<std::future<gemm_results>> tasks;
   for (auto it=arg_list.device_ids.begin(); it!=arg_list.device_ids.end(); ++it){
         tasks.push_back(
            std::async(
               run_gemm<T>,
               A, B, arg_list.iter_count, arg_list.rep_count, *it)
         );
   }

   return tasks;
}


int
main(int argc, char *argv[]){
   auto const arg_list = args::parse(argc, argv);

   if (!args::validate(arg_list)){
     args::usage(argv[0], std::cerr);
     return 1;
   }


   std::vector<std::future<gemm_results>> tasks;
   switch (arg_list.dtype){
      case precision_t::fp32: {
        tasks = dispatch_gemms<float>(arg_list);
        break;
      }
      case precision_t::fp64: {
        tasks = dispatch_gemms<double>(arg_list);
        break;
      }
      default: {
        break;
      }
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
