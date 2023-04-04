#ifndef _UTILS_H_
#define _UTILS_H_


#include <string>
#include <vector>
#include <cuda_runtime.h>


using vstring = std::vector<std::string>;

/** Split input string by given delimiter
 *
 * @param str input string to split
 * @param del delimiter to split input string with
 */
vstring split(std::string const& str, char del);


/** Return local time representation in YYYY-mm-DD HH:MM:SS.zzz format
 *
 */
std::string now_str();


inline
cudaError_t
check_err_impl(cudaError_t err, char const* file, char const* caller_name, int line){
#ifndef DEBUG
   return err;
#else
   if (err == cudaSuccess){
     return err;
   }

   fprintf(
      stderr,
      "[%s:%d] %s: %s\n",
      file,
      line,
      cudaGetErrorName(err),
      cudaGetErrorString(err)
   );
   exit(1);
   return err;
#endif /* DEBUG */
}


inline
cudaError_t
check_last_err_impl(char const* file, char const* caller_name, int line){
   auto err = cudaGetLastError();

   if (err == cudaSuccess){
     return err;
   }

   fprintf(
      stderr,
      "[%s:%d] %s: %s\n",
      file,
      line,
      cudaGetErrorName(err),
      cudaGetErrorString(err)
   );
   exit(1);
   return err;
}


#define check_err(err) check_err_impl(err, __FILE__, __func__, __LINE__)
#define check_last_err() check_last_err_impl(__FILE__, __func__, __LINE__)



/** Insert input delimeter between vector elements
 *
 * @param vec input vector
 * @param del delimiter character
 * @return string compose of 'vec' elements separated by 'del'
 */
std::string
join(std::vector<std::string> const& vec, std::string const& del);


/** Return the maximum length of data in map
 *
 * @param data_map ordered or unordered data collection
 * @param accessor functor to access data individual collection data
 * @return maximum size of the data collection
 */
template <typename Map, typename F>
size_t
max_size(Map  const& data_map, F const& accessor){
   unsigned int max_size = 0;
   for (auto it=data_map.begin(); it!=data_map.end(); ++it){
     max_size = std::max<size_t>(accessor(it->second).size(), max_size);
   }
   return max_size;
}


/** Naive function to return file extension
 *
 * Don't want to add newer std as req
 *
 * @param path file path or filename
 * @return file extenison or empty string if none
 */
std::string
extension(std::string const& path);


/** Convert string to lower case
 *
 * @param str input string
 * @return string converted to lower case
 */
std::string
to_lower(std::string const& str);


struct Stats{
  double mean;
  double max;
  double min;
  double variance;
};



Stats
basic_stats(std::vector<double> const& data);


#endif /* _UTILS_H_ */
