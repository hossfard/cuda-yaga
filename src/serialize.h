#ifndef SERIALIZE_H_
#define SERIALIZE_H_


#include <string>
#include <vector>
#include <ostream>
#include <unordered_map>
#include <sstream>
#include "utils.h"
#include "args.h"
#include "dgemm.h"
#include "dquery.h"



namespace jo{
  std::ostream&
  operator<<(std::ostream &stream, std::string const& d);

  struct jkey{
    std::string value;
  };


  inline std::ostream&
  operator<<(std::ostream &stream, jkey const& d){
     return stream << d.value << ":";
  }

  template <typename T>
  std::ostream&
  operator<<(std::ostream &stream, std::vector<T> const& data){
     if (data.size() == 0){
       stream << "[]";
       return stream;
     }

     using jo::operator<<;

     stream << "[";
     for (size_t i=0; i<data.size(); ++i){
       stream << data[i];
       if (i != data.size() - 1){
         stream << ",";
       }
     }
     stream << "]";
     return stream;
  }

};


std::ostream&
jserialize_str(std::vector<std::string> const& data, std::ostream &stream);


std::string
jserialize(vstring const& data);


std::string
jserialize(args const& inp);


void
serialize(
      std::unordered_map<int, gemm_results> const& perf,
      std::vector<dstate_snapshots> const& device_hist,
      std::vector<device_info> const& devices,
      sys_info const& sysinfo,
      args const& inp,
      std::ostream &stream);


void
serialize_csv(
      std::unordered_map<int, gemm_results> const& map,
      args const& inp,
      std::ostream &stream);



#endif /* SERIALIZE_H_ */
