#include "serialize.h"
#include "utils.h"




namespace jo{

  std::ostream&
  operator<<(std::ostream &stream, std::string const& d){
     using std::operator<<;
     return stream << "\"" << d << "\"";
  }

};


std::ostream&
operator<<(std::ostream& stream, args const& inp){
   using jo::operator<<;

   stream << "{";
     stream << jo::jkey({"m"}) << inp.m << ",";
     stream << jo::jkey({"n"}) << inp.n << ",";
     stream << jo::jkey({"k"}) << inp.k << ",";
     stream << jo::jkey({"i"}) << inp.iter_count << ",";
     stream << jo::jkey({"r"}) << inp.rep_count << ",";
     stream << jo::jkey({"p"}) << to_string(inp.dtype) << ",";
     stream << jo::jkey({"o"}) << inp.output_fn << ",";

     stream << "\"d\":" << inp.device_ids;
   stream << "}\n";

   return stream;
}


std::ostream&
operator<<(std::ostream& stream, dstate_snapshots const& state){
   using jo::operator<<;

   stream << "{";
     stream << "\"id\":" << state.id << ",";
     stream << "\"t\":" << state.t << ",";
     stream << "\"temp.gpu\":" << state.temperature << ",";
     stream << "\"power\":" << state.power << ",";
     stream << "\"clock.sm\":" << state.smClock << ",";
     stream << "\"clock.mem\":" << state.memClock << ",";
     stream << "\"fans.speed\":" << state.fans;
   stream << "}";
   return stream;
}


void
serialize(
      std::unordered_map<int, gemm_results> const& map,
      std::vector<dstate_snapshots> const& device_hist,
      args const& inp,
      std::ostream &stream){

   using jo::operator<<;

   stream << "{\n";

     // Flops
     stream << jo::jkey({"devices"}) << inp.device_ids << ",\n";
     stream << jo::jkey({"perf"}) << "[\n";
       for (int i=0; i<inp.device_ids.size(); ++i){
         stream << "{\n";
           auto const id = inp.device_ids[i];
           stream << jo::jkey({"id"}) << i << ",";
           stream << jo::jkey({"t"}) << map.at(id).time_points << ",";
           stream << jo::jkey({"flops"}) << map.at(id).flops;
         stream << "}\n" << ((i < inp.device_ids.size()-1) ? "," : "");
       }
     stream << "],\n";

     // input
     stream << jo::jkey({"args"}) << inp;

   stream << "}";
}


void
serialize_csv(
      std::unordered_map<int, gemm_results> const& map,
      args const& inp,
      std::ostream &stream){

   // Write headers
   std::vector<std::string> cols;
   for (auto it=map.begin(); it!=map.end(); ++it){
     auto const dev_id = it->first;
     auto const& data = it->second;
     cols.push_back("t_" + std::to_string(dev_id));
     cols.push_back("flops_" + std::to_string(dev_id));
   }
   auto const header_str = join(cols, ",");
   stream << header_str << "\n";

   auto const N = max_size(map, [](gemm_results const& elem){
      return elem.flops; }
   );

   // Write data
   for (size_t i=0; i<N; ++i){
     std::vector<std::string> row;

     for (auto it=map.begin(); it!=map.end(); ++it){
       auto const dev_id = it->first;
       auto const& data = it->second;

       std::string t, x;
       if (i < data.time_points.size()){
         t = "\"" + data.time_points[i] + "\"";
         x = std::to_string(data.flops[i]);
       }

       row.push_back(t);
       row.push_back(x);
     }

     stream << join(row, ",") << "\n";
   }
}
