#ifndef DQUERY_H_
#define DQUERY_H_

#include <nvml.h>


inline nvmlReturn_t
nv_check_err(nvmlReturn_t stat){
   if (stat != NVML_SUCCESS){
     printf("Failed to initialize NVML: %s\n", nvmlErrorString(stat));
     exit(1);
   }

   return stat;
}


struct dstate_snapshot{
  std::vector<unsigned int> fans;
  std::string  t;
  unsigned int temperature;
  unsigned int power;
  unsigned int memClock;
  unsigned int smClock;
};


struct dstate_snapshots{
  dstate_snapshots(unsigned int id) : id(id) { }

  using uintvec = std::vector<unsigned int>;

  dstate_snapshots& operator+=(dstate_snapshot const& snap){
     fans.push_back(snap.fans);
     t.push_back(snap.t);
     temperature.push_back(snap.temperature);
     power.push_back(snap.power);
     memClock.push_back(snap.memClock);
     smClock.push_back(snap.smClock);
     return *this;
  }

  unsigned int id;
  std::vector<std::string>  t;
  std::vector<uintvec> fans;
  std::vector<unsigned int> temperature;
  std::vector<unsigned int> power;
  std::vector<unsigned int> memClock;
  std::vector<unsigned int> smClock;
};

// using dstate_snapshots = std::vector<dstate_snapshot>;


inline dstate_snapshot
snapshot(nvmlDevice_t device){
   dstate_snapshot ret;

   ret.t = now_str();

   using uint = unsigned int;
   // Temperature
   nv_check_err(
      nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &ret.temperature));

   // Fans
   uint fan_count = -1;
   nv_check_err(nvmlDeviceGetNumFans(device, &fan_count));
   for (unsigned int id=0; id<fan_count; ++id){
     unsigned int speed;
     nv_check_err(nvmlDeviceGetFanSpeed_v2(device, id, &speed));
     ret.fans.push_back(speed);
   }
   // Clocks
   nv_check_err(nvmlDeviceGetClock(
      device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &ret.smClock));
   nv_check_err(nvmlDeviceGetClock(
      device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &ret.memClock));
   // power
   nvmlDeviceGetPowerUsage(device, &ret.power);

   return ret;
}


inline dstate_snapshot
snapshot(unsigned int device_id){
   nvmlDevice_t device;
   nv_check_err(nvmlDeviceGetHandleByIndex_v2(device_id, &device));
   return snapshot(device);
}


class nvmlctx{
public:
  nvmlctx(){
     nv_check_err(nvmlInit());
  }
  nvmlctx(nvmlctx &&other) = delete;
  nvmlctx(nvmlctx const& other) = delete;
  nvmlctx& operator=(nvmlctx const& other) = delete;
  ~nvmlctx(){
     nv_check_err(nvmlShutdown());
  }
};


#endif /* DQUERY_H_ */
