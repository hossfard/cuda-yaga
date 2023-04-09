#ifndef DQUERY_H_
#define DQUERY_H_

#include <vector>
#include <string>
#include <nvml.h>



nvmlReturn_t
nv_check_err(nvmlReturn_t stat);


struct dstate_snapshot{
  std::vector<unsigned int> fans;
  std::string  t;
  unsigned int temperature;
  unsigned int power;
  unsigned int memClock;
  unsigned int smClock;
};


struct device_info{
  unsigned int id;
  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  char vbios[NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE];
  nvmlPciInfo_t pci;
  unsigned int fan_count;
};


struct sys_info{
  int cuda_driver;
  char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
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


dstate_snapshot
snapshot(nvmlDevice_t device);


dstate_snapshot
snapshot(unsigned int device_id);


sys_info
nv_sys_info();


device_info
nv_device_info(unsigned int id);


std::vector<device_info>
nv_device_list_info(std::vector<int> ids);


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
