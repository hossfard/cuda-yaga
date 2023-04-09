#include "dquery.h"
#include "utils.h"



device_info
nv_device_info(unsigned int id){
   nvmlctx ctx;
   device_info ret;
   nvmlDevice_t device;
   ret.id = id;
   nvmlDeviceGetHandleByIndex_v2(id, &device);
   nvmlDeviceGetPciInfo(device, &ret.pci);
   nvmlDeviceGetNumFans(device, &ret.fan_count);
   nvmlDeviceGetName(device, ret.name, NVML_DEVICE_NAME_BUFFER_SIZE);
   nvmlDeviceGetVbiosVersion(device, ret.vbios, NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE);

   return ret;
}


std::vector<device_info>
nv_device_list_info(std::vector<int> ids){
   std::vector<device_info> ret;
   for (auto id : ids){
     ret.push_back(nv_device_info(id));
   }
   return ret;
}


sys_info
nv_sys_info(){
   nvmlctx ctx;
   sys_info ret;
   nvmlSystemGetDriverVersion(ret.driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
   nvmlSystemGetCudaDriverVersion_v2(&ret.cuda_driver);
   return ret;
}


dstate_snapshot
snapshot(unsigned int device_id){
   nvmlDevice_t device;
   nv_check_err(nvmlDeviceGetHandleByIndex_v2(device_id, &device));
   return snapshot(device);
}


dstate_snapshot
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


nvmlReturn_t
nv_check_err(nvmlReturn_t stat){
   if (stat != NVML_SUCCESS){
     printf("Failed to initialize NVML: %s\n", nvmlErrorString(stat));
     exit(1);
   }

   return stat;
}
