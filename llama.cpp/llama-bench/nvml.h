#pragma once

typedef void* nvmlDevice_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
} nvmlProcessInfo_v1_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
} nvmlProcessInfo_v3_t;

bool nvml_init();

bool nvml_get_device(nvmlDevice_t *device, unsigned int index);

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy);

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power);

bool nvml_get_memory_usage(nvmlDevice_t device, float *memory);

bool nvml_shutdown();