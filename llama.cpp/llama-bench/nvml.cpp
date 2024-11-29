#include <cosmo.h>
#include <dlfcn.h>

#include "nvml.h"
#include "llama.cpp/common.h"

#define IMPORT_NVML_FUNCTION(func_name, func_type) \
    ok &= !!(nvml.func_name = reinterpret_cast<func_type>(imp(lib, #func_name)))

#define NVML_FUNCTION_CALL(func_name, error_msg, ...) \
    do { \
        if (!nvml.func_name) { \
            tinylog(__func__, ": error: " #func_name " not imported\n", NULL); \
            return false; \
        } \
        int status = nvml.func_name(__VA_ARGS__); \
        if (status != 0) { \
            tinylog(__func__, ": error: " error_msg "\n", NULL); \
            return false; \
        } \
    } while(0)

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static struct Nvml {
    int (*nvmlInit_v2)(void);
    int (*nvmlDeviceGetCount_v2)(unsigned int *deviceCount);
    int (*nvmlDeviceGetHandleByIndex_v2)(unsigned int index, void **device);
    int (*nvmlDeviceGetTotalEnergyConsumption)(void *device, unsigned long long *energy);
    int (*nvmlDeviceGetPowerUsage)(void *device, unsigned int *power);
    int (*nvmlDeviceGetComputeRunningProcesses_v1)(void *device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
    int (*nvmlDeviceGetComputeRunningProcesses_v2)(void *device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
    int (*nvmlDeviceGetComputeRunningProcesses_v3)(void *device, unsigned int *infoCount, nvmlProcessInfo_v3_t *infos);
    int (*nvmlShutdown)(void);
} nvml;

bool nvml_init() {
    // TODO go find the .so or .dll
    void *lib = cosmo_dlopen("libnvidia-ml.so", RTLD_LAZY);
    bool ok = true;

    IMPORT_NVML_FUNCTION(nvmlInit_v2, int (*)(void));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetCount_v2, int (*)(unsigned int*));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetHandleByIndex_v2, int (*)(unsigned int, void**));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetTotalEnergyConsumption, int (*)(void*, unsigned long long*));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetPowerUsage, int (*)(void*, unsigned int*));
    // TODO need a smarter way to import them similar to the nvtop code
    // IMPORT_NVML_FUNCTION(nvmlDeviceGetComputeRunningProcesses_v1, int (*)(void*, unsigned int*, nvmlProcessInfo_v1_t*));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetComputeRunningProcesses_v2, int (*)(void*, unsigned int*, nvmlProcessInfo_v2_t*));
    IMPORT_NVML_FUNCTION(nvmlDeviceGetComputeRunningProcesses_v3, int (*)(void*, unsigned int*, nvmlProcessInfo_v3_t*));
    IMPORT_NVML_FUNCTION(nvmlShutdown, int (*)(void));

    if (!ok) {
        tinylog(__func__, ": error: not all nvml symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    NVML_FUNCTION_CALL(nvmlInit_v2, "failed to initialize NVML");

    return true;
}

bool nvml_get_device(nvmlDevice_t *device) {
    NVML_FUNCTION_CALL(nvmlDeviceGetHandleByIndex_v2, "failed to get device", 0, device);
    return true;
}

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power) {
    NVML_FUNCTION_CALL(nvmlDeviceGetPowerUsage, "failed to get power usage", device, power);
    return true;
}

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy) {
    NVML_FUNCTION_CALL(nvmlDeviceGetTotalEnergyConsumption, "failed to get energy consumption", device, energy);
    return true;
}

// return memory used in MiB
bool nvml_get_memory_usage(nvmlDevice_t device, float *memory) {
    unsigned int infoCount;
    nvmlProcessInfo_v2_t *proc;
    float vramUsed = 0;

    int status = nvml.nvmlDeviceGetComputeRunningProcesses_v2(device, &infoCount, NULL);
    proc = (nvmlProcessInfo_v2_t *)malloc(sizeof(nvmlProcessInfo_v2_t) * (infoCount + 1));
    status = nvml.nvmlDeviceGetComputeRunningProcesses_v2(device, &infoCount, proc);

    if (status != 0) {
        printf("status: %d. infoCount %d\n", status, infoCount);
        tinylog(__func__, ": error: failed to get process count\n", NULL);
        return false;
    }

    for (unsigned int i = 0; i < infoCount; i++) {
        vramUsed += (float)proc[i].usedGpuMemory / 1024.0 / 1024.0 ;
    }


    *memory = vramUsed;

    return true;
}

bool nvml_shutdown() {
    if (!nvml.nvmlShutdown) {
        tinylog(__func__, ": error: NVML library not initialized\n", NULL);
        return false;
    }
    NVML_FUNCTION_CALL(nvmlShutdown, "failed to shutdown NVML");
    return true;
}
