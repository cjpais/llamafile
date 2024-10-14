#include <cosmo.h>
#include <dlfcn.h>

#include "nvml.h"
#include "llama.cpp/common.h"

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
    int (*nvmlShutdown)(void);
    // TODO? nvmlDeviceGetPowerManagementLimit or similar
} nvml;

bool nvml_init() {
    void *lib = cosmo_dlopen("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so", RTLD_LAZY);
    bool ok = true;

    ok &= !!(nvml.nvmlInit_v2 = reinterpret_cast<int (*)(void)>(imp(lib, "nvmlInit_v2")));
    ok &= !!(nvml.nvmlDeviceGetCount_v2 = reinterpret_cast<int (*)(unsigned int*)>(imp(lib, "nvmlDeviceGetCount_v2")));
    ok &= !!(nvml.nvmlDeviceGetHandleByIndex_v2 = reinterpret_cast<int (*)(unsigned int, void**)>(imp(lib, "nvmlDeviceGetHandleByIndex_v2")));
    ok &= !!(nvml.nvmlDeviceGetTotalEnergyConsumption = reinterpret_cast<int (*)(void*, unsigned long long*)>(imp(lib, "nvmlDeviceGetTotalEnergyConsumption")));
    ok &= !!(nvml.nvmlDeviceGetPowerUsage = reinterpret_cast<int (*)(void*, unsigned int*)>(imp(lib, "nvmlDeviceGetPowerUsage")));
    ok &= !!(nvml.nvmlShutdown = reinterpret_cast<int (*)(void)>(imp(lib, "nvmlShutdown")));

    if (!ok) {
        tinylog(__func__, ": error: not all nvml symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    int status = nvml.nvmlInit_v2();
    if (status != 0) {  // Assuming 0 is success, which is common for many APIs
        tinylog(__func__, ": error: failed to initialize NVML\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    return true;
}

// TODO should wrap these in a macro or function.
bool nvml_get_device(nvmlDevice_t *device) {
    if (!nvml.nvmlDeviceGetHandleByIndex_v2) {
        tinylog(__func__, ": error: nvmlDeviceGetHandleByIndex_v2 not imported\n", NULL);
        return false;
    }

    int status = nvml.nvmlDeviceGetHandleByIndex_v2(0, device);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get device\n", NULL);
        return false;
    }
    return true;
}

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power) {
    if (!nvml.nvmlDeviceGetPowerUsage) {
        tinylog(__func__, ": error: nvmlDeviceGetPowerUsage not imported\n", NULL);
        return false;
    }

    int status = nvml.nvmlDeviceGetPowerUsage(device, power);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get power usage\n", NULL);
        return false;
    }
    return true;
}

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy) {
    if (!nvml.nvmlDeviceGetTotalEnergyConsumption) {
        tinylog(__func__, ": error: nvmlDeviceGetTotalEnergyConsumption not imported\n", NULL);
        return false;
    }

    int status = nvml.nvmlDeviceGetTotalEnergyConsumption(device, energy);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get energy consumption\n", NULL);
        return false;
    }
    return true;
}

bool nvml_shutdown() {
    // check the nvml library is initialized
    // TODO validate this by calling shutdown before init
    if (!nvml.nvmlShutdown) {
        tinylog(__func__, ": error: NVML library not initialized\n", NULL);
        return false;
    }

    int status = nvml.nvmlShutdown();
    if (status != 0) {
        tinylog(__func__, ": error: failed to shutdown NVML\n", NULL);
        return false;
    }
    return true;
}

// void get_nvml_info() {
//     printf("===== NVML information =====\n\n");
//     // TODO find



//     int status = nvml.nvmlInit_v2();
//     if (status != 0) {  // Assuming 0 is success, which is common for many APIs
//         tinylog(__func__, ": error: failed to initialize NVML\n", NULL);
//         cosmo_dlclose(lib);
//         return;
//     }
//     unsigned int device_count;
//     status = nvml.nvmlDeviceGetCount_v2(&device_count);
//     printf("Number of devices: %d\n", device_count);

//     for (int i = 0; i < device_count; i++) {
//         void *device;
//         status = nvml.nvmlDeviceGetHandleByIndex_v2(i, &device);
//         unsigned long long energy;
//         status = nvml.nvmlDeviceGetTotalEnergyConsumption(device, &energy);
//         printf("Energy: %llu\n", energy);
//         unsigned int power;
//         status = nvml.nvmlDeviceGetPowerUsage(device, &power);
//         printf("Power: %u\n", power);
//     }

//     printf("NVML initialized successfully\n");
//     cosmo_dlclose(lib);
// }