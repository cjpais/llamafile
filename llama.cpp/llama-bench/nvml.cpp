#include <cosmo.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include "nvml.h"
#include "llama.cpp/common.h"

#define IMPORT_NVML_FUNCTION(func_name, func_type) \
    do { \
        void* func_ptr = imp(lib, #func_name); \
        if (!func_ptr) { \
            ok = false; \
        } else if (IsWindows()) { \
            nvml.func_name.windows_abi = reinterpret_cast<__attribute__((__ms_abi__)) func_type>(func_ptr); \
            ok &= true; \
        } else { \
            nvml.func_name.default_abi = reinterpret_cast<func_type>(func_ptr); \
            ok &= true; \
        } \
    } while(0)

#define NVML_FUNCTION_CALL(FUNC, ...) \
  (IsWindows() ? nvml.FUNC.windows_abi(__VA_ARGS__) : nvml.FUNC.default_abi(__VA_ARGS__))

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static struct Nvml {
    union {
        int (*default_abi)(void);
        int (__attribute__((__ms_abi__)) *windows_abi)(void);
    } nvmlInit_v2;

    union {
        int (*default_abi)(unsigned int *deviceCount);
        int (__attribute__((__ms_abi__)) *windows_abi)(unsigned int *deviceCount);
    } nvmlDeviceGetCount_v2;

    union {
        int (*default_abi)(unsigned int index, void **device);
        int (__attribute__((__ms_abi__)) *windows_abi)(unsigned int index, void **device);
    } nvmlDeviceGetHandleByIndex_v2;

    union {
        int (*default_abi)(void *device, unsigned long long *energy);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned long long *energy);
    } nvmlDeviceGetTotalEnergyConsumption;

    union {
        int (*default_abi)(void *device, unsigned int *power);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned int *power);
    } nvmlDeviceGetPowerUsage;

    union {
        int (*default_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos);
    } nvmlDeviceGetComputeRunningProcesses_v1;

    union {
        int (*default_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);
    } nvmlDeviceGetComputeRunningProcesses_v2;

    union {
        int (*default_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v3_t *infos);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned int *infoCount, nvmlProcessInfo_v3_t *infos);
    } nvmlDeviceGetComputeRunningProcesses_v3;

    union {
        int (*default_abi)(void);
        int (__attribute__((__ms_abi__)) *windows_abi)(void);
    } nvmlShutdown;
} nvml;

static bool FileExists(const char *path) {
    struct stat st;
    return !stat(path, &st);
}

static bool get_nvml_bin_path(char path[PATH_MAX]) {
    // create filename of executable
    char name[NAME_MAX];
    if (IsWindows())
        strlcpy(name, "nvml.dll", PATH_MAX);
    else
        strlcpy(name, "libnvidia-ml.so", PATH_MAX);

    printf("Looking for nvml library: %s\n", name);

    // search for it on $PATH
    if (commandv(name, path, PATH_MAX)) {
        printf("Found nvml on path: %s\n", path);
        return path;
    }

    // if on windows look for the following paths
    // Standard driver install: %ProgramW6432%\"NVIDIA Corporation"\NVSMI\
    // DCH driver install: \Windows\System32
    if (IsWindows()) {
        const char *program_files = getenv("ProgramW6432");
        if (!program_files) {
            tinylog(__func__, ": note: $ProgramW6432 not set\n", NULL);
            return false;
        }
        strlcpy(path, program_files, PATH_MAX);
        strlcat(path, "\\NVIDIA Corporation\\NVSMI\\", PATH_MAX);
        strlcat(path, name, PATH_MAX);
        printf("Attempting to load %s\n", path);
        if (FileExists(path)) {
            return true;
        } else {
            tinylog(__func__, ": note: ", path, " does not exist\n", NULL);
            strlcpy(path, "C:\\Windows\\System32\\", PATH_MAX);
            strlcat(path, name, PATH_MAX);
            if (FileExists(path)) {
                return true;
            } else {
                tinylog(__func__, ": note: ", path, " does not exist\n", NULL);
                return false;
            }
        }
    } else {
        // just return "libnvidia-ml.so" for now
        strlcpy(path, name, PATH_MAX);
        return true;
    }
}

bool nvml_init() {
    char dso[PATH_MAX];
    if (!get_nvml_bin_path(dso)) {
        tinylog(__func__, ": error: failed to find nvml library\n", NULL);
        return false;
    }

    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
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

    NVML_FUNCTION_CALL(nvmlInit_v2);
    unsigned int deviceCount;
    NVML_FUNCTION_CALL(nvmlDeviceGetCount_v2, &deviceCount);

    return true;
}

bool nvml_get_device(nvmlDevice_t *device) {
    NVML_FUNCTION_CALL(nvmlDeviceGetHandleByIndex_v2, 0, device);
    return true;
}

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power) {
    NVML_FUNCTION_CALL(nvmlDeviceGetPowerUsage, device, power);
    return true;
}

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy) {
    NVML_FUNCTION_CALL(nvmlDeviceGetTotalEnergyConsumption, device, energy);
    return true;
}

// return memory used in MiB
// bool nvml_get_memory_usage(nvmlDevice_t device, float *memory) {
//     unsigned int infoCount;
//     nvmlProcessInfo_v2_t *proc;
//     float vramUsed = 0;

//     int status = nvml.nvmlDeviceGetComputeRunningProcesses_v2(device, &infoCount, NULL);
//     proc = (nvmlProcessInfo_v2_t *)malloc(sizeof(nvmlProcessInfo_v2_t) * (infoCount + 1));
//     status = nvml.nvmlDeviceGetComputeRunningProcesses_v2(device, &infoCount, proc);

//     if (status != 0) {
//         printf("status: %d. infoCount %d\n", status, infoCount);
//         tinylog(__func__, ": error: failed to get process count\n", NULL);
//         return false;
//     }

//     for (unsigned int i = 0; i < infoCount; i++) {
//         vramUsed += (float)proc[i].usedGpuMemory / 1024.0 / 1024.0 ;
//     }


//     *memory = vramUsed;

//     return true;
// }

bool nvml_shutdown() {
    NVML_FUNCTION_CALL(nvmlShutdown);
    return true;
}
