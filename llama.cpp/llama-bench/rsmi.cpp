#include <cosmo.h>
#include <dlfcn.h>

#include "rsmi.h"
#include "llama.cpp/common.h"

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

typedef enum {
  RSMI_AVERAGE_POWER = 0,            //!< Average Power
  RSMI_CURRENT_POWER,                //!< Current / Instant Power
  RSMI_INVALID_POWER = 0xFFFFFFFF    //!< Invalid / Undetected Power
} RSMI_POWER_TYPE;

static struct Rsmi {
    int (*rsmi_init)(uint64_t init_flags);
    int (*rsmi_num_monitor_devices)(uint32_t *num_devices);
    int (*rsmi_dev_id_get)(uint32_t dv_ind, uint16_t *id);
    int (*rsmi_dev_power_get)(uint32_t dv_ind, uint64_t *power, RSMI_POWER_TYPE *type);
} rsmi;

void get_rocm_smi_info() {
    printf("===== ROCm SMI information =====\n\n");
    // TODO find
    void *lib = cosmo_dlopen("/opt/rocm/lib/librocm_smi64.so", RTLD_LAZY);

    bool ok = true;

    ok &= !!(rsmi.rsmi_init = reinterpret_cast<int (*)(uint64_t)>(imp(lib, "rsmi_init")));
    ok &= !!(rsmi.rsmi_num_monitor_devices = reinterpret_cast<int (*)(uint32_t*)>(imp(lib, "rsmi_num_monitor_devices")));
    ok &= !!(rsmi.rsmi_dev_id_get = reinterpret_cast<int (*)(uint32_t, uint16_t*)>(imp(lib, "rsmi_dev_id_get")));
    ok &= !!(rsmi.rsmi_dev_power_get = reinterpret_cast<int (*)(uint32_t, uint64_t*, RSMI_POWER_TYPE*)>(imp(lib, "rsmi_dev_power_get")));

    if (!ok) {
        tinylog(__func__, ": error: not all rocm smi symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return;
    }

    int status = rsmi.rsmi_init(0);
    if (status != 0) {
        tinylog(__func__, ": error: failed to initialize ROCm SMI\n", NULL);
        cosmo_dlclose(lib);
        return;
    }
    uint32_t num_devices;
    uint16_t dev_id;
    uint64_t power;
    RSMI_POWER_TYPE type;
    status = rsmi.rsmi_num_monitor_devices(&num_devices);
    printf("Number of devices: %d\n", num_devices);

    for (int i = 0; i < num_devices; i++) {
        status = rsmi.rsmi_dev_id_get(i, &dev_id);
        printf("Device ID: %d\n", dev_id);
        status = rsmi.rsmi_dev_power_get(i, &power, &type);
        printf("Power: %lu. Type: %d\n", power, type);
    }

    printf("ROCm SMI initialized successfully\n");
    cosmo_dlclose(lib);
}