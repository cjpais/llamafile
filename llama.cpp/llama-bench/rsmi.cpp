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
    int (*rsmi_dev_current_socket_power_get)(uint32_t dv_ind, uint64_t *power); // in uW
    int (*rsmi_dev_power_ave_get)(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power);
    int (*rsmi_dev_energy_count_get)(uint32_t dv_ind, uint64_t *power, float *counter_resolution, uint64_t *timestamp);
    int (*rsmi_shut_down)(void);
} rsmi;

void rsmi_init() {
    // TODO find across platforms.
    void *lib = cosmo_dlopen("/opt/rocm/lib/librocm_smi64.so", RTLD_LAZY);

    bool ok = true;

    ok &= !!(rsmi.rsmi_init = reinterpret_cast<int (*)(uint64_t)>(imp(lib, "rsmi_init")));
    ok &= !!(rsmi.rsmi_num_monitor_devices = reinterpret_cast<int (*)(uint32_t*)>(imp(lib, "rsmi_num_monitor_devices")));
    ok &= !!(rsmi.rsmi_dev_id_get = reinterpret_cast<int (*)(uint32_t, uint16_t*)>(imp(lib, "rsmi_dev_id_get")));
    ok &= !!(rsmi.rsmi_dev_power_get = reinterpret_cast<int (*)(uint32_t, uint64_t*, RSMI_POWER_TYPE*)>(imp(lib, "rsmi_dev_power_get")));
    ok &= !!(rsmi.rsmi_dev_current_socket_power_get = reinterpret_cast<int (*)(uint32_t, uint64_t*)>(imp(lib, "rsmi_dev_current_socket_power_get")));
    ok &= !!(rsmi.rsmi_dev_power_ave_get = reinterpret_cast<int (*)(uint32_t, uint32_t, uint64_t*)>(imp(lib, "rsmi_dev_power_ave_get")));
    ok &= !!(rsmi.rsmi_dev_energy_count_get = reinterpret_cast<int (*)(uint32_t, uint64_t*, float*, uint64_t*)>(imp(lib, "rsmi_dev_energy_count_get")));
    ok &= !!(rsmi.rsmi_shut_down = reinterpret_cast<int (*)()>(imp(lib, "rsmi_shut_down")));

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
}

double rsmi_get_avg_power() {
    uint64_t power;

    int status = rsmi.rsmi_dev_power_ave_get(0, 0, &power);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get power\n", NULL);
        return 0;
    }

    return (double)power;
}

double rsmi_get_power() {
    uint64_t power;
    RSMI_POWER_TYPE type;

    printf("rsmi_get_power\n");
    int status = rsmi.rsmi_dev_power_get(0, &power, &type);
    printf("rsmi_get_power status: %d\n", status);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get power\n", NULL);
        return 0;
    }

    return (double)power;
}

double rsmi_dev_energy_count_get() {
    uint64_t power;
    float counter_resolution;
    uint64_t timestamp;

    int status = rsmi.rsmi_dev_energy_count_get(0, &power, &counter_resolution, &timestamp);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get power\n", NULL);
        return 0;
    }

    return (double)power;
}

double rsmi_get_power_instant() {
    uint64_t power;

    int status = rsmi.rsmi_dev_current_socket_power_get(0, &power);
    if (status != 0) {
        tinylog(__func__, ": error: failed to get power\n", NULL);
        return 0;
    }

    return (double)power;
}

void rsmi_shutdown() {
    int status = rsmi.rsmi_shut_down();
    if (status != 0) {
        tinylog(__func__, ": error: failed to shutdown ROCm SMI\n", NULL);
    }
}