#include <cosmo.h>
#include <dlfcn.h>

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
} rsmi_lib;