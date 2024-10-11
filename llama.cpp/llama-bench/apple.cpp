
#include <cosmo.h>
#include <dlfcn.h>

#include "apple.h"
#include "llama.cpp/common.h"

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

typedef void* CFStringRef;
typedef void* CFDictionaryRef;
typedef void* CFMutableDictionaryRef;
typedef void* CFTypeRef;
typedef void* CFArrayRef;
typedef void* IOReportSubscriptionRef;
typedef void* CVoidRef;
typedef int CFIndex;

static struct IOReport {
    CFDictionaryRef (*IOReportCopyChannelsInGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
    IOReportSubscriptionRef (*IOReportCreateSubscription)(CVoidRef, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
    CFDictionaryRef (*IOReportCreateSamples)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
    CFDictionaryRef (*IOReportCreateSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
    CFStringRef (*IOReportChannelGetChannelName)(CFDictionaryRef);
    int64_t (*IOReportSimpleGetIntegerValue)(CFDictionaryRef, int32_t);
    CFStringRef (*IOReportChannelGetUnitLabel)(CFDictionaryRef);
} io_report;

static struct CoreFoundation {
    CFMutableDictionaryRef (*CFDictionaryCreateMutableCopy)(CVoidRef, long, CFDictionaryRef);
    long (*CFDictionaryGetCount)(CFDictionaryRef);
    void (*CFShow)(CFTypeRef);
    CVoidRef (*CFDictionaryGetValue)(CFDictionaryRef, CVoidRef);
    CFStringRef (*CFStringCreateWithCString)(CVoidRef, const char*, int);
    void (*CFRelease)(CFTypeRef); 
    int (*CFArrayGetCount)(CFArrayRef);
    CFTypeRef (*CFArrayGetValueAtIndex)(CFArrayRef, int);
    CFArrayRef (*CFArrayCreateCopy)(CVoidRef, CFArrayRef);
    bool (*CFStringGetCString)(CFStringRef, char *, int, int);
} core_foundation;

void* load_lib_ioreport() {
    void *lib = cosmo_dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY);
    if (!lib) {
        tinylog(__func__, ": error: failed to open IOKit framework\n", NULL);
        return NULL;
    }

    bool ok = true;

    printf("LOADING IOREPORT\n");

    ok &= !!(io_report.IOReportCopyChannelsInGroup = (CFDictionaryRef (*)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t))imp(lib, "IOReportCopyChannelsInGroup"));
    ok &= !!(io_report.IOReportCreateSubscription = (IOReportSubscriptionRef (*)(CVoidRef, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef))imp(lib, "IOReportCreateSubscription"));
    ok &= !!(io_report.IOReportCreateSamples = (CFDictionaryRef (*)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef))imp(lib, "IOReportCreateSamples"));
    ok &= !!(io_report.IOReportCreateSamplesDelta = (CFDictionaryRef (*)(CFDictionaryRef, CFDictionaryRef, CFTypeRef))imp(lib, "IOReportCreateSamplesDelta"));
    ok &= !!(io_report.IOReportChannelGetChannelName = (CFStringRef (*)(CFDictionaryRef))imp(lib, "IOReportChannelGetChannelName"));
    ok &= !!(io_report.IOReportSimpleGetIntegerValue = (int64_t (*)(CFDictionaryRef, int32_t))imp(lib, "IOReportSimpleGetIntegerValue"));
    ok &= !!(io_report.IOReportChannelGetUnitLabel = (CFStringRef (*)(CFDictionaryRef))imp(lib, "IOReportChannelGetUnitLabel"));

    if (!ok) {
        tinylog(__func__, ": error: not all IOReport symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return NULL;
    }

    printf("IOReport initialized successfully\n");
    printf("LOADING COREFOUNDATION\n");

    ok &= !!(core_foundation.CFDictionaryCreateMutableCopy = (CFMutableDictionaryRef (*)(void*, long, CFDictionaryRef))imp(lib, "CFDictionaryCreateMutableCopy"));
    ok &= !!(core_foundation.CFDictionaryGetCount = (long (*)(CFDictionaryRef))imp(lib, "CFDictionaryGetCount"));
    ok &= !!(core_foundation.CFShow = (void (*)(CFTypeRef))imp(lib, "CFShow"));
    ok &= !!(core_foundation.CFDictionaryGetValue = ( void* (*)(CFDictionaryRef,  void*))imp(lib, "CFDictionaryGetValue"));
    ok &= !!(core_foundation.CFStringCreateWithCString = (CFStringRef (*)(CVoidRef, const char*, int))imp(lib, "CFStringCreateWithCString"));
    ok &= !!(core_foundation.CFRelease = (void (*)(CFTypeRef))imp(lib, "CFRelease"));
    ok &= !!(core_foundation.CFArrayGetCount = (int (*)(CFArrayRef))imp(lib, "CFArrayGetCount"));
    ok &= !!(core_foundation.CFArrayGetValueAtIndex = (CFTypeRef (*)(CFArrayRef, int))imp(lib, "CFArrayGetValueAtIndex"));
    ok &= !!(core_foundation.CFArrayCreateCopy = (CFArrayRef (*)(CVoidRef, CFArrayRef))imp(lib, "CFArrayCreateCopy"));
    ok &= !!(core_foundation.CFStringGetCString = (bool (*)(CFStringRef, char *, int, int))imp(lib, "CFStringGetCString"));

    if (!ok) {
        tinylog(__func__, ": error: not all CoreFoundation symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return NULL;
    }

    printf("CoreFoundation initialized successfully\n");
    return lib;
}

CFMutableDictionaryRef get_power_channels() {
    CFStringRef energy_str = core_foundation.CFStringCreateWithCString(NULL, "Energy Model", 0x08000100);
    CFDictionaryRef channels = io_report.IOReportCopyChannelsInGroup(energy_str, NULL, 0, 0, 0);
    core_foundation.CFRelease(energy_str);

    CFMutableDictionaryRef channels_mut = core_foundation.CFDictionaryCreateMutableCopy(NULL, core_foundation.CFDictionaryGetCount(channels), channels);
    core_foundation.CFRelease(channels);

    return channels_mut;
}

IOReportSubscriptionRef get_subscription(CFMutableDictionaryRef channels_mut) {
    CFMutableDictionaryRef subscription;
    IOReportSubscriptionRef s = io_report.IOReportCreateSubscription(NULL, channels_mut, &subscription, 0, NULL);
    return s;
}

CFArrayRef sample_power(IOReportSubscriptionRef sub, CFMutableDictionaryRef channels, int dur_ms) {
    CFDictionaryRef sample_start = io_report.IOReportCreateSamples(sub, channels, NULL);
    printf("Sleeping\n");
    usleep(dur_ms * 1000); // Convert ms to microseconds
    printf("Awoke\n");
    CFDictionaryRef sample_end = io_report.IOReportCreateSamples(sub, channels, NULL);
    CFDictionaryRef samp_delta = io_report.IOReportCreateSamplesDelta(sample_start, sample_end, NULL);

    core_foundation.CFRelease(sample_start);
    core_foundation.CFRelease(sample_end);

    CFStringRef key = core_foundation.CFStringCreateWithCString(NULL, "IOReportChannels", 0x08000100);
    CFArrayRef report = core_foundation.CFDictionaryGetValue(samp_delta, key);
    CFArrayRef reportCopy = core_foundation.CFArrayCreateCopy(NULL, report);

    core_foundation.CFRelease(key);
    core_foundation.CFRelease(samp_delta);

    return reportCopy;
}

static void print_channel_energy(const char* buffer, double energy, const char* unit) {
    double joules;
    if (strcmp(unit, "mJ") == 0) {
        joules = energy / 1e3;
    } else if (strcmp(unit, "uJ") == 0) {
        joules = energy / 1e6;
    } else if (strcmp(unit, "nJ") == 0) {
        joules = energy / 1e9;
    } else {
        printf("Unknown unit: %s\n", unit);
        return;
    }

    printf("%s Energy: %.0f. Energy: %.2fJ. Watts: %.2fW\n", buffer, energy, joules, joules);
}

static bool get_cstring_from_cfstring(CFStringRef cfString, char* buffer, size_t bufferSize) {
    return core_foundation.CFStringGetCString(cfString, buffer, bufferSize, 0x08000100);
}

static char* get_unit_label(CFDictionaryRef item) {
    static char unit[64];
    CFStringRef u = io_report.IOReportChannelGetUnitLabel(item);
    if (u) {
        if (!get_cstring_from_cfstring(u, unit, sizeof(unit))) {
            strcpy(unit, "Unknown");
        }
        core_foundation.CFRelease(u);
    } else {
        strcpy(unit, "N/A");
    }
    return unit;
}

static void process_channel(CFDictionaryRef item) {
    CFStringRef n = io_report.IOReportChannelGetChannelName(item);
    if (!n) {
        printf("Failed to get channel name\n");
        return;
    }

    char name[64] = {0};
    if (!get_cstring_from_cfstring(n, name, sizeof(name))) {
        printf("Failed to get channel name\n");
        core_foundation.CFRelease(n);
        return;
    }

    char* unit = get_unit_label(item);

    if (strcmp(name, "CPU Energy") == 0 || strcmp(name, "GPU Energy") == 0 || strstr(name, "ANE") != NULL) {
        double energy = (double)io_report.IOReportSimpleGetIntegerValue(item, 0);
        print_channel_energy(name, energy, unit);
    }

    core_foundation.CFRelease(n);
}

static void print_samples(CFArrayRef samples) {
    CFIndex count = core_foundation.CFArrayGetCount(samples);
    for (CFIndex i = 0; i < count; i++) {
        CFDictionaryRef item = core_foundation.CFArrayGetValueAtIndex(samples, i);
        process_channel(item);
    }
}

void get_ioreport_info() {
    void* lib = load_lib_ioreport();
    if (!lib) {
        printf("Failed to load IOReport library\n");
        return;
    }


    CFMutableDictionaryRef power_channel = get_power_channels();
    if (!power_channel) {
        printf("Failed to get power channels\n");
        cosmo_dlclose(lib);
        return;
    }

    IOReportSubscriptionRef sub = get_subscription(power_channel);
    if (!sub) {
        printf("Failed to create subscription\n");
        core_foundation.CFRelease(power_channel);
        cosmo_dlclose(lib);
        return;
    }

    CFArrayRef sample;
    for (int i = 0; i < 30; i++) {
        sample = sample_power(sub, power_channel, 1000);
        print_samples(sample);
    }

    core_foundation.CFRelease(power_channel);
    core_foundation.CFRelease(sub);
    core_foundation.CFRelease(sample);

    cosmo_dlclose(lib);
}
