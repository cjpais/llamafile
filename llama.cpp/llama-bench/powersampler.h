#pragma once

#include <pthread.h>
#include <time.h>
#include <vector>
#include "nvml.h"
#include "rsmi.h"
#include "apple.h"

struct PowerSampler {
    // vars
    long sample_length_ms_;

    timespec sampling_start_time_;
    timespec sampling_end_time_;
    double energy_consumed_start_;

    std::vector<double> samples_;

    bool is_sampling_;
    pthread_t sampling_thread_;
    mutable pthread_mutex_t samples_mutex_;

    // funcs
    PowerSampler(long sample_length_ms);
    virtual ~PowerSampler();

    void start();
    void stop();

    // this returns the instantaneous power in microwatts
    virtual double getInstantaneousPower() = 0;
    
    // this returns the energy consumed in millijoules
    virtual double getEnergyConsumed() = 0;

private:
    static void* sampling_thread_func(void* arg);
};

struct NvidiaPowerSampler : public PowerSampler {
    nvmlDevice_t device_;
    unsigned long long start_joules_;
    unsigned long long end_joules_;

    NvidiaPowerSampler(long sample_length_ms);
    ~NvidiaPowerSampler() override;

protected:
    // void startSampling() override;
    // void stopSampling() override;
    // PowerSample computeSample() override;
    double getInstantaneousPower() override;
    double getEnergyConsumed() override;
};

struct AMDPowerSampler : public PowerSampler {
    AMDPowerSampler(long sample_length_ms);
    ~AMDPowerSampler() override;

protected:
    // void startSampling() override;
    // void stopSampling() override;
    // PowerSample computeSample() override;
    double getInstantaneousPower() override;
    double getEnergyConsumed() override;
};

struct ApplePowerSampler : public PowerSampler {
    CFMutableDictionaryRef power_channel_;
    IOReportSubscriptionRef sub_;

    ApplePowerSampler(long sample_length_ms);
    ~ApplePowerSampler() override;

protected:
    // void startSampling() override;
    // void stopSampling() override;
    // PowerSample computeSample() override;
    double getInstantaneousPower() override;
    double getEnergyConsumed() override;
};

PowerSampler* getPowerSampler(long sample_length_ms);
