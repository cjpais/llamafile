#pragma once

#include <pthread.h>
#include <time.h>
#include <vector>

struct PowerSample {
    double watts;
    time_t start_time;
    time_t end_time;
};

struct PowerSampler {
    std::vector<PowerSample> samples_;
    long sample_length_ms_;
    bool is_sampling_;
    pthread_t sampling_thread_;
    mutable pthread_mutex_t samples_mutex_;

    PowerSampler(long sample_length_ms);
    virtual ~PowerSampler();

    void start();
    void stop();
    std::vector<PowerSample> getSamples() const;

    virtual PowerSample sample() const = 0;

private:
    static void* sampling_thread_func(void* arg);
};

struct NvidiaPowerSampler : public PowerSampler {
    NvidiaPowerSampler(long sample_length_ms);
    ~NvidiaPowerSampler() override;

protected:
    PowerSample sample() const override;
};

struct AMDPowerSampler : public PowerSampler {
    AMDPowerSampler(long sample_length_ms);
    ~AMDPowerSampler() override;

protected:
    PowerSample sample() const override;
};

struct ApplePowerSampler : public PowerSampler {
    ApplePowerSampler(long sample_length_ms);
    ~ApplePowerSampler() override;

protected:
    PowerSample sample() const override;
};

PowerSampler* getPowerSampler(long sample_length_ms);
