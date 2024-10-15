#include "powersampler.h"
#include <unistd.h>

#include "llamafile/llamafile.h"

PowerSampler::PowerSampler(long sample_length_ms)
    : sample_length_ms_(sample_length_ms), is_sampling_(false) {
    pthread_mutex_init(&samples_mutex_, nullptr);
}

PowerSampler::~PowerSampler() {
    if (is_sampling_) {
        stop();
    }
    pthread_mutex_destroy(&samples_mutex_);
}

void PowerSampler::start() {
    if (!is_sampling_) {
        is_sampling_ = true;
        samples_.clear();
        sampling_start_time_ = timespec_real();
        energy_consumed_start_ = getEnergyConsumed();
        pthread_create(&sampling_thread_, nullptr, sampling_thread_func, this);
    }
}

void PowerSampler::stop() {
    if (is_sampling_) {
        is_sampling_ = false;
        sampling_end_time_ = timespec_real();
        double energy_consumed_end = getEnergyConsumed();

        long long sampling_time = timespec_tomillis(timespec_sub(sampling_end_time_, sampling_start_time_));
        double energy_consumed = energy_consumed_end - energy_consumed_start_;
        pthread_join(sampling_thread_, nullptr);

        // average the samples
        double total_milliwatts = 0;
        for (double milliwatts : samples_) {
            total_milliwatts += milliwatts;
        }
        double avg_milliwatts = total_milliwatts / samples_.size();
        printf("Average power consumption from samples: %.2f mW, %.2f W\n", avg_milliwatts, avg_milliwatts / 1000);
        printf("Total energy consumed: %.2f mJ, %.2fJ in %d ms\n", energy_consumed, energy_consumed / 1000,  sampling_time);
        printf("Average power from energy consumed: %.2f W \n", energy_consumed / sampling_time);
    }
}

void* PowerSampler::sampling_thread_func(void* arg) {
    PowerSampler* sampler = static_cast<PowerSampler*>(arg);
    while (sampler->is_sampling_) {
        usleep(sampler->sample_length_ms_ * 1000); // Convert ms to microseconds

        // on the first iteration wait 100ms to make sure the system gets something reasonable for us.
        double power = sampler->getInstantaneousPower();
        fprintf(stderr, "Power: %.2fmW %.2fW\n", power, power / 1000);

        pthread_mutex_lock(&sampler->samples_mutex_);
        sampler->samples_.push_back(power);
        pthread_mutex_unlock(&sampler->samples_mutex_);
    }
    return nullptr;
}

// NvidiaPowerSampler implementation

NvidiaPowerSampler::NvidiaPowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        // TODO should validate it worked.
        nvml_init();

        // TODO hardcoded to 0 in nvml
        nvml_get_device(&device_);
    }

NvidiaPowerSampler::~NvidiaPowerSampler() {
    nvml_shutdown();
}

double NvidiaPowerSampler::getInstantaneousPower() {
    unsigned int mw;
    if (!nvml_get_power_usage(device_, &mw)) {
        return 0.0;
    }
    return (double)mw;
}

double NvidiaPowerSampler::getEnergyConsumed() {
    unsigned long long mj;
    if (!nvml_get_energy_consumption(device_, &mj)) {
        return 0.0;
    }
    return (double)mj;
}

// AMDPowerSampler implementation

AMDPowerSampler::AMDPowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        rsmi_init();
    }

AMDPowerSampler::~AMDPowerSampler() {
    rsmi_shutdown();
}

double AMDPowerSampler::getInstantaneousPower() {
    double uw;
    if (!rsmi_get_power(&uw)) {
        return 0.0;
    }
    // Convert microwatts to milliwatts
    return uw / 1000.0;
}

double AMDPowerSampler::getEnergyConsumed() {
    double uj;
    if (!rsmi_get_energy_count(&uj)) {
        return 0.0;
    }
    // Convert microjoules to millijoules
    return uj / 1000.0;
}

// ApplePowerSampler implementation

ApplePowerSampler::ApplePowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        init_apple_mon();
        power_channel_ = am_get_power_channels();
        sub_ = am_get_subscription(power_channel_);
    }

ApplePowerSampler::~ApplePowerSampler() {
    am_release(power_channel_);
    am_release(sub_);
}

double ApplePowerSampler::getInstantaneousPower() {
    return 0;
}

// TODO this needs to be a void*?
double ApplePowerSampler::getEnergyConsumed() {
    CFDictionaryRef sample = am_sample_power(sub_, power_channel_);
    double mj = am_sample_to_millijoules(sample);
    return mj;
}

// Function to get appropriate PowerSampler based on the system
PowerSampler* getPowerSampler(long sample_length_ms) {
    if (llamafile_has_gpu() && FLAG_gpu != LLAMAFILE_GPU_DISABLE) {
        if (llamafile_has_metal()) {
            return new ApplePowerSampler(sample_length_ms);
        } else if (llamafile_has_amd_gpu()) {
            return new AMDPowerSampler(sample_length_ms);
        } else if (llamafile_has_cuda()) {
            return new NvidiaPowerSampler(sample_length_ms);
        }
    }

    return NULL;
}
