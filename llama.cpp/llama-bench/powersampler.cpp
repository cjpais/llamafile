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
        pthread_join(sampling_thread_, nullptr);

        // average the samples
        double total_watts = 0;
        for (double watts : samples_) {
            total_watts += watts;
        }
        double avg_watts = total_watts / samples_.size();
        printf("Average power consumption from samples: %.2f W\n", avg_watts);
        printf("Total energy consumed: %.2f J in %d ms\n", energy_consumed_end - energy_consumed_start_, timespec_tomillis(timespec_sub(sampling_end_time_, sampling_start_time_)));
        printf("Average power from energy consumed: %.2f W\n", (energy_consumed_end - energy_consumed_start_) / (timespec_tomillis(timespec_sub(sampling_end_time_, sampling_start_time_)) / 1000));
    }
}

void* PowerSampler::sampling_thread_func(void* arg) {
    PowerSampler* sampler = static_cast<PowerSampler*>(arg);
    while (sampler->is_sampling_) {
        usleep(sampler->sample_length_ms_ * 1000); // Convert ms to microseconds

        // on the first iteration wait 100ms to make sure the system gets something reasonable for us.
        double power = sampler->getInstantaneousPower();
        fprintf(stderr, "Power: %.2f W\n", power);

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

// TODO there is a more consise way of doing this for sure.
double NvidiaPowerSampler::getInstantaneousPower() {
    unsigned int power;
    nvml_get_power_usage(device_, &power);
    return (double)power / 1000.0;
}

double NvidiaPowerSampler::getEnergyConsumed() {
    unsigned long long energy;
    nvml_get_energy_consumption(device_, &energy);
    return (double)energy / 1000.0;
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
    return rsmi_get_power();
}

double AMDPowerSampler::getEnergyConsumed() {
    return rsmi_get_power_instant();
}

// ApplePowerSampler implementation

ApplePowerSampler::ApplePowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        init_apple_mon();
        power_channel_ = get_power_channels();
        sub_ = get_subscription(power_channel_);
    }

ApplePowerSampler::~ApplePowerSampler() {
    am_release(power_channel_);
    am_release(sub_);
}

double ApplePowerSampler::getInstantaneousPower() {
    // Placeholder implementation
    // print_object(sample);
    return 0; // Example value
}

// TODO this needs to be a void*?
double ApplePowerSampler::getEnergyConsumed() {
    CFDictionaryRef sample = sample_power(sub_, power_channel_);
    double millijoules = sample_to_millijoules(sample);
    printf("Millijoules: %.2f\n", millijoules);
    return millijoules / 1000.0;
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
