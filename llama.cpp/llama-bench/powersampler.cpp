#include "powersampler.h"
#include <unistd.h>

#include "llamafile/llamafile.h"
#include "nvml.h"


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
        pthread_create(&sampling_thread_, nullptr, sampling_thread_func, this);
    }
}

void PowerSampler::stop() {
    if (is_sampling_) {
        is_sampling_ = false;
        pthread_join(sampling_thread_, nullptr);
    }
}

std::vector<PowerSample> PowerSampler::getSamples() const {
    std::vector<PowerSample> samples;
    pthread_mutex_lock(&samples_mutex_);
    samples = samples_;
    pthread_mutex_unlock(&samples_mutex_);
    return samples;
}

void* PowerSampler::sampling_thread_func(void* arg) {
    PowerSampler* sampler = static_cast<PowerSampler*>(arg);
    while (sampler->is_sampling_) {
        printf("Sampling power...\n");
        PowerSample sample = sampler->sample();
        pthread_mutex_lock(&sampler->samples_mutex_);
        sampler->samples_.push_back(sample);
        pthread_mutex_unlock(&sampler->samples_mutex_);
        usleep(sampler->sample_length_ms_ * 1000); // Convert ms to microseconds
    }
    return nullptr;
}

// NvidiaPowerSampler implementation

NvidiaPowerSampler::NvidiaPowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        nvml_init();
    }

NvidiaPowerSampler::~NvidiaPowerSampler() {
    nvml_shutdown();
}

PowerSample NvidiaPowerSampler::sample() const {
    // Placeholder implementation
    PowerSample sample;
    sample.watts = 100.0; // Example value
    sample.start_time = time(nullptr);
    sample.end_time = sample.start_time + sample_length_ms_ / 1000;
    return sample;
}

// AMDPowerSampler implementation

AMDPowerSampler::AMDPowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {}

AMDPowerSampler::~AMDPowerSampler() {}

PowerSample AMDPowerSampler::sample() const {
    // Placeholder implementation
    PowerSample sample;
    sample.watts = 90.0; // Example value
    sample.start_time = time(nullptr);
    sample.end_time = sample.start_time + sample_length_ms_ / 1000;
    return sample;
}

// ApplePowerSampler implementation

ApplePowerSampler::ApplePowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {}

ApplePowerSampler::~ApplePowerSampler() {}

PowerSample ApplePowerSampler::sample() const {
    // Placeholder implementation
    PowerSample sample;
    sample.watts = 80.0; // Example value
    sample.start_time = time(nullptr);
    sample.end_time = sample.start_time + sample_length_ms_ / 1000;
    return sample;
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
