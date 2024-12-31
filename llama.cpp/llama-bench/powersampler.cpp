#include "powersampler.h"
#include <unistd.h>
#include <cosmo.h>

#include "llamafile/llamafile.h"
#include "llama.cpp/ggml-metal.h"

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

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 1*1024*1024); // set the stack size to 1MB
        pthread_create(&sampling_thread_, &attr, sampling_thread_func, this);
    }
}

// TODO return a sample, with max vram and avg power
power_sample_t PowerSampler::stop() {
    power_sample_t result = {0.0, 0.0f};
    if (is_sampling_) {
        is_sampling_ = false;
        sampling_end_time_ = timespec_real();
        double energy_consumed_end = getEnergyConsumed();

        long long sampling_time = timespec_tomillis(timespec_sub(sampling_end_time_, sampling_start_time_));
        double energy_consumed = energy_consumed_end - energy_consumed_start_;
        pthread_join(sampling_thread_, nullptr);

        // average the samples
        double total_milliwatts = 0;
        // TODO max vram could be wrong. we really need an initial sample before 
        // anything starts to get an accurate reading
        float max_vram = 0;

        if (samples_.size() > 1) {
            for (int i = 0; i < samples_.size(); i++) {
                total_milliwatts += samples_[i].power;
                if (samples_[i].vram > max_vram) {
                    max_vram = samples_[i].vram;
                }
            }
        }

        double avg_milliwatts = total_milliwatts / samples_.size();
        double avg_watts = avg_milliwatts / 1e3;
        double avg_watts_energy = energy_consumed / sampling_time;

        if (FLAG_verbose) {
            printf("Average power consumption from samples: %.2f mW, %.2f W\n", avg_milliwatts, avg_milliwatts / 1000);
            printf("Total energy consumed: %.2f mJ, %.2fJ in %d ms\n", energy_consumed, energy_consumed / 1000,  sampling_time);
            printf("Average power from energy consumed: %.2f W \n", energy_consumed / sampling_time);
        }

        // TODO decide on default..?
        // pick the higher reading of the two
        result.power = (avg_watts > avg_watts_energy) ? avg_watts : avg_watts_energy;
        result.vram = max_vram;
    }

    return result;
}

// this will return the latest sample in mw
power_sample_t PowerSampler::getLatestSample() {
    pthread_mutex_lock(&samples_mutex_);

    if (samples_.empty()) {
        pthread_mutex_unlock(&samples_mutex_);
        return {0.0, 0.0f};
    }

    power_sample_t sample = samples_.back();
    pthread_mutex_unlock(&samples_mutex_);

    return sample;
}

void* PowerSampler::sampling_thread_func(void* arg) {
    PowerSampler* sampler = static_cast<PowerSampler*>(arg);

    while (sampler->is_sampling_) {
        usleep(sampler->sample_length_ms_ * 1000); // Convert ms to microseconds
        power_sample_t sample = sampler->sample();

        pthread_mutex_lock(&sampler->samples_mutex_);
        sampler->samples_.push_back(sample);
        pthread_mutex_unlock(&sampler->samples_mutex_);

    }

    return nullptr;
}

// NvidiaPowerSampler implementation

NvidiaPowerSampler::NvidiaPowerSampler(long sample_length_ms, unsigned int main_gpu)
    : PowerSampler(sample_length_ms) {
        bool ok = nvml_init();
        if (!ok) {
            throw std::runtime_error("Failed to initialize NVML");
        }

        ok = nvml_get_device(&device_, main_gpu);
        if (!ok) {
            throw std::runtime_error("Failed to get NVML device");
        }
    }

NvidiaPowerSampler::~NvidiaPowerSampler() {
    nvml_shutdown();
}

power_sample_t NvidiaPowerSampler::sample() {
    power_sample_t sample;
    unsigned int mw;

    // if (nvml_get_memory_usage(device_, &sample.vram)) { }
    if (!nvml_get_power_usage(device_, &mw)) {
        // TODO return a bool instead? error?
    }

    sample.power = (double)mw;

    return sample;
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

power_sample_t AMDPowerSampler::sample() {
    power_sample_t sample;

    double power;
    float vram;

    if (!rsmi_get_power(&power)) { }
    if (!rsmi_get_memory_usage(&vram)) { }

    sample.power = power;
    sample.vram = vram;

    return sample;
}

double AMDPowerSampler::getEnergyConsumed() {
    double uj;
    // if (!rsmi_get_energy_count(&uj)) {
    //     return 0.0;
    // }
    return uj;
}

// ApplePowerSampler implementation

ApplePowerSampler::ApplePowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {
        init_apple_mon();
        power_channel_ = am_get_power_channels();
        sub_ = am_get_subscription(power_channel_);
        last_sample_time_ = timespec_tomillis(timespec_real());
        last_sample_mj_ = getEnergyConsumed();
        metal_backend_ = ggml_backend_metal_init();
    }

ApplePowerSampler::~ApplePowerSampler() {
    am_release(power_channel_);
    am_release(sub_);
}

power_sample_t ApplePowerSampler::sample() {
    float used;
    float total;
    ggml_backend_metal_get_device_memory_usage(metal_backend_, &used, &total);

    long long time = timespec_tomillis(timespec_real());
    double mj = getEnergyConsumed();

    double power = (mj - last_sample_mj_) / (time - last_sample_time_);
    // TODO we have to be careful with this as it is not protected by a mutex
    last_sample_time_ = time;
    last_sample_mj_ = mj;

    // convert to power in milliwatts
    power_sample_t sample = {power * 1e3, used};

    return sample;
}

// TODO this needs to be a void*?
double ApplePowerSampler::getEnergyConsumed() {
    double mj = am_sample_to_millijoules(am_sample_power(sub_, power_channel_));
    return mj;
}

// DummyPowerSampler implementation

// AMDPowerSampler2::AMDPowerSampler2(long sample_length_ms)
//     : PowerSampler(sample_length_ms) {
//         amdgpu_init();
//     }

// power_sample_t AMDPowerSampler2::sample() {
//     return {0.0, 0.0f};
// }

// double AMDPowerSampler2::getEnergyConsumed() {
//     return 0.0;
// }

// DummyPowerSampler implementation

DummyPowerSampler::DummyPowerSampler(long sample_length_ms)
    : PowerSampler(sample_length_ms) {}

power_sample_t DummyPowerSampler::sample() {
    return {0.0, 0.0f};
}

double DummyPowerSampler::getEnergyConsumed() {
    return 0.0;
}

// Update getPowerSampler function to return DummyPowerSampler if no other sampler is found
// TODO handle which GPU to use (for NVIDIA)
PowerSampler* getPowerSampler(long sample_length_ms, unsigned int main_gpu) {
    if (IsXnu()) {
        return new ApplePowerSampler(sample_length_ms);
    } else if (llamafile_has_gpu() && FLAG_gpu != LLAMAFILE_GPU_DISABLE) {
        if (llamafile_has_amd_gpu()) {
            // TODO change this to AMD power sampler when it works.
            // return new AMDPowerSampler(sample_length_ms);
            // return new AMDPowerSampler2(sample_length_ms);
            return new DummyPowerSampler(sample_length_ms);
        } else if (llamafile_has_cuda()) {
            try {
                return new NvidiaPowerSampler(sample_length_ms, main_gpu);
            } catch (const std::exception& e) {
                // Log the error if needed
                printf("NVIDIA initialization failed: %s\n", e.what());
                return new DummyPowerSampler(sample_length_ms);
            }
        }
    }

    return new DummyPowerSampler(sample_length_ms);
}