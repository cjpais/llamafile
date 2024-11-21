// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <cosmo.h>
#include <dlfcn.h>
#include <libgen.h>
#include <pthread.h>
#include <mutex> // TODO replace with pthreads
#include <sys/stat.h>
#include <sys/auxv.h>
#include <libc/intrin/x86.h>
#include "llama.cpp/cores.h"
#include <libc/sysv/consts/hwcap.h>

#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-metal.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/string.h"
#include "llama.cpp/common.h"
#include "llama.cpp/ggml-cuda.h"
#include "llamafile/llamafile.h"

#include "powersampler.h"

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

template<class T>
static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template<class T>
static std::vector<T> split(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template<typename T, typename F>
static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template<typename T>
static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T)v.size();
}

template<typename T>
static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev = std::sqrt(sq_sum / (T)(v.size() - 1) - mean * mean * (T)v.size() / (T)(v.size() - 1));
    return stdev;
}

#ifdef __x86_64__
static void cpuid(unsigned leaf, unsigned subleaf, unsigned *info) {
    asm("movq\t%%rbx,%%rsi\n\t"
        "cpuid\n\t"
        "xchgq\t%%rbx,%%rsi"
        : "=a"(info[0]), "=S"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "0"(leaf), "2"(subleaf));
}
#endif // __x86_64__

static std::string get_cpu_info() { // [jart]
    std::string id;

#ifdef __x86_64__
    union { // [jart]
        char str[64];
        unsigned reg[16];
    } u = {0};
    cpuid(0x80000002, 0, u.reg + 0*4);
    cpuid(0x80000003, 0, u.reg + 1*4);
    cpuid(0x80000004, 0, u.reg + 2*4);
    int len = strlen(u.str);
    while (len > 0 && u.str[len - 1] == ' ')
        u.str[--len] = 0;
    id = u.str;
#else
    if (IsLinux()) {
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (!strncmp(buf, "model name", 10) ||
                    startswith(buf, "Model\t\t:")) { // e.g. raspi
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        id = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
    }
    if (IsXnu()) {
        // TODO we can also do something similar to https://github.com/vladkens/macmon/blob/main/src/sources.rs#L424
        char cpu_name[128] = {0};
        size_t size = sizeof(cpu_name);
        if (sysctlbyname("machdep.cpu.brand_string", cpu_name, &size, NULL, 0) != -1) {
            id = cpu_name;
        }

        // TODO IF ARCH IS ARM
        // Get number of performance cores on macos
        int num_perf0_cpu;
        size = sizeof(num_perf0_cpu);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &num_perf0_cpu, &size, NULL, 0) != -1) {
            id += " ";
            id += std::to_string(num_perf0_cpu);
            id += "P";
        }

        // Get number of efficiency cores on macos
        int num_perf1_cpu;
        size = sizeof(num_perf1_cpu);
        if (sysctlbyname("hw.perflevel1.logicalcpu", &num_perf1_cpu, &size, NULL, 0) != -1) {
            id += "+";
            id += std::to_string(num_perf1_cpu);
            id += "E";
        }

    }
#endif
    id = replace_all(id, " 96-Cores", "");
    id = replace_all(id, "(TM)", "");
    id = replace_all(id, "(R)", "");

    std::string march;
#ifdef __x86_64__
    if (__cpu_march(__cpu_model.__cpu_subtype))
        march = __cpu_march(__cpu_model.__cpu_subtype);
#else
    long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ASIMDHP)
        march += "+fp16";
    if (hwcap & HWCAP_ASIMDDP)
        march += "+dotprod";
#endif

    if (!march.empty()) {
        bool empty = id.empty();
        if (!empty)
            id += " (";
        id += march;
        if (!empty)
            id += ")";
    }

    return id;
}

#define MAX_STRING_LENGTH 256

typedef struct {
    char llamafile_version[MAX_STRING_LENGTH];
    char llama_commit[MAX_STRING_LENGTH];
} RuntimeInfo;

typedef struct {
    char kernel_type[MAX_STRING_LENGTH];
    char kernel_release[MAX_STRING_LENGTH];
    char version[MAX_STRING_LENGTH];
    char system_architecture[MAX_STRING_LENGTH];
    char cpu[MAX_STRING_LENGTH];
    double ram_gb;
} SystemInfo;

typedef struct {
    char name[MAX_STRING_LENGTH];
    char manufacturer[MAX_STRING_LENGTH];
    double total_memory_gb;
    int core_count;
    double capability;
} GPUInfo;

static void get_runtime_info(RuntimeInfo* info) {
    if (info == NULL) return;

    strncpy(info->llamafile_version, LLAMAFILE_VERSION_STRING, MAX_STRING_LENGTH - 1);
    strncpy(info->llama_commit, LLAMA_COMMIT, MAX_STRING_LENGTH - 1);

    // printf("\033[0;35m\n===== llamafile bench runtime information =====\n\n");
    // printf("%-20s \033[1m%s\033[22m\n", "llamafile version:", info->llamafile_version);
    // printf("%-20s %s\n", "llama.cpp commit:", info->llama_commit);
    // printf("\n===============================================\n\n\033[0m");
}

static void get_sys_info(SystemInfo* info) {
    if (info == NULL) return;

    struct utsname names;
    if (uname(&names)) {
        return;
    }

    struct sysinfo si;
    if (sysinfo(&si)) {
        return;
    }

    strncpy(info->kernel_type, names.sysname, MAX_STRING_LENGTH - 1);
    strncpy(info->kernel_release, names.release, MAX_STRING_LENGTH - 1);
    // TODO on darwin we might want to get from systemprofiler SPSoftwareDataType os_version
    strncpy(info->version, names.version, MAX_STRING_LENGTH - 1);
    strncpy(info->system_architecture, names.machine, MAX_STRING_LENGTH - 1);

    std::string cpu_info = get_cpu_info();
    strncpy(info->cpu, cpu_info.c_str(), MAX_STRING_LENGTH - 1);

    info->ram_gb = si.totalram * si.mem_unit / 1073741824.0;

    // printf("===== system information =====\n\n");
    // printf("%-20s %s\n", "Kernel Type:", info->kernel_type);
    // printf("%-20s %s\n", "Kernel Release:", info->kernel_release);
    // printf("%-20s %s\n", "Version:", info->version);
    // printf("%-20s %s\n", "System Architecture:", info->system_architecture);
    // printf("%-20s %s\n", "CPU:", info->cpu);
    // printf("%-20s %.2f GiB\n", "RAM:", info->ram_gb);
    // printf("\n===============================\n\n");
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// TODO rename to accelerator. can be cpu or gpu dependent on enablement of ngl
static void get_gpu_info(GPUInfo* info) {
    if (info == NULL) return;

    // TODO: Check if GPU is enabled. otherwise get cpu info.

    if (llamafile_has_cuda()) {
        int count = ggml_backend_cuda_get_device_count();
        if (count > 0) {
            struct ggml_cuda_device_properties props;
            ggml_backend_cuda_get_device_properties(0, &props);

            strncpy(info->name, props.name, MAX_STRING_LENGTH - 1);
            info->total_memory_gb = props.totalGlobalMem / 1073741824.0;
            info->core_count = props.multiProcessorCount;
            info->capability = atof(props.compute);
            strncpy(info->manufacturer, llamafile_has_amd_gpu() ? "AMD" : "NVIDIA", MAX_STRING_LENGTH - 1);

            // printf("\033[0;32m===== GPU information =====\n\n");
            // printf("%-26s %s\n", "GPU Name:", info->name);
            // printf("%-26s %.2f GiB\n", "VRAM:", info->total_memory_gb);
            // printf("%-26s %d\n", "Streaming Multiprocessors:", info->core_count);
            // printf("%-26s %.1f\n", "CUDA Capability:", info->capability);
            // printf("\n============================\n\n\033[0m");
        }
    }

    if (llamafile_has_metal()) {
        // TODO there is probably a cleaner way of doing this. we should only need to init once.
        // this is probably the same issue why the other thing is init multiple time too
        struct ggml_metal_device_properties props;

        std::string command = "system_profiler SPDisplaysDataType | grep \"Total Number of Cores:\" | awk '{print $5}'";
        std::string num_cores = exec(command.c_str());
        props.core_count = std::stoi(num_cores);

        // Remove any trailing newline
        if (!num_cores.empty() && num_cores[num_cores.length()-1] == '\n') {
            num_cores.erase(num_cores.length()-1);
        }


        ggml_backend_t result = ggml_backend_metal_init();

        ggml_backend_metal_get_device_properties(result, &props);

        strncpy(info->name, props.name, MAX_STRING_LENGTH - 1);
        info->total_memory_gb = props.memory;
        info->core_count = props.core_count;
        info->capability = props.metal_version;
        strncpy(info->manufacturer, "APPLE", MAX_STRING_LENGTH - 1);

        // printf("\033[0;32m===== GPU information =====\n\n");
        // printf("%-26s %s\n", "GPU Name:", props.name);
        // printf("%-26s %.2f GiB\n", "VRAM:", props.memory);
        // printf("%-26s %d\n", "Core Count:", props.core_count);
        // printf("%-26s %d\n", "Metal Version:", props.metal_version);
        // printf("%-26s %d\n", "GPU Family:", props.gpu_family);
        // printf("%-26s %d\n", "Common GPU Family:", props.gpu_family_common);
        // printf("\n============================\n\n\033[0m");
    }
    // TODO: other backends (metal)
    // macos: get gpu cores `system_profiler -detailLevel basic SPDisplaysDataType | grep 'Total Number of Cores'`
}

// command line params
enum output_formats {CSV, JSON, MARKDOWN, SQL};

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CSV:      return "csv";
        case JSON:     return "json";
        case MARKDOWN: return "md";
        case SQL:      return "sql";
        default: GGML_ASSERT(!"invalid output format");
    }
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:  return "none";
        case LLAMA_SPLIT_MODE_LAYER: return "layer";
        case LLAMA_SPLIT_MODE_ROW:   return "row";
        default: GGML_ASSERT(!"invalid split mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

struct cmd_params {
    std::vector<std::string> model;
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int> n_batch;
    std::vector<int> n_ubatch;
    std::vector<ggml_type> type_k;
    std::vector<ggml_type> type_v;
    std::vector<int> n_threads;
    std::vector<int> n_gpu_layers;
    std::vector<llama_split_mode> split_mode;
    std::vector<int> main_gpu;
    std::vector<bool> no_kv_offload;
    std::vector<bool> flash_attn;
    std::vector<std::vector<float>> tensor_split;
    std::vector<bool> use_mmap;
    std::vector<bool> embeddings;
    ggml_numa_strategy numa;
    int reps;
    bool verbose;
    output_formats output_format;
};

static const cmd_params cmd_params_defaults = {
    /* model         */ {}, // [jart] no default guessing
    /* n_prompt      */ {512},
    /* n_gen         */ {16},
    /* n_pg          */ {},
    /* n_batch       */ {2048},
    /* n_ubatch      */ {512},
    /* type_k        */ {X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16},
    /* type_v        */ {X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16},
    /* n_threads     */ {cpu_get_num_math()},
    /* n_gpu_layers  */ {0},
    /* split_mode    */ {LLAMA_SPLIT_MODE_LAYER},
    /* main_gpu      */ {0},
    /* no_kv_offload */ {false},
    /* flash_attn    */ {false},
    /* tensor_split  */ {std::vector<float>(llama_max_devices(), 0.0f)},
    /* use_mmap      */ {true},
    /* embeddings    */ {false},
    /* numa          */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps          */ 4,
    /* verbose       */ false,
    /* output_format */ MARKDOWN
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>              (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    printf("  -p, --n-prompt <n>                  (default: %s)\n", join(cmd_params_defaults.n_prompt, ",").c_str());
    printf("  -n, --n-gen <n>                     (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    printf("  -pg <pp,tg>                         (default: %s)\n", join(transform_to_str(cmd_params_defaults.n_pg, pair_str), ",").c_str());
    printf("  -b, --batch-size <n>                (default: %s)\n", join(cmd_params_defaults.n_batch, ",").c_str());
    printf("  -ub, --ubatch-size <n>              (default: %s)\n", join(cmd_params_defaults.n_ubatch, ",").c_str());
    printf("  -ctk, --cache-type-k <t>            (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_k, ggml_type_name), ",").c_str());
    printf("  -ctv, --cache-type-v <t>            (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_v, ggml_type_name), ",").c_str());
    printf("  -t, --threads <n>                   (default: %s)\n", join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -ngl, --n-gpu-layers <n>            (default: %s)\n", join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    printf("  -sm, --split-mode <none|layer|row>  (default: %s)\n", join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
    printf("  -mg, --main-gpu <i>                 (default: %s)\n", join(cmd_params_defaults.main_gpu, ",").c_str());
    printf("  -nkvo, --no-kv-offload <0|1>        (default: %s)\n", join(cmd_params_defaults.no_kv_offload, ",").c_str());
    printf("  -fa, --flash-attn <0|1>             (default: %s)\n", join(cmd_params_defaults.flash_attn, ",").c_str());
    printf("  -mmp, --mmap <0|1>                  (default: %s)\n", join(cmd_params_defaults.use_mmap, ",").c_str());
    printf("  --numa <distribute|isolate|numactl> (default: disabled)\n");
    printf("  -embd, --embeddings <0|1>           (default: %s)\n", join(cmd_params_defaults.embeddings, ",").c_str());
    printf("  -ts, --tensor-split <ts0/ts1/..>    (default: 0)\n");
    printf("  -r, --repetitions <n>               (default: %d)\n", cmd_params_defaults.reps);
    printf("  -o, --output <csv|json|md|sql>      (default: %s)\n", output_format_str(cmd_params_defaults.output_format));
    printf("  -v, --verbose                       (default: %s)\n", cmd_params_defaults.verbose ? "1" : "0");
    printf("\n");
    printf("Multiple values can be given for each parameter by separating them with ',' or by specifying the parameter multiple times.\n");
}

static ggml_type ggml_type_from_name(const std::string & s) {
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }

    return GGML_TYPE_COUNT;
}


static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params;
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";
    const char split_delim = ',';

    params.verbose = cmd_params_defaults.verbose;
    params.output_format = cmd_params_defaults.output_format;
    params.reps = cmd_params_defaults.reps;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace (arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], split_delim);
            params.model.insert(params.model.end(), p.begin(), p.end());
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
        } else if (arg == "-pg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], ',');
            if (p.size() != 2) {
                invalid_param = true;
                break;
            }
            params.n_pg.push_back({std::stoi(p[0]), std::stoi(p[1])});
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_ubatch.insert(params.n_ubatch.end(), p.begin(), p.end());
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], split_delim);
            std::vector<ggml_type> types;
            for (const auto & t : p) {
                ggml_type gt = ggml_type_from_name(t);
                if (gt == GGML_TYPE_COUNT) {
                    invalid_param = true;
                    break;
                }
                types.push_back(gt);
            }
            params.type_k.insert(params.type_k.end(), types.begin(), types.end());
        } else if (arg == "-ctv" || arg == "--cache-type-v") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], split_delim);
            std::vector<ggml_type> types;
            for (const auto & t : p) {
                ggml_type gt = ggml_type_from_name(t);
                if (gt == GGML_TYPE_COUNT) {
                    invalid_param = true;
                    break;
                }
                types.push_back(gt);
            }
            params.type_v.insert(params.type_v.end(), types.begin(), types.end());
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            FLAG_gpu = LLAMAFILE_GPU_AUTO;
            auto p = split<int>(argv[i], split_delim);
            params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
        } else if (arg == "-sm" || arg == "--split-mode") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], split_delim);
            std::vector<llama_split_mode> modes;
            for (const auto & m : p) {
                llama_split_mode mode;
                if (m == "none") {
                    mode = LLAMA_SPLIT_MODE_NONE;
                } else if (m == "layer") {
                    mode = LLAMA_SPLIT_MODE_LAYER;
                } else if (m == "row") {
                    mode = LLAMA_SPLIT_MODE_ROW;
                } else {
                    invalid_param = true;
                    break;
                }
                modes.push_back(mode);
            }
            params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
        } else if (arg == "-mg" || arg == "--main-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.main_gpu = split<int>(argv[i], split_delim);
        } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.no_kv_offload.insert(params.no_kv_offload.end(), p.begin(), p.end());
        } else if (arg == "--numa") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            } else {
                std::string value(argv[i]);
                /**/ if (value == "distribute" || value == "" ) { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
                else if (value == "isolate")                    { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
                else if (value == "numactl")                    { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
                else { invalid_param = true; break; }
            }
        } else if (arg == "-fa" || arg == "--flash-attn") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.flash_attn.insert(params.flash_attn.end(), p.begin(), p.end());
        } else if (arg == "--recompile") {
            FLAG_recompile = true;            
        } else if (arg == "-mmp" || arg == "--mmap") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.use_mmap.insert(params.use_mmap.end(), p.begin(), p.end());
        } else if (arg == "-embd" || arg == "--embeddings") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.embeddings.insert(params.embeddings.end(), p.begin(), p.end());
        } else if (arg == "-ts" || arg == "--tensor-split") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            for (auto ts : split<std::string>(argv[i], split_delim)) {
                // split string by ; and /
                const std::regex regex{R"([;/]+)"};
                std::sregex_token_iterator it{ts.begin(), ts.end(), regex, -1};
                std::vector<std::string> split_arg{it, {}};
                GGML_ASSERT(split_arg.size() <= llama_max_devices());

                std::vector<float> tensor_split(llama_max_devices());
                for (size_t i = 0; i < llama_max_devices(); ++i) {
                    if (i < split_arg.size()) {
                        tensor_split[i] = std::stof(split_arg[i]);
                    } else {
                        tensor_split[i] = 0.0f;
                    }
                }
                params.tensor_split.push_back(tensor_split);
            }
        } else if (arg == "-r" || arg == "--repetitions") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.reps = std::stoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (argv[i] == std::string("csv")) {
                params.output_format = CSV;
            } else if (argv[i] == std::string("json")) {
                params.output_format = JSON;
            } else if (argv[i] == std::string("md")) {
                params.output_format = MARKDOWN;
            } else if (argv[i] == std::string("sql")) {
                params.output_format = SQL;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg[0] == '-') {
            invalid_param = true;
            break;
        } else {
            // [jart] let me glob without needing -m flag
            auto p = split<std::string>(argv[i], split_delim);
            params.model.insert(params.model.end(), p.begin(), p.end());
        }
    }
    if (invalid_param) {
        fprintf(stderr, "%s: invalid parameter for argument: %s\n", program_invocation_name, arg.c_str());
        exit(1);
    }
    if (params.model.empty()) {
        fprintf(stderr, "%s: missing operand\n", program_invocation_name, arg.c_str());
        exit(1);
    }

    // [jart] sort larger models first
    std::sort(params.model.begin(), params.model.end(), [](const std::string& a, const std::string& b) {
        struct stat statA, statB;
        printf("a: %s\n", a.c_str());
        if (stat(a.c_str(), &statA)) {
            perror(a.c_str());
            exit(1);
        }
        if (stat(b.c_str(), &statB)) {
            perror(b.c_str());
            exit(1);
        }
        return statA.st_size > statB.st_size;
    });

    // set defaults
    if (params.model.empty())        { params.model = cmd_params_defaults.model; }
    if (params.n_prompt.empty())     { params.n_prompt = cmd_params_defaults.n_prompt; }
    if (params.n_gen.empty())        { params.n_gen = cmd_params_defaults.n_gen; }
    if (params.n_pg.empty())         { params.n_pg = cmd_params_defaults.n_pg; }
    if (params.n_batch.empty())      { params.n_batch = cmd_params_defaults.n_batch; }
    if (params.n_ubatch.empty())     { params.n_ubatch = cmd_params_defaults.n_ubatch; }
    if (params.type_k.empty())       { params.type_k = cmd_params_defaults.type_k; }
    if (params.type_v.empty())       { params.type_v = cmd_params_defaults.type_v; }
    if (params.n_gpu_layers.empty()) { params.n_gpu_layers = cmd_params_defaults.n_gpu_layers; }
    if (params.split_mode.empty())   { params.split_mode = cmd_params_defaults.split_mode; }
    if (params.main_gpu.empty())     { params.main_gpu = cmd_params_defaults.main_gpu; }
    if (params.no_kv_offload.empty()){ params.no_kv_offload = cmd_params_defaults.no_kv_offload; }
    if (params.flash_attn.empty())   { params.flash_attn = cmd_params_defaults.flash_attn; }
    if (params.tensor_split.empty()) { params.tensor_split = cmd_params_defaults.tensor_split; }
    if (params.use_mmap.empty())     { params.use_mmap = cmd_params_defaults.use_mmap; }
    if (params.embeddings.empty())   { params.embeddings = cmd_params_defaults.embeddings; }
    if (params.n_threads.empty())    { params.n_threads = cmd_params_defaults.n_threads; }

    return params;
}

struct cmd_params_instance {
    std::string model;
    int n_prompt;
    int n_gen;
    int n_batch;
    int n_ubatch;
    ggml_type type_k;
    ggml_type type_v;
    int n_threads;
    int n_gpu_layers;
    llama_split_mode split_mode;
    int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    std::vector<float> tensor_split;
    bool use_mmap;
    bool embeddings;

    llama_model_params to_llama_mparams() const {
        llama_model_params mparams = llama_model_default_params();

        mparams.n_gpu_layers = n_gpu_layers;
        mparams.split_mode = split_mode;
        mparams.main_gpu = main_gpu;
        mparams.tensor_split = tensor_split.data();
        mparams.use_mmap = use_mmap;

        return mparams;
    }

    bool equal_mparams(const cmd_params_instance & other) const {
        return model == other.model &&
               n_gpu_layers == other.n_gpu_layers &&
               split_mode == other.split_mode &&
               main_gpu == other.main_gpu &&
               use_mmap == other.use_mmap &&
               tensor_split == other.tensor_split;
    }

    llama_context_params to_llama_cparams() const {
        llama_context_params cparams = llama_context_default_params();

        cparams.n_ctx = n_prompt + n_gen;
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_ubatch;
        cparams.type_k = type_k;
        cparams.type_v = type_v;
        cparams.offload_kqv = !no_kv_offload;
        cparams.flash_attn = flash_attn;
        cparams.embeddings = embeddings;

        return cparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    for (const auto & m : params.model)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & sm : params.split_mode)
    for (const auto & mg : params.main_gpu)
    for (const auto & ts : params.tensor_split)
    for (const auto & mmp : params.use_mmap)
    for (const auto & embd : params.embeddings)
    for (const auto & nb : params.n_batch)
    for (const auto & nub : params.n_ubatch)
    for (const auto & tk : params.type_k)
    for (const auto & tv : params.type_v)
    for (const auto & nkvo : params.no_kv_offload)
    for (const auto & fa : params.flash_attn)
    for (const auto & nt : params.n_threads) {
        for (const auto & n_prompt : params.n_prompt) {
            if (n_prompt == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ n_prompt,
                /* .n_gen        = */ 0,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
            };
            instances.push_back(instance);
        }

        for (const auto & n_gen : params.n_gen) {
            if (n_gen == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ 0,
                /* .n_gen        = */ n_gen,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
            };
            instances.push_back(instance);
        }

        for (const auto & n_pg : params.n_pg) {
            if (n_pg.first == 0 && n_pg.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ n_pg.first,
                /* .n_gen        = */ n_pg.second,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
            };
            instances.push_back(instance);
        }
    }

    return instances;
}

struct time_interval {
    uint64_t start;
    uint64_t end;
};

struct test_config {
    int n_prompt;
    int n_gen;
};

enum token_metric {
    TOTAL_TPS,
    PROMPT_TPS,
    GEN_TPS
};

struct test {
    static const std::string build_commit;
    static const int build_number;
    static const bool cuda;
    static const bool opencl;
    static const bool vulkan;
    static const bool kompute;
    static const bool metal;
    static const bool sycl;
    static const bool gpu_blas;
    static const bool blas;
    static const std::string cpu_info;
    static const std::string gpu_info;
    std::string name;
    std::string model_name;
    std::string model_filename;
    std::string model_type;
    std::string model_quant_str;
    std::string model_params_str;
    uint64_t model_size;
    uint64_t model_n_params;
    int n_batch;
    int n_ubatch;
    int n_threads;
    ggml_type type_k;
    ggml_type type_v;
    int n_gpu_layers;
    llama_split_mode split_mode;
    int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    std::vector<float> tensor_split;
    bool use_mmap;
    bool embeddings;
    int n_prompt;
    int n_gen;
    int reps;
    mutable std::mutex t_gen_mutex;
    std::atomic_bool test_completed{false};
    volatile int curr_run;
    volatile int t_gen; // this is the total number of tokens generated
    volatile int t_processed; // this is the total number of tokens processed
    power_sample_t monitor_result;
    std::string test_time;
    std::vector<time_interval> test_intervals;
    std::vector<time_interval> prompt_intervals;
    std::vector<time_interval> gen_intervals;
    std::vector<uint64_t> time_to_first_token;
    llama_context * ctx;
    PowerSampler * pwr_sampler;

    // TODO add sampler
    test(const cmd_params_instance & inst, const llama_model * lmodel, llama_context * context, int repetitions, PowerSampler * sampler) {
        model_filename = basename(strdup(inst.model.c_str()));  // [jart]
        char buf[128];
        llama_model_desc(lmodel, buf, sizeof(buf));
        model_type = buf;
        llama_model_meta_val_str(lmodel, "general.name", buf, sizeof(buf));
        model_name = buf;
        llama_model_quant_str(lmodel, buf, sizeof(buf));
        model_quant_str = buf;
        model_size = llama_model_size(lmodel);
        model_n_params = llama_model_n_params(lmodel);
        llama_model_meta_val_str(lmodel, "general.size_label", buf, sizeof(buf));
        model_params_str = buf;
        n_batch = inst.n_batch;
        n_ubatch = inst.n_ubatch;
        n_threads = inst.n_threads;
        type_k = inst.type_k;
        type_v = inst.type_v;
        n_gpu_layers = inst.n_gpu_layers;
        split_mode = inst.split_mode;
        main_gpu = inst.main_gpu;
        no_kv_offload = inst.no_kv_offload;
        flash_attn = inst.flash_attn;
        tensor_split = inst.tensor_split;
        use_mmap = inst.use_mmap;
        embeddings = inst.embeddings;
        n_prompt = inst.n_prompt;
        n_gen = inst.n_gen;
        reps = repetitions;
        test_completed = false;
        curr_run = 0;
        t_gen = 0;
        t_processed = 0;
        monitor_result = {0.0, 0.0f};
        pwr_sampler = sampler;

        if (n_prompt > 0 && n_gen == 0) {
            snprintf(buf, sizeof(buf), "pp%d", n_prompt);
        } else if (n_gen > 0 && n_prompt == 0) {
            snprintf(buf, sizeof(buf), "tg%d", n_gen);
        } else {
            snprintf(buf, sizeof(buf), "pp%d+tg%d", n_prompt, n_gen);
        }
        name = buf;

        // RFC 3339 date-time format
        time_t t = time(NULL);
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        ctx = context;
    }

    void run() {
        llama_kv_cache_clear(ctx);

        // warmup run
        // TODO: add warmup run, to lower stddev
        // if (n_prompt > 0) {
        //     test_prompt();
        // }
        // if (n_gen > 0) {
        //     test_gen();
        // }

        // run the test for however many repetitions specified
        pwr_sampler->start();
        for (int i = 0; i < reps; i++) {
            curr_run = i;
            llama_kv_cache_clear(ctx);
            llamafile_govern();

            time_interval interval;
            interval.start = get_time_ns();
            interval.end = 0;
            test_intervals.push_back(interval);

            if (n_prompt > 0) {
                test_prompt();
            }
            if (n_gen > 0) {
                test_gen();
            }

            test_intervals.back().end = get_time_ns();
        }
        monitor_result = pwr_sampler->stop();

        test_completed = true;
    }

    void test_prompt() {
        llama_set_n_threads(ctx, n_threads, n_threads);

        const llama_model * model = llama_get_model(ctx);
        const int32_t n_vocab = llama_n_vocab(model);

        std::vector<llama_token> tokens(n_batch);

        int n_processed = 0;

        time_interval interval;
        interval.start = get_time_ns();
        interval.end = 0;
        prompt_intervals.push_back(interval);

        while (n_processed < n_prompt) {
            int n_tokens = std::min(n_prompt - n_processed, n_batch);
            tokens[0] = n_processed == 0 && llama_add_bos_token(model) ? llama_token_bos(model) : std::rand() % n_vocab;
            for (int i = 1; i < n_tokens; i++) {
                tokens[i] = std::rand() % n_vocab;
            }
            llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens, n_processed, 0));
            n_processed += n_tokens;
            t_processed = n_processed;
        }

        llama_synchronize(ctx);

        prompt_intervals.back().end = get_time_ns();
    }

    void test_gen() {
        llama_set_n_threads(ctx, n_threads, n_threads);

        const llama_model * model = llama_get_model(ctx);
        const int32_t n_vocab = llama_n_vocab(model);

        llama_token token = llama_add_bos_token(model) ? llama_token_bos(model) : std::rand() % n_vocab;

        time_interval interval;
        interval.start = get_time_ns();
        interval.end = 0;
        gen_intervals.push_back(interval);

        for (int i = 0; i < n_gen; i++) {
            llama_decode(ctx, llama_batch_get_one(&token, 1, n_prompt + i, 0));
            llama_synchronize(ctx);
            if (i == 0) {
                uint64_t ttft = get_time_ns() - test_intervals.back().start;
                time_to_first_token.push_back(ttft);
            }
            token = std::rand() % n_vocab;
            t_gen = i + 1;
        }

        gen_intervals.back().end = get_time_ns();
    }

    std::vector<uint64_t> get_samples_ns(token_metric metric = TOTAL_TPS) const {
        const std::vector<time_interval>& intervals = 
            metric == PROMPT_TPS ? prompt_intervals :
            metric == GEN_TPS ? gen_intervals : 
            test_intervals;

        std::vector<uint64_t> samples_ns;
        for (const auto & interval : intervals) {
            if (interval.end == 0) {
                continue;
            }
            samples_ns.push_back(interval.end - interval.start);
        }
        return samples_ns;
    }

    uint64_t avg_ns(token_metric metric = TOTAL_TPS) const {
        std::vector<uint64_t> samples_ns = get_samples_ns(metric);
        return ::avg(samples_ns);
    }

    uint64_t stdev_ns(token_metric metric = TOTAL_TPS) const {
        std::vector<uint64_t> samples_ns = get_samples_ns(metric);
        return ::stdev(samples_ns);
    }

    float get_power() const {
        if (monitor_result.power > 0) {
            return monitor_result.power;
        } else {
            // the sample is in mw, convert to w
            return pwr_sampler->getLatestSample().power / 1000.0f;
        }
    }

    std::vector<double> get_ts(token_metric metric = TOTAL_TPS) const {
        int n_tokens = 0;
        switch (metric) {
            case TOTAL_TPS:
                n_tokens = n_prompt + n_gen;
                break;
            case PROMPT_TPS:
                n_tokens = n_prompt;
                break;
            case GEN_TPS:
                n_tokens = n_gen;
                break;
        }

        std::vector<double> ts;
        std::vector<uint64_t> samples_ns = get_samples_ns(metric);
        std::transform(samples_ns.begin(), samples_ns.end(), std::back_inserter(ts),
            [n_tokens](uint64_t t) { return 1e9 * n_tokens / t; });
        return ts;
    }

    double avg_ts(token_metric metric = TOTAL_TPS) const {
        return ::avg(get_ts(metric));
    }

    double stdev_ts(token_metric metric = TOTAL_TPS) const {
        return ::stdev(get_ts(metric));
    }

    double ttft() const {
        if (time_to_first_token.empty()) {
            return 0.0;
        }
        return avg(ttft_times);
    }

    static std::string get_backend() {
        if (cuda) {
            return GGML_CUDA_NAME;
        }
        if (opencl) {
            return "OpenCL";
        }
        if (vulkan) {
            return "Vulkan";
        }
        if (kompute) {
            return "Kompute";
        }
        if (metal) {
            return "Metal";
        }
        // if (sycl) {
        //     return GGML_SYCL_NAME;
        // }
        if (gpu_blas) {
            return "GPU BLAS";
        }
        if (blas) {
            return "BLAS";
        }

        return "CPU";
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "build_commit", "build_number",
            "model_name", "model_quant_str", "model_params_str",
            // "cuda", "opencl", "vulkan", "kompute", "metal", "sycl", "gpu_blas", "blas",
            // "cpu_info", "gpu_info",
            "model_filename", "model_type", "model_size", "model_n_params",
            // "n_batch", "n_ubatch",
            // "n_threads", "type_k", "type_v",
            // "n_gpu_layers", "split_mode",
            // "main_gpu", "no_kv_offload", "flash_attn",
            // "tensor_split", "use_mmap", "embeddings",
            "n_prompt", "n_gen", "test_time",
            "avg_time_ms", "stddev_time_ms",
            "prompt_tps", "prompt_tps_watt", "prompt_tps_stddev",
            "gen_tps", "gen_tps_watt", "gen_tps_stddev",
            "name", "power_watts", "vram_used_mb", "ttft_ms"
        };
        return fields;
    }

    enum field_type {STRING, BOOL, INT, FLOAT};

    static field_type get_field_type(const std::string & field) {
        if (field == "build_number" || field == "n_batch" || field == "n_ubatch" ||
            field == "n_threads" ||
            field == "model_size" || field == "model_n_params" ||
            field == "n_gpu_layers" || field == "main_gpu" ||
            field == "n_prompt" || field == "n_gen" ||
            field == "avg_time_ms" || field == "stddev_time_ms" || 
            field == "ttft_ms") {
            return INT;
        }
        if (field == "cuda" || field == "opencl"  || field == "vulkan" || field == "kompute" || field == "metal" ||
            field == "gpu_blas" || field == "blas" || field == "sycl" ||field == "f16_kv" || field == "no_kv_offload" ||
            field == "flash_attn" || field == "use_mmap" || field == "embeddings") {
            return BOOL;
        }
        if (field == "prompt_tps" || field == "prompt_tps_watt" || field == "prompt_tps_stddev" ||
            field == "gen_tps" || field == "gen_tps_watt" || field == "gen_tps_stddev" ||
            field == "power_watts" || field == "vram_used_mb") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        int max_nonzero = 0;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i <= max_nonzero; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
            tensor_split_str += buf;
            if (i < max_nonzero) {
                tensor_split_str += "/";
            }
        }
        double power = get_power();

        std::vector<std::string> values = {
            build_commit, std::to_string(build_number),
            model_name, model_quant_str, model_params_str,
            // std::to_string(cuda), std::to_string(opencl), std::to_string(vulkan), std::to_string(vulkan),
            // std::to_string(metal), std::to_string(sycl), std::to_string(gpu_blas), std::to_string(blas),
            // cpu_info, gpu_info,
            model_filename, model_type, std::to_string(model_size), std::to_string(model_n_params),
            // std::to_string(n_batch), std::to_string(n_ubatch),
            // std::to_string(n_threads), ggml_type_name(type_k), ggml_type_name(type_v),
            // std::to_string(n_gpu_layers), split_mode_str(split_mode),
            // std::to_string(main_gpu), std::to_string(no_kv_offload), std::to_string(flash_attn),
            // tensor_split_str, std::to_string(use_mmap), std::to_string(embeddings),
            std::to_string(n_prompt), std::to_string(n_gen), test_time,
            std::to_string(avg_ns() / 1e6), std::to_string(stdev_ns() / 1e6),
            std::to_string(avg_ts(PROMPT_TPS)), std::to_string(avg_ts(PROMPT_TPS) / power), std::to_string(stdev_ts(PROMPT_TPS)),
            std::to_string(avg_ts(GEN_TPS)), std::to_string(avg_ts(GEN_TPS) / power), std::to_string(stdev_ts(GEN_TPS)),
            name, std::to_string(power), std::to_string(monitor_result.vram), std::to_string(ttft() / 1e6)
        };
        return values;
    }

    std::map<std::string, std::string> get_map() const {
        std::map<std::string, std::string> map;
        auto fields = get_fields();
        auto values = get_values();
        std::transform(fields.begin(), fields.end(), values.begin(),
                std::inserter(map, map.end()), std::make_pair<const std::string &, const std::string &>);
        return map;
    }
};

const std::string test::build_commit = LLAMA_COMMIT;
const int         test::build_number = LLAMA_BUILD_NUMBER;
const bool        test::cuda         = false; // !!ggml_cpu_has_cuda(); // [jart]
const bool        test::opencl       = false; // !!ggml_cpu_has_clblast(); // [jart]
const bool        test::vulkan       = false; // !!ggml_cpu_has_vulkan(); // [jart]
const bool        test::kompute      = false; // !!ggml_cpu_has_kompute(); // [jart]
const bool        test::metal        = false; // !!ggml_cpu_has_metal(); // [jart]
const bool        test::gpu_blas     = false; // !!ggml_cpu_has_gpublas(); // [jart]
const bool        test::blas         = false; // !!ggml_cpu_has_blas(); // [jart]
const bool        test::sycl         = false; // !!ggml_cpu_has_sycl(); // [jart]
const std::string test::cpu_info     = get_cpu_info();
const std::string test::gpu_info     = ""; //get_gpu_info(); // [jart]

struct printer {
    virtual ~printer() {}

    FILE * fout;
    virtual void print_header(const cmd_params & params, GPUInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) { (void) params; }
    virtual void print_test(const test & t) = 0;
    virtual void print_footer() { }
};

struct csv_printer : public printer {
    static std::string escape_csv(const std::string & field) {
        std::string escaped = "\"";
        for (auto c : field) {
            if (c == '"') {
                escaped += "\"";
            }
            escaped += c;
        }
        escaped += "\"";
        return escaped;
    }

    void print_header(const cmd_params & params, GPUInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override  {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "%s\n", join(fields, ",").c_str());
        (void) params;
    }

    void print_test(const test & t) override {
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_csv);
        fprintf(fout, "%s\n", join(values, ",").c_str());
    }
};

struct json_printer : public printer {
    bool first = true;

    static std::string escape_json(const std::string & value) {
        std::string escaped;
        for (auto c : value) {
            if (c == '"') {
                escaped += "\\\"";
            } else if (c == '\\') {
                escaped += "\\\\";
            } else  if (c <= 0x1f) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                escaped += buf;
            } else {
                escaped += c;
            }
        }
        return escaped;
    }

    static std::string format_value(const std::string & field, const std::string & value) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "\"" + escape_json(value) + "\"";
            case test::BOOL:
                return value == "0" ? "false" : "true";
            default:
                return value;
        }
    }

void print_header(const cmd_params & params, GPUInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
    fprintf(fout, "{\n");
    
    // Print RuntimeInfo object
    fprintf(fout, "  \"runtime_info\": {\n");
    fprintf(fout, "    \"name\": \"%s\",\n", "llamafile");
    fprintf(fout, "    \"version\": \"%s\",\n", runtime_info.llamafile_version);
    fprintf(fout, "    \"commit\": \"%s\"\n", runtime_info.llama_commit);
    fprintf(fout, "  },\n");

    // Print SystemInfo object
    fprintf(fout, "  \"system_info\": {\n");
    fprintf(fout, "    \"cpu_name\": \"%s\",\n", sys_info.cpu);
    fprintf(fout, "    \"cpu_arch\": \"%s\",\n", sys_info.system_architecture);
    fprintf(fout, "    \"ram_gb\": %.2f,\n", sys_info.ram_gb);
    fprintf(fout, "    \"kernel_type\": \"%s\",\n", sys_info.kernel_type);
    fprintf(fout, "    \"kernel_release\": \"%s\",\n", sys_info.kernel_release);
    fprintf(fout, "    \"version\": \"%s\"\n", sys_info.version);
    fprintf(fout, "  },\n");

    // Print GPUInfo object
    fprintf(fout, "  \"accelerator_info\": {\n");
    fprintf(fout, "    \"name\": \"%s\",\n", gpu_info.name);
    fprintf(fout, "    \"manufacturer\": \"%s\",\n", gpu_info.manufacturer);
    fprintf(fout, "    \"memory_gb\": %.2f,\n", gpu_info.total_memory_gb);
    // TODO
    fprintf(fout, "    \"type\": \"%s\"\n", (FLAG_gpu >= 0 && llamafile_has_gpu()) ? "GPU" : "CPU");
    fprintf(fout, "  },\n");

    // Start the results array
    fprintf(fout, "  \"results\": [\n");
    
    (void) params;
}

    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "      \"%s\": %s,\n", fields.at(i).c_str(), format_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        if (first) {
            first = false;
        } else {
            fprintf(fout, ",\n");
        }
        fprintf(fout, "    {\n");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "      \"samples_ns\": [ %s ],\n", join(t.get_samples_ns(), ", ").c_str());
        fprintf(fout, "      \"samples_ts\": [ %s ]\n", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "    }");
        fflush(fout);
    }

    void print_footer() override {
        fprintf(fout, "\n  ]\n");
        fprintf(fout, "}");
    }
};

struct update_t_gen_column_args {
    const test & t;
    printer* p;
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 15; // [jart]
        }
        if (field == "cpu_info") {
            return test::cpu_info.size(); // [jart]
        }
        if (field == "model_filename") {
            return 40; // [jart]
        }
        if (field == "size" || field == "params") {
            return 10;
        }
        if (field == "n_gpu_layers") {
            return 3;
        }
        if (field == "test") {
            return 13;
        }
        if (field == "vram") {
            return 15;
        }

        int width = std::max((int)field.length(), 10);

        if (test::get_field_type(field) == test::STRING) {
            return -width;
        }
        return width;
    }

    static std::string get_field_display_name(const std::string & field) {
        if (field == "n_gpu_layers") {
            return "ngl";
        }
        if (field == "split_mode") {
            return "sm";
        }
        if (field == "n_threads") {
            return "threads";
        }
        if (field == "no_kv_offload") {
            return "nkvo";
        }
        if (field == "flash_attn") {
            return "fa";
        }
        if (field == "use_mmap") {
            return "mmap";
        }
        if (field == "embeddings") {
            return "embd";
        }
        if (field == "tensor_split") {
            return "ts";
        }
        return field;
    }

    void print_header(const cmd_params & params, GPUInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
        // select fields to print
        // fields.emplace_back("cpu_info"); // [jart]
        // fields.emplace_back("gpu_info"); // [jart]
        // fields.emplace_back("model_filename");
        // fields.emplace_back("model_type");
        // fields.emplace_back("model");
        fields.emplace_back("test");
        fields.emplace_back("run number");
        // fields.emplace_back("size"); // [jart]
        // fields.emplace_back("params"); // [jart]
        // fields.emplace_back("backend"); // [jart]
        fields.emplace_back("avg time"); // [jart]
        fields.emplace_back("power");
        fields.emplace_back("vram");
        bool is_cpu_backend = test::get_backend() == "CPU" || test::get_backend() == "BLAS";
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        // if (params.n_threads.size() > 1 || params.n_threads != cmd_params_defaults.n_threads || is_cpu_backend) {
        //     fields.emplace_back("n_threads");
        // }
        if (params.n_batch.size() > 1 || params.n_batch != cmd_params_defaults.n_batch) {
            fields.emplace_back("n_batch");
        }
        if (params.n_ubatch.size() > 1 || params.n_ubatch != cmd_params_defaults.n_ubatch) {
            fields.emplace_back("n_ubatch");
        }
        if (params.type_k.size() > 1 || params.type_k != cmd_params_defaults.type_k) {
            fields.emplace_back("type_k");
        }
        if (params.type_v.size() > 1 || params.type_v != cmd_params_defaults.type_v) {
            fields.emplace_back("type_v");
        }
        if (params.main_gpu.size() > 1 || params.main_gpu != cmd_params_defaults.main_gpu) {
            fields.emplace_back("main_gpu");
        }
        if (params.split_mode.size() > 1 || params.split_mode != cmd_params_defaults.split_mode) {
            fields.emplace_back("split_mode");
        }
        if (params.no_kv_offload.size() > 1 || params.no_kv_offload != cmd_params_defaults.no_kv_offload) {
            fields.emplace_back("no_kv_offload");
        }
        if (params.flash_attn.size() > 1 || params.flash_attn != cmd_params_defaults.flash_attn) {
            fields.emplace_back("flash_attn");
        }
        if (params.tensor_split.size() > 1 || params.tensor_split != cmd_params_defaults.tensor_split) {
            fields.emplace_back("tensor_split");
        }
        if (params.use_mmap.size() > 1 || params.use_mmap != cmd_params_defaults.use_mmap) {
            fields.emplace_back("use_mmap");
        }
        if (params.embeddings.size() > 1 || params.embeddings != cmd_params_defaults.embeddings) {
            fields.emplace_back("embeddings");
        }
        fields.emplace_back("tokens processed");
        fields.emplace_back("pp t/s");
        fields.emplace_back("tg t/s");
        fields.emplace_back("pp t/s/watt");
        fields.emplace_back("tg t/s/watt");
        fields.emplace_back("ttft");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        float power = t.get_power();

        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char buf[128];
            if (field == "model") {
              value = t.model_type;
            } else if (field == "size") {
                if (t.model_size < 1024*1024*1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.model_size / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            } else if (field == "params") {
                snprintf(buf, sizeof(buf), "%d", t.model_n_params);
                // if (t.model_n_params < 1000*1000*1000) {
                //     snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
                // } else {
                //     snprintf(buf, sizeof(buf), "%.2f B", t.model_n_params / 1e9);
                // }
                value = buf;
            } else if (field == "backend") {
                value = test::get_backend();
            } else if (field == "run number") {
                snprintf(buf, sizeof(buf), "%d/%d", t.curr_run + 1, t.reps);
                value = buf;
            } else if (field == "test") {
                value = t.name;
            } else if (field == "pp t/s") {
                snprintf(buf, sizeof(buf), "%.2f", t.avg_ts(PROMPT_TPS));

                value = buf;
            } else if (field == "tg t/s") {
                snprintf(buf, sizeof(buf), "%.2f", t.avg_ts(GEN_TPS));

                value = buf;
            } else if (field == "tokens processed") {
                int num_generated = t.t_gen + (t.curr_run * t.n_gen);
                int num_processed = t.t_processed + (t.curr_run * t.n_prompt);

                snprintf(buf, sizeof(buf), "%d / %d", num_generated + num_processed,  (t.n_gen * t.reps) + (t.n_prompt * t.reps));

                value = buf;
            } else if (field == "pp t/s/watt") {
                snprintf(buf, sizeof(buf), "%.4f", t.avg_ts(PROMPT_TPS) / power);

                value = buf;
            } else if (field == "tg t/s/watt") {
                snprintf(buf, sizeof(buf), "%.4f", t.avg_ts(GEN_TPS) / power);

                value = buf;
            } else if (field == "ttft") {
                snprintf(buf, sizeof(buf), "%.2f ms", t.ttft() / 1e6);

                value = buf;
            } else if (field == "power") {
                if (t.monitor_result.power > 0) {
                    snprintf(buf, sizeof(buf), "%.2f W", t.monitor_result.power);
                    value = buf;
                } else {
                    // read instant power
                    power_sample_t sample = t.pwr_sampler->getLatestSample();
                    snprintf(buf, sizeof(buf), "%.2f W", sample.power / 1e3);
                }

                value = buf;
            } else if (field == "vram") {
                if (t.monitor_result.vram > 0) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.monitor_result.vram);
                    value = buf;
                } else {
                    // read instant vram
                    power_sample_t sample = t.pwr_sampler->getLatestSample();
                    snprintf(buf, sizeof(buf), "%.2f MiB", sample.vram);
                    value = buf;
                }
            } else if (field == "avg time") {
                float avg_ms = t.avg_ns() / 1e6;

                if (avg_ms < 1000) {
                    snprintf(buf, sizeof(buf), "%.2f ms", avg_ms);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f s", avg_ms / 1e3);
                }

                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = replace_all(replace_all(vmap.at(field), ".gguf", ""), ".llamafile", ""); // [jart]
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            // if (field == "t/s") { // [jart]
            //     // HACK: the utf-8 character is 2 bytes
            //     width += 1;
            // }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

    void print_footer() override {
        // fprintf(fout, "\nbuild: %s (%d)\n", test::build_commit.c_str(), test::build_number); // [jart]
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "TEXT";
            case test::BOOL:
            case test::INT:
                return "INTEGER";
            case test::FLOAT:
                return "REAL";
            default:
                assert(false);
                exit(1);
        }
    }

    void print_header(const cmd_params & params, GPUInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS test (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields.at(i).c_str(), get_sql_field_type(fields.at(i)).c_str(),  i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n");
        fprintf(fout, "\n");
        (void) params;
    }

    void print_test(const test & t) override {
        fprintf(fout, "INSERT INTO test (%s) ", join(test::get_fields(), ", ").c_str());
        fprintf(fout, "VALUES (");
        std::vector<std::string> values = t.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values.at(i).c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

void* print_num_generated_periodically(void* args) {
    test * t = (test *) args;
    while (1) { // Loop indefinitely until the thread is cancelled from outside
        printf("num generated: %d\n", t->t_gen);
        fflush(stdout); // Ensure the output is printed immediately
        usleep(100 * 1000); // Sleep for 100 milliseconds
    }
    return NULL; // This line is technically unreachable in this example
}

void* update_t_gen_column(void* args) {
    update_t_gen_column_args* argv = static_cast<update_t_gen_column_args*>(args);
    const test & t = argv->t;
    printer* p = argv->p;

    // Check if printer is markdown printer
    markdown_printer* md_printer = dynamic_cast<markdown_printer*>(p);
    if (!md_printer) {
        // For non-markdown printers, wait until test is completed
        while (!t.test_completed) {
            usleep(100000);
        }
        p->print_test(t);
        return nullptr;
    }

    // For markdown printer, update continuously
    p->print_test(t);
    int last_t_gen = 0;
    while (!t.test_completed) {
        last_t_gen = t.t_gen;
        // Move up to the previous line and clear it
        printf("\033[A"); // Move up
        printf("\033[2K"); // Clear the entire line

        // Re-print the entire row with the updated t_gen value
        p->print_test(t);

        fflush(stdout);

        usleep(100000); // sleep for 100ms (100,000 microseconds)
    }
    // TODO this last print is probably uncessary.
    printf("\033[A"); // Move up
    printf("\033[2K"); // Clear the entire line
    p->print_test(t);
    return nullptr;
}

__attribute__((__constructor__(101))) static void init(void) {
    FLAG_gpu = LLAMAFILE_GPU_DISABLE; // [jart]
}

int main(int argc, char ** argv) {
    ShowCrashReports();

    std::vector<test_config> baseline_tests = {
        {1024, 16},     // 64:1 title generation
        {4096, 256},    // 16:1 content summarization
        {2048, 256},    // 8:1  lots of code to fix
        {2048, 768},    // 3:1  standard code chat
        {1024, 1024},   // 1:1  code back and forth
        {384, 1152},    // 1:3  code gen with back and forth
        {64, 1024},     // 1:16 code gen/ideation
        {16, 1536}      // 1:96 QA, Storytelling
    };

    // try to set locale for unicode characters in markdown
    setlocale(LC_CTYPE, "C.UTF-8");  // [jart]

    cmd_params params = parse_cmd_params(argc, argv);
    FLAGS_READY = true;

    GPUInfo gpu_info;
    if (FLAG_gpu != LLAMAFILE_GPU_DISABLE) {
        get_gpu_info(&gpu_info);
    }

    RuntimeInfo runtime_info;
    get_runtime_info(&runtime_info);

    SystemInfo sys_info;
    get_sys_info(&sys_info);


    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
        ggml_backend_metal_log_set_callback(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize printer
    std::unique_ptr<printer> p;
    switch (params.output_format) {
        case CSV:
            p.reset(new csv_printer());
            break;
        case JSON:
            p.reset(new json_printer());
            break;
        case MARKDOWN:
            p.reset(new markdown_printer());
            break;
        case SQL:
            p.reset(new sql_printer());
            break;
        default:
            assert(false);
            exit(1);
    }
    p->fout = stdout;
    p->print_header(params, gpu_info, runtime_info, sys_info);

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    llama_model * lmodel = nullptr;
    const cmd_params_instance * prev_inst = nullptr;

    PowerSampler * sampler = getPowerSampler(100);

    pthread_t print_thread;

    for (const auto & base_inst : params_instances) {
        int num_gen = base_inst.n_prompt > 0 ? 4096: 2048;
        for (int context_size = 16; context_size <= num_gen; context_size *= 2) {
            // TODO this is a total hack.
            cmd_params_instance inst = base_inst;
            if (base_inst.n_prompt > 0) {
                inst.n_prompt = context_size;
            } else {
                inst.n_gen = context_size;
            }

            // keep the same model between tests when possible
            if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
                if (lmodel) {
                    llama_free_model(lmodel);
                }

                lmodel = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
                if (lmodel == NULL) {
                    fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                    return 1;
                }

                // TODO build a json payload still..
                // printf("Model N Params: %d\n", llama_model_n_params(lmodel));

                prev_inst = &inst;
            }

            llama_context_params cparams = inst.to_llama_cparams();
            cparams.n_ctx = context_size;

            llama_context * ctx = llama_new_context_with_model(lmodel, cparams);
            if (ctx == NULL) {
                fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
                llama_free_model(lmodel);
                return 1;
            }

            test t(inst, lmodel, ctx, params.reps, sampler);

            update_t_gen_column_args argv = {t, p.get()};
            pthread_t update_thread;
            int rc = pthread_create(&update_thread, NULL, update_t_gen_column, &argv);
            if (rc) {
                std::cerr << "Error creating pthread: " << rc << std::endl;
                return EXIT_FAILURE;
            }
            t.run();

            pthread_join(update_thread, NULL);

            llama_print_timings(ctx);

            llama_free(ctx);
        }
    }

    for (const auto & test_cfg : baseline_tests) {
        cmd_params_instance inst = params_instances.front();
        inst.n_prompt = test_cfg.n_prompt;
        inst.n_gen = test_cfg.n_gen;

        if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
            if (lmodel) {
                llama_free_model(lmodel);
            }

            lmodel = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
            if (lmodel == NULL) {
                fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                return 1;
            }
            prev_inst = &inst;
        }

        llama_context_params cparams = inst.to_llama_cparams();
        cparams.n_ctx = test_cfg.n_prompt + test_cfg.n_gen;


        llama_context * ctx = llama_new_context_with_model(lmodel, cparams);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
            llama_free_model(lmodel);
            return 1;
        }

        test t(inst, lmodel, ctx, params.reps, sampler);

        update_t_gen_column_args argv = {t, p.get()};
        pthread_t update_thread;
        int rc = pthread_create(&update_thread, NULL, update_t_gen_column, &argv);
        if (rc) {
            std::cerr << "Error creating pthread: " << rc << std::endl;
            return EXIT_FAILURE;
        }
        t.run();

        pthread_join(update_thread, NULL);

        llama_print_timings(ctx);

        llama_free(ctx);
    }

    llama_free_model(lmodel);

    p->print_footer();

    llama_backend_free();

    return 0;
}
