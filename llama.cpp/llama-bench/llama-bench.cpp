// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include <algorithm>
#include <array>
#include <cassert>
// #include <chrono> [jart]
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
#include <iostream>
#include <string>
#include <vector>
#include <cosmo.h>
#include <dlfcn.h>
#include <libgen.h>
#include <pthread.h>
#include <mutex> // TODO replace with pthreads
#include <atomic> // TODO similar
#include <sys/stat.h>
#include <libc/intrin/x86.h>
#include <libc/sysv/consts/hwcap.h>

#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include "http.h"
#include "powersampler.h"
#include "ascii_digits.h"
#include "system.h"
#include "utils.h"
#include "cmd.h"

#include "llama.cpp/cores.h"
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-metal.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/string.h"
#include "llama.cpp/common.h"
#include "llama.cpp/ggml-cuda.h"

#include "llamafile/llamafile.h"
#include "llamafile/compute.h"
#include "llamafile/json.h"

using jt::Json;



// command line params

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
    test(const cmd_params& inst, const llama_model * lmodel, llama_context * context, int repetitions, PowerSampler * sampler) {
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
            interval.start = utils::get_time_ns();
            interval.end = 0;
            test_intervals.push_back(interval);

            if (n_prompt > 0) {
                test_prompt();
            }
            if (n_gen > 0) {
                test_gen();
            }

            test_intervals.back().end = utils::get_time_ns();
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
        interval.start = utils::get_time_ns();
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

        prompt_intervals.back().end = utils::get_time_ns();
    }

    void test_gen() {
        llama_set_n_threads(ctx, n_threads, n_threads);

        const llama_model * model = llama_get_model(ctx);
        const int32_t n_vocab = llama_n_vocab(model);

        llama_token token = llama_add_bos_token(model) ? llama_token_bos(model) : std::rand() % n_vocab;

        time_interval interval;
        interval.start = utils::get_time_ns();
        interval.end = 0;
        gen_intervals.push_back(interval);

        for (int i = 0; i < n_gen; i++) {
            llama_decode(ctx, llama_batch_get_one(&token, 1, n_prompt + i, 0));
            llama_synchronize(ctx);
            if (i == 0) {
                uint64_t ttft = utils::get_time_ns() - test_intervals.back().start;
                time_to_first_token.push_back(ttft);
            }
            token = std::rand() % n_vocab;
            t_gen = i + 1;
        }

        gen_intervals.back().end = utils::get_time_ns();
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
        return utils::avg(samples_ns);
    }

    uint64_t stdev_ns(token_metric metric = TOTAL_TPS) const {
        std::vector<uint64_t> samples_ns = get_samples_ns(metric);
        return utils::stdev(samples_ns);
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
        return utils::avg(get_ts(metric));
    }

    double stdev_ts(token_metric metric = TOTAL_TPS) const {
        return utils::stdev(get_ts(metric));
    }

    double get_tps_watt(token_metric metric = TOTAL_TPS) const {
        double power = get_power();
        double ts = avg_ts(metric);

        if (ts == 0.0 || power == 0.0) {
            return 0.0;
        }

        return avg_ts(metric) / get_power();
    }

    double ttft() const {
        if (time_to_first_token.empty()) {
            return 0.0;
        }
        return utils::avg(time_to_first_token);
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
            // "name", "power_watts", "vram_used_mb", "ttft_ms"
            "name", "power_watts", "ttft_ms",
            "main_gpu"
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
            std::to_string(avg_ts(PROMPT_TPS)), std::to_string(get_tps_watt(PROMPT_TPS)), std::to_string(stdev_ts(PROMPT_TPS)),
            std::to_string(avg_ts(GEN_TPS)), std::to_string(get_tps_watt(GEN_TPS)), std::to_string(stdev_ts(GEN_TPS)),
            // name, std::to_string(power), std::to_string(monitor_result.vram), std::to_string(ttft() / 1e6)
            name, std::to_string(power), std::to_string(ttft() / 1e6),
            std::to_string(main_gpu)
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
const std::string test::cpu_info     = llamafile_describe_cpu();
const std::string test::gpu_info     = ""; //get_gpu_info(); // [jart]

struct OutputWriter {
    virtual ~OutputWriter() {};
    virtual void write(const char* buf, ...) = 0;
    virtual void flush() = 0;
};

struct FileWriter : public OutputWriter {
    FILE* fout;

    FileWriter(FILE* f): fout(f) {}

    void write(const char* format, ...) override {
        va_list args;
        va_start(args, format);
        vfprintf(fout, format, args);
        va_end(args);
    }
    
    void flush() override {
        fflush(fout);
    }
};

struct StringWriter : public OutputWriter {
    std::string& output;

    StringWriter(std::string& str) : output(str) {}

    void write(const char* format, ...) override {
        va_list args;
        va_start(args, format);
        char tmp[1024];
        vsnprintf(tmp, sizeof(tmp), format, args);
        output += tmp;
        va_end(args);
    }

    void flush() override {}
};

struct printer {
    virtual ~printer() {}

    std::unique_ptr<OutputWriter> writer;
    
    void set_file_output(FILE* fout) {
        writer = std::make_unique<FileWriter>(fout);
    }
    
    void set_string_output(std::string& output) {
        writer = std::make_unique<StringWriter>(output);
    }

    virtual void print_header(const cmd_params & params, AcceleratorInfo accelerator_info, RuntimeInfo runtime_info, SystemInfo sys_info) { (void) params; }
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

    void print_header(const cmd_params & params, AcceleratorInfo accelerator_info, RuntimeInfo runtime_info, SystemInfo sys_info) override  {
        std::vector<std::string> fields = test::get_fields();
        writer->write("%s\n", utils::join(fields, ",").c_str());
        (void) params;
    }

    void print_test(const test & t) override {
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_csv);
        writer->write("%s\n", utils::join(values, ",").c_str());
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

void print_header(const cmd_params & params, AcceleratorInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
    writer->write("{\n");
    
    // Print RuntimeInfo object
    writer->write("  \"runtime_info\": {\n");
    writer->write("    \"name\": \"%s\",\n", "llamafile");
    writer->write("    \"version\": \"%s\",\n", runtime_info.llamafile_version);
    writer->write("    \"commit\": \"%s\"\n", runtime_info.llama_commit);
    writer->write("  },\n");

    // Print SystemInfo object
    writer->write("  \"system_info\": {\n");
    writer->write("    \"cpu_name\": \"%s\",\n", sys_info.cpu);
    writer->write("    \"cpu_arch\": \"%s\",\n", sys_info.system_architecture);
    writer->write("    \"ram_gb\": %.2f,\n", sys_info.ram_gb);
    writer->write("    \"kernel_type\": \"%s\",\n", sys_info.kernel_type);
    writer->write("    \"kernel_release\": \"%s\",\n", sys_info.kernel_release);
    writer->write("    \"version\": \"%s\"\n", sys_info.version);
    writer->write("  },\n");

    // Print GPUInfo object
    writer->write("  \"accelerator_info\": {\n");
    writer->write("    \"name\": \"%s\",\n", gpu_info.name);
    writer->write("    \"manufacturer\": \"%s\",\n", gpu_info.manufacturer);
    writer->write("    \"memory_gb\": %.2f,\n", gpu_info.total_memory_gb);
    // TODO
    writer->write("    \"type\": \"%s\"\n", (FLAG_gpu >= 0 && llamafile_has_gpu()) ? "GPU" : "CPU");
    writer->write("  },\n");

    // Start the results array
    writer->write("  \"results\": [\n");
    
    (void) params;
}

    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            writer->write("      \"%s\": %s,\n", fields.at(i).c_str(), format_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        if (first) {
            first = false;
        } else {
            writer->write(",\n");
        }
        writer->write("    {\n");
        print_fields(test::get_fields(), t.get_values());
        writer->write("      \"samples_ns\": [ %s ],\n", utils::join(t.get_samples_ns(), ", ").c_str());
        writer->write("      \"samples_ts\": [ %s ]\n", utils::join(t.get_ts(), ", ").c_str());
        writer->write("    }");
        writer->flush();
    }

    void print_footer() override {
        writer->write("\n  ]\n");
        writer->write("}");
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
        // if (field == "vram") {
        //     return 15;
        // }

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

    void print_header(const cmd_params & params, AcceleratorInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
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
        // fields.emplace_back("vram");
        bool is_cpu_backend = test::get_backend() == "CPU" || test::get_backend() == "BLAS";
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        fields.emplace_back("tokens processed");
        fields.emplace_back("pp t/s");
        fields.emplace_back("tg t/s");
        fields.emplace_back("pp t/s/watt");
        fields.emplace_back("tg t/s/watt");
        fields.emplace_back("ttft");

        writer->write("|");
        for (const auto & field : fields) {
            writer->write(" %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        writer->write("\n");
        writer->write("|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            writer->write(" %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        writer->write("\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        float power = t.get_power();

        writer->write("|");
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
                snprintf(buf, sizeof(buf), "%.4f", t.get_tps_watt(PROMPT_TPS));

                value = buf;
            } else if (field == "tg t/s/watt") {
                snprintf(buf, sizeof(buf), "%.4f", t.get_tps_watt(GEN_TPS));

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
            writer->write(" %*s |", width, value.c_str());
        }
        writer->write("\n");
    }

    void print_footer() override {
        // writer->write("\nbuild: %s (%d)\n", test::build_commit.c_str(), test::build_number); // [jart]
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

    void print_header(const cmd_params & params, AcceleratorInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
        std::vector<std::string> fields = test::get_fields();
        writer->write("CREATE TABLE IF NOT EXISTS test (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            writer->write("  %s %s%s\n", fields.at(i).c_str(), get_sql_field_type(fields.at(i)).c_str(),  i < fields.size() - 1 ? "," : "");
        }
        writer->write(");\n");
        writer->write("\n");
        (void) params;
    }

    void print_test(const test & t) override {
        writer->write("INSERT INTO test (%s) ", utils::join(test::get_fields(), ", ").c_str());
        writer->write("VALUES (");
        std::vector<std::string> values = t.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            writer->write("'%s'%s", values.at(i).c_str(), i < values.size() - 1 ? ", " : "");
        }
        writer->write(");\n");
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

std::string getUserConfirmation() {
    std::string user_input;
    printf("\nDo you want to send the data to the public database? (yes/no): ");
    std::getline(std::cin, user_input);
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);
    return user_input;
}

__attribute__((__constructor__(101))) static void init(void) {
    FLAG_gpu = LLAMAFILE_GPU_AUTO;
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

    LoadZipArgs(&argc, &argv);

    // try to set locale for unicode characters in markdown
    setlocale(LC_CTYPE, "C.UTF-8");  // [jart]

    cmd_params params = parse_cmd_params(argc, argv);
    FLAGS_READY = true;

    RuntimeInfo runtime_info;
    get_runtime_info(&runtime_info);

    SystemInfo sys_info;
    get_sys_info(&sys_info);

    AcceleratorInfo accelerator_info;
    get_accelerator_info(&accelerator_info, &params);

    unsigned int main_gpu = params.main_gpu;

    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
        ggml_backend_metal_log_set_callback(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    std::string req_payload;
    json_printer* req_printer = new json_printer();
    req_printer->set_string_output(req_payload);
    req_printer->print_header(params, accelerator_info, runtime_info, sys_info);

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
    p->set_file_output(stdout);
    p->print_header(params, accelerator_info, runtime_info, sys_info);

    // std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    llama_model * lmodel = nullptr;
    // const cmd_params_instance * prev_inst = nullptr;

    PowerSampler * sampler = getPowerSampler(100, main_gpu);

    pthread_t print_thread;

    // for (const auto & base_inst : params_instances) {
    //     int num_gen = base_inst.n_prompt > 0 ? 4096: 2048;
    //     for (int context_size = 16; context_size <= num_gen; context_size *= 2) {
    //         // TODO this is a total hack.
    //         cmd_params_instance inst = base_inst;
    //         if (base_inst.n_prompt > 0) {
    //             inst.n_prompt = context_size;
    //         } else {
    //             inst.n_gen = context_size;
    //         }

    //         // keep the same model between tests when possible
    //         if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
    //             if (lmodel) {
    //                 llama_free_model(lmodel);
    //             }

    //             lmodel = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
    //             if (lmodel == NULL) {
    //                 fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
    //                 return 1;
    //             }

    //             // TODO build a json payload still..
    //             // printf("Model N Params: %d\n", llama_model_n_params(lmodel));

    //             prev_inst = &inst;
    //         }

    //         llama_context_params cparams = inst.to_llama_cparams();
    //         cparams.n_ctx = context_size;

    //         llama_context * ctx = llama_new_context_with_model(lmodel, cparams);
    //         if (ctx == NULL) {
    //             fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
    //             llama_free_model(lmodel);
    //             return 1;
    //         }

    //         test t(inst, lmodel, ctx, params.reps, sampler);

    //         update_t_gen_column_args argv = {t, p.get()};
    //         pthread_t update_thread;
    //         int rc = pthread_create(&update_thread, NULL, update_t_gen_column, &argv);
    //         if (rc) {
    //             fprintf(stderr, "Error creating pthread: %d\n", rc);
    //             return EXIT_FAILURE;
    //         }
    //         t.run();

    //         pthread_join(update_thread, NULL);

    //         llama_print_timings(ctx);

    //         llama_free(ctx);
    //     }
    // }

    for (const auto & test_cfg : baseline_tests) {
        cmd_params inst = params;
        inst.n_prompt = test_cfg.n_prompt;
        inst.n_gen = test_cfg.n_gen;

        if (!lmodel) {
            lmodel = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
            if (lmodel == NULL) {
                fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                return 1;
            }
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
            fprintf(stderr, "Error creating pthread: %d\n", rc);
            return EXIT_FAILURE;
        }
        t.run();

        pthread_join(update_thread, NULL);
        req_printer->print_test(t);

        llama_print_timings(ctx);

        llama_free(ctx);
    }

    llama_free_model(lmodel);

    p->print_footer();
    req_printer->print_footer();

    llama_backend_free();

    std::pair<Json::Status, Json> data =
              Json::parse(req_payload);

    if (data.first != Json::success) {
        printf("Error parsing json\n");
        return 1;
    }

    if (!data.second.isObject()) {
        printf("Json is not an object\n");
        return 1;
    }
    if (data.second["results"].isArray()) {
        std::vector<Json> results = data.second["results"].getArray();
        
        double total_prompt_tps = 0.0;
        double total_gen_tps = 0.0;
        double total_ttft_ms = 0.0;
        double total_power_watts = 0.0;
        int valid_count = 0;

        for (const auto & result : results) {
            if (result.isObject()) {
                bool valid_entry = true;
                
                // Check if all required fields exist and are numbers
                if (!result.contains("prompt_tps") || 
                    !result.contains("gen_tps") || 
                    !result.contains("ttft_ms") || 
                    !result.contains("power_watts")) {
                    valid_entry = false;
                } else {
                    // Get a non-const reference to access the values
                    auto& obj = const_cast<Json&>(result);
                    if (!obj["prompt_tps"].isNumber() ||
                        !obj["gen_tps"].isNumber() ||
                        !obj["ttft_ms"].isNumber() ||
                        !obj["power_watts"].isNumber()) {
                        valid_entry = false;
                    }
                }

                if (valid_entry) {
                    auto& obj = const_cast<Json&>(result);
                    total_prompt_tps += obj["prompt_tps"].getNumber();
                    total_gen_tps += obj["gen_tps"].getNumber();
                    total_ttft_ms += obj["ttft_ms"].getNumber();
                    total_power_watts += obj["power_watts"].getNumber();
                    valid_count++;
                }
            }
        }

        if (valid_count > 0) {
            double avg_prompt_tps = total_prompt_tps / valid_count;
            double avg_gen_tps = total_gen_tps / valid_count;
            double avg_ttft_ms = total_ttft_ms / valid_count;


            // calculate the geometric mean of the performance values for a score
            double score = pow(avg_prompt_tps * avg_gen_tps * (1000 / avg_ttft_ms), 1.0 / 3.0) * 10;
            printf("\n\033[1;35mYour LocalScore:\n\n", score);
            ascii_display::printLargeNumber((int)score);
            printf("\033[0m\n");
            printf("- Prompt Processing: \t %.2f tok/s\n", avg_prompt_tps);
            printf("- Token Generation: \t %.2f tok/s\n", avg_gen_tps);
            printf("- Time to First Token:\t %.2f ms\n", avg_ttft_ms);
        } else {
            printf("No valid results found in the array\n");
        }
    } else {
        printf("Results is not an array\n");
    }

    // Ask user for confirmation before sending the data
    std::string user_cnf;
    if (!params.send_results) {
        user_cnf = getUserConfirmation();
    }

    // TODO make this a func or something, also retry if it fails to send. 3 times. backoff
    if (user_cnf == "yes" || user_cnf == "y" || params.send_results) {
        printf("\nSending data to the public database...\n");
        Response response = POST("https://llamascore.vercel.app/api/store/results", req_payload, {
            {"Content-Type", "application/json"}
        });

        if (response.status == 200) {
            printf("Data sent to the public database.\n");

            // parse the response json
            std::pair<Json::Status, Json> json =
              Json::parse(response.body);

            if (json.first != Json::success) {
                printf("Error parsing response json\n");
                return 1;
            }
            if (!json.second.isObject()) {
                printf("Response json is not an object\n");
                return 1;
            }

            if (json.second["id"].isString()) {
                printf("Result Link: https://llamascore.vercel.app/result/%s\n", json.second["id"].getString().c_str());
            }
        } else {
            printf("Error sending data to the public database. Status: %d\n", response.status);
        }
    } else {
        printf("\nData not sent to the public database.\n");
    }

    return 0;
}
