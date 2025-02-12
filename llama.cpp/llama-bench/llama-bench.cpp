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
#include "benchmark.h"

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
        writer->write("      \"samples_ns\": [ %s ]\n", utils::join(t.get_samples_ns(), ", ").c_str());
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

    int calculate_total_width() const {
        int total_width = 0;
        for (const auto & field : fields) {
            int width = get_field_width(field);
            if (width < 0) {
                width = std::abs(width);
            }
            total_width += width;
        }
        total_width += fields.size() * 3 + 1;
        return total_width;
    }    

    void print_header(const cmd_params & params, AcceleratorInfo gpu_info, RuntimeInfo runtime_info, SystemInfo sys_info) override {
        fields.emplace_back("test");
        fields.emplace_back("run number");
        fields.emplace_back("avg time"); // [jart]
        fields.emplace_back("power");
        fields.emplace_back("tokens processed");
        fields.emplace_back("pp t/s");
        fields.emplace_back("tg t/s");
        // fields.emplace_back("pp t/s/watt");
        // fields.emplace_back("tg t/s/watt");
        fields.emplace_back("ttft");

        int total_width = calculate_total_width();

        std::string border(total_width, '-');
        border[0] = '+';
        border[total_width-1] = '+';

        // Create the GPU info string and calculate padding
        char gpu_info_str[256];
        int content_length = snprintf(gpu_info_str, sizeof(gpu_info_str), 
                                    "%s - %.2f GiB", 
                                    gpu_info.name, 
                                    gpu_info.total_memory_gb);

        if (content_length < 0 || static_cast<size_t>(content_length + 2) > total_width) {
            throw std::runtime_error("GPU info string too long for display width");
        }
        int padding = (total_width - 2 - content_length) / 2;


        writer->write("%s\n", border.c_str());
        writer->write("|%*s%s%*s|\n",
            padding, "", 
            gpu_info_str,
            padding + (content_length % 2 == 0 ? 0 : 1), "");
        writer->write("%s\n", border.c_str());

        writer->write("|");
        for (const auto & field : fields) {
            writer->write(" %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        writer->write("\n");
        writer->write("|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            writer->write(" %s |", std::string(std::abs(width), '-').c_str());
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
                if (t.gen_intervals.size() != 0) {
                    time_interval curr_interval = t.gen_intervals[t.curr_run];
                
                    if (curr_interval.end == 0) {
                        // get the live tps instead of avg
                        uint64_t elapsed_ns = utils::get_time_ns() - curr_interval.start;
                        float elapsed_s = elapsed_ns / 1e9;
                        float tps = t.t_gen / elapsed_s;
                        snprintf(buf, sizeof(buf), "%.2f", tps);
                    }
                }
            
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
                float ttft = t.ttft() / 1e6;

                if (ttft < 1000) {
                    snprintf(buf, sizeof(buf), "%.2f ms", ttft);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f s", ttft / 1e3);
                }

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
        int total_width = calculate_total_width();
        std::string border(total_width, '-');
        border[0] = '+';
        border[total_width-1] = '+';
        writer->write("%s\n", border.c_str());
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
    printf("\nDo you want to your results to the public database? (yes/no): ");
    std::getline(std::cin, user_input);
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);
    return user_input;
}

__attribute__((__constructor__(101))) static void init(void) {
    FLAG_gpu = LLAMAFILE_GPU_AUTO;
}

static void warmup_run(llama_model *model, llama_context *ctx, cmd_params inst) {
    printf("Warming up... ");
    int n_batch = inst.n_batch;
    int n_processed = 0;
    int n_prompt = inst.n_prompt;
    int n_gen = inst.n_gen;

    const int32_t n_vocab = llama_n_vocab(model);
    std::vector<llama_token> tokens(n_batch);

    llama_kv_cache_clear(ctx);

    // warmup prompt
    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = n_processed == 0 && llama_add_bos_token(model)
                        ? llama_token_bos(model)
                        : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        llama_decode(
            ctx, llama_batch_get_one(tokens.data(), n_tokens, n_processed, 0));
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);

    // warmup gen
    llama_token token = llama_add_bos_token(model) ? llama_token_bos(model)
                                                   : std::rand() % n_vocab;
    for (int i = 0; i < n_gen; i++) {
        llama_decode(ctx, llama_batch_get_one(&token, 1, n_prompt + i, 0));
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }

    llama_free(ctx);

    printf("Warmup complete.\n\n");
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


    PowerSampler * sampler = getPowerSampler(100, main_gpu);
    pthread_t print_thread;

    cmd_params inst = params;
    inst.n_prompt = 1024;
    inst.n_gen = 16;

    llama_model * lmodel = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
    if (lmodel == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
        return 1;
    }
    llama_context_params cparams = inst.to_llama_cparams();
    cparams.n_ctx = inst.n_prompt + inst.n_gen;
    llama_context * ctx = llama_new_context_with_model(lmodel, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
        llama_free_model(lmodel);
        return 1;
    }
    warmup_run(lmodel, ctx, inst);

    p->print_header(params, accelerator_info, runtime_info, sys_info);
    for (const auto & test_cfg : baseline_tests) {
        inst.n_prompt = test_cfg.n_prompt;
        inst.n_gen = test_cfg.n_gen;

        cparams = inst.to_llama_cparams();
        cparams.n_ctx = test_cfg.n_prompt + test_cfg.n_gen;

        ctx = llama_new_context_with_model(lmodel, cparams);
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

        // llama_print_timings(ctx);

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
            // printf("\n\033[1;35mYour LocalScore:\n\n", score);
            printf("\n\033[1;35m");
            ascii_display::print_logo();
            printf("\n");
            ascii_display::printLargeNumber((int)score);
            printf("\033[0m\n");
            printf("\033[32mToken Generation: \t \033[1;32m%.2f\033[0m \033[3;32mtok/s\033[0m\n", avg_gen_tps);
            printf("\033[36mPrompt Processing: \t \033[1;36m%.2f\033[0m \033[3;36mtok/s\033[0m\n", avg_prompt_tps);
            printf("\033[33mTime to First Token:\t \033[1;33m%.2f\033[0m \033[3;33mms\033[0m\n", avg_ttft_ms);
            printf("\033[0m");
        } else {
            printf("No valid results found in the array\n");
        }
    } else {
        printf("Results is not an array\n");
    }

    // Ask user for confirmation before sending the data
    std::string user_cnf;
    if (params.send_results == SEND_ASK) {
        user_cnf = getUserConfirmation();
    }

    // TODO make this a func or something, also retry if it fails to send. 3 times. backoff
    if (user_cnf == "yes" || user_cnf == "y" || params.send_results == SEND_YES) {
        printf("\nSending results...\n");
        Response response = POST("https://mbp.tail73f30.ts.net/api/results", req_payload, {
            {"Content-Type", "application/json"}
        });

        if (response.status == 200) {
            // printf("Results sent!\n");
            printf("Results body: %s\n", response.body.c_str());

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

            if (json.second["id"].isNumber()) {
                printf("Result Link: https://llamascore.vercel.app/result/%d\n", (int)json.second["id"].getNumber());
            }
        } else {
            printf("Error sending data to the public database. Status: %d\n", response.status);
        }
    } else {
        printf("\nResults Not Submitted.\n");
    }

    return 0;
}
