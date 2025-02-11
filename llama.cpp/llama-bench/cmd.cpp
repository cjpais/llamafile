#include <algorithm>

#include "cmd.h"
#include "llama.cpp/cores.h"
#include <cosmo.h>

static const cmd_params cmd_params_defaults = {
    /* model         */ "", // [jart] no default guessing
    /* n_prompt      */ 0,
    /* n_gen         */ 0,
    /* n_batch       */ 2048,
    /* n_ubatch      */ 512,
    /* type_k        */ X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16,
    /* type_v        */ X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16,
    /* n_threads     */ cpu_get_num_math(),
    /* gpu           */ LLAMAFILE_GPU_AUTO,
    /* n_gpu_layers  */ 9999,
    /* split_mode    */ LLAMA_SPLIT_MODE_NONE,
    /* main_gpu      */ UINT_MAX,
    /* no_kv_offload */ false,
    /* flash_attn    */ false,
    /* tensor_split  */ std::vector<float>(llama_max_devices(), 0.0f),
    /* use_mmap      */ true,
    /* embeddings    */ false,
    /* numa          */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps          */ 1,
    /* verbose       */ false,
    /* send_results  */ false,
    /* output_format */ MARKDOWN,
};

llama_model_params cmd_params::to_llama_mparams() const {
    llama_model_params mparams = llama_model_default_params();

    mparams.n_gpu_layers = n_gpu_layers;
    mparams.split_mode = split_mode;
    mparams.main_gpu = main_gpu;
    mparams.tensor_split = tensor_split.data();
    mparams.use_mmap = use_mmap;

    return mparams;
}

bool cmd_params::equal_mparams(const cmd_params & other) const {
    return model == other.model &&
           n_gpu_layers == other.n_gpu_layers &&
           split_mode == other.split_mode &&
           main_gpu == other.main_gpu && 
           use_mmap == other.use_mmap &&
           tensor_split == other.tensor_split;
}

llama_context_params cmd_params::to_llama_cparams() const {
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

cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params = cmd_params_defaults;
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";
    const char split_delim = ',';

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-mg" || arg == "--main-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.main_gpu = std::stoi(argv[i]);
        } else if (arg == "-fa" || arg == "--flash-attn") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.flash_attn = std::stoi(argv[i]);
        } else if (arg == "--recompile") {
            FLAG_recompile = true;            
        } else if (arg == "--gpu" || arg == "-g") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            FLAG_gpu = llamafile_gpu_parse(argv[i]);
            if (FLAG_gpu == LLAMAFILE_GPU_ERROR) {
                fprintf(stderr, "error: invalid --gpu flag value: %s\n", argv[i]);
                exit(1);
            }
            if (FLAG_gpu >= 0) {
                params.n_gpu_layers = 9999;
            } else if (FLAG_gpu == LLAMAFILE_GPU_DISABLE) {
                params.n_gpu_layers = 0;
            }
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
        } else if (arg == "-y" || arg == "--send-results") {
            params.send_results = true;
        } else if (arg[0] == '-') {
            invalid_param = true;
            break;
        } else {
            params.model = argv[i];
        }
    }
    if (invalid_param) {
        fprintf(stderr, "%s: invalid parameter for argument: %s\n", program_invocation_name, arg.c_str());
        exit(1);
    }
    if (params.model.empty()) {
        fprintf(stderr, "%s: missing operand\n", program_invocation_name);
        exit(1);
    }

    return params;
}

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CSV:      return "csv";
        case JSON:     return "json";
        case MARKDOWN: return "md";
        case SQL:      return "sql";
        default: GGML_ASSERT(!"invalid output format");
    }
}

void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>                     (default: %s)\n", cmd_params_defaults.model.c_str());
    printf("  -g, --gpu <auto|amd|apple|nvidia|disabled> (default: \"auto\")\n");
    printf("  -mg, --main-gpu <i>                        (default: %d)\n", cmd_params_defaults.main_gpu);
    printf("  -o, --output <csv|json|md|sql>             (default: %s)\n", output_format_str(cmd_params_defaults.output_format));
    printf("  -v, --verbose                              (default: %s)\n", cmd_params_defaults.verbose ? "1" : "0");
    printf("  -y, --send-results                         (default: %s)\n", cmd_params_defaults.send_results ? "1" : "0");
}