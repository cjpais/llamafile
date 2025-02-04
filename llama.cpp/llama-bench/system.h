#pragma once

#include <string>

#define MAX_STRING_LENGTH 256

// Forward declaration if needed
struct cmd_params;

// Core data structures
struct RuntimeInfo {
    char llamafile_version[MAX_STRING_LENGTH];
    char llama_commit[MAX_STRING_LENGTH];
};

struct SystemInfo {
    char kernel_type[MAX_STRING_LENGTH];
    char kernel_release[MAX_STRING_LENGTH];
    char version[MAX_STRING_LENGTH];
    char system_architecture[MAX_STRING_LENGTH];
    char cpu[MAX_STRING_LENGTH];
    double ram_gb;
};

struct AcceleratorInfo {
    char name[MAX_STRING_LENGTH];
    char manufacturer[MAX_STRING_LENGTH];
    double total_memory_gb;
    int core_count;
    double capability;
};

// Public interface
void get_runtime_info(RuntimeInfo* info);
void get_sys_info(SystemInfo* info);
void get_accelerator_info(AcceleratorInfo* info, cmd_params* params);