#pragma once
#include "helper_cuda.h"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <string>

using string = std::string;
using std::to_string;

struct cmp_result {
    bool equal;
    int n_false;
    int n_inf;
    int n_nan;
    string debug_info;
};

inline void
init_random(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i)
        data[i] = (rand() / (double)RAND_MAX);
}

inline void
init_const(float* data, size_t size, float value) {
    for (size_t i = 0; i < size; ++i)
        data[i] = value;
}

inline void
init_range(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i)
        data[i] = i;
}

inline float*
alloc_mat_host(size_t memsize) {
    float* mat;
    checkCudaErrors(cudaMallocHost(&mat, memsize));
    return mat;
}

inline float*
alloc_mat_device(size_t memsize) {
    float* mat;
    checkCudaErrors(cudaMalloc(&mat, memsize));
    return mat;
}

inline cmp_result
compare_mats(float* mat_ref,
             float* mat,
             size_t n_elem,
             float eps = 1e-4,
             bool print = false,
             bool debug = false) {
    int n_false = 0;
    int n_nan = 0;
    int n_inf = 0;
    bool equal = true;
    string debug_info{};
    for (size_t i = 0; i < n_elem; i++) {
        if (std::isnan(mat[i])) {
            n_nan += 1;
            n_false += 1;
            continue;
        }
        if (std::isinf(mat[i])) {
            n_inf += 1;
            n_false += 1;
            continue;
        }
        if ((std::fabs((mat_ref[i] - mat[i]) / mat_ref[i])) > eps) {
            if (print) printf("mat_ref[%lu] != mat[%lu]; %f != %f\n", i, i, mat_ref[i], mat[i]);
            n_false += 1;
        }
    }
    if (debug) {
        debug_info += "n_false = " + to_string(n_false) + ", ";
        debug_info += "n_nan = " + to_string(n_nan) + ", ";
        debug_info += "n_inf = " + to_string(n_inf) + "\n";
    }
    if (n_false > 0) equal = false;
    return cmp_result{equal, n_false, n_inf, n_nan, debug_info};
}

inline void
print_matrix(float* mat, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", mat[i]);
    }
    printf("\n\n");
}