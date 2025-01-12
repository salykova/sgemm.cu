#pragma once
#include <cstdlib>
#include <stdio.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char*
_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS : return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED : return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED : return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE : return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH : return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR : return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED : return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR : return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED : return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR : return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef __DRIVER_TYPES_H__
static const char*
_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char*
_cudaGetErrorEnum(CUresult error) {
    static char unknown[] = "<unknown>";
    const char* ret = NULL;
    cuGetErrorName(error, &ret);
    return ret ? ret : unknown;
}
#endif

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr,
                "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(result),
                _cudaGetErrorEnum(result),
                func);
        exit(EXIT_FAILURE);
    }
}

void
l2flush() {
    int dev_id{};
    int m_l2_size{};
    void* buffer;
    checkCudaErrors(cudaGetDevice(&dev_id));
    checkCudaErrors(cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    if (m_l2_size > 0) {
        checkCudaErrors(cudaMalloc(&buffer, static_cast<std::size_t>(m_l2_size)));
        int* m_l2_buffer = reinterpret_cast<int*>(buffer);
        checkCudaErrors(cudaMemset(m_l2_buffer, 0, static_cast<std::size_t>(m_l2_size)));
        checkCudaErrors(cudaFree(m_l2_buffer));
    }
}