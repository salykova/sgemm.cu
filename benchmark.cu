#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_matrix.h>
#include <helper_string.h>
#include <sgemm.cuh>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using string = std::string;

#define MATSIZE_MIN_DEFAULT    1024
#define MATSIZE_STEP_DEFAULT   512
#define NPTS_DEFAULT           21
#define WARMUP_MATSIZE_DEFAULT 4096
#define WARMUP_NITER_DEFAULT   500
#define SAVEDIR_DEFAULT        "benchmark_results"
#define FILE_NAME_DEFAULT      "sgemm.cu"

int
main(int argc, char** argv) {

    std::vector<string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(string{argv[i]});
    }

    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    int npts = get_cmd_line_arg_int(args, "npts", NPTS_DEFAULT);
    int warmup_matsize = get_cmd_line_arg_int(args, "wmsize", WARMUP_MATSIZE_DEFAULT);
    int warmup_niter = get_cmd_line_arg_int(args, "wniter", WARMUP_NITER_DEFAULT);
    string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);
#if CUBLAS == 1
    string file_name = "cuBLAS";
#else
    string file_name = get_cmd_line_arg_string(args, "fname", FILE_NAME_DEFAULT);
#endif
    int sep_len = 25;

#if CUBLAS == 1
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (warmup_niter > 0) {
        printf("%.*s\n", sep_len, "===================================================");
        printf("Warm-up\n");
        printf("%.*s\n", sep_len, "===================================================");

        int m = warmup_matsize, n = warmup_matsize, k = warmup_matsize;
        int lda = k, ldb = n, ldc = n;

        float* A_host = alloc_mat_host(m * lda * sizeof(float));
        float* B_host = alloc_mat_host(k * ldb * sizeof(float));
        float* C_host = alloc_mat_host(m * ldc * sizeof(float));

        float* A_device = alloc_mat_device(m * lda * sizeof(float));
        float* B_device = alloc_mat_device(k * ldb * sizeof(float));
        float* C_device = alloc_mat_device(m * ldc * sizeof(float));

        init_random(A_host, m * lda);
        init_random(B_host, k * ldb);
        init_random(C_host, m * ldc);
        checkCudaErrors(
            cudaMemcpy(A_device, A_host, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(B_device, B_host, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_device, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup_niter; i++) {
#if CUBLAS == 1
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        m,
                        n,
                        k,
                        &alpha,
                        B_device,
                        ldb,
                        A_device,
                        lda,
                        &beta,
                        C_device,
                        ldc);
#else
            sgemm(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
#endif
            cudaDeviceSynchronize();
            fflush(stdout);
            printf("\r%i / %u", i + 1, warmup_niter);
        }
        printf("\n\n");
        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
    }

    std::vector<int> avg_gflops(npts, 0);
    std::vector<int> matsizes(npts, 0);

    printf("%.*s\n", sep_len, "===================================================");
    printf("Benchmark\n");
    printf("%.*s\n", sep_len, "===================================================");
    for (int i = 0; i < npts; i++) {
        int matsize = matsize_min + i * matsize_step;
        matsizes[i] = matsize;
        int m = matsize, n = matsize, k = matsize;
        int lda = k, ldb = n, ldc = n;

        float* A_host = alloc_mat_host(m * lda * sizeof(float));
        float* B_host = alloc_mat_host(k * ldb * sizeof(float));
        float* C_host = alloc_mat_host(m * ldc * sizeof(float));

        float* A_device = alloc_mat_device(m * lda * sizeof(float));
        float* B_device = alloc_mat_device(k * ldb * sizeof(float));
        float* C_device = alloc_mat_device(m * ldc * sizeof(float));

        init_random(A_host, m * lda);
        init_random(B_host, k * ldb);
        init_random(C_host, m * ldc);

        checkCudaErrors(
            cudaMemcpy(A_device, A_host, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(B_device, B_host, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_device, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        size_t FLOP = 2 * (size_t)m * n * k;
        double GFLOP = FLOP * 1e-9f;

        int n_iter = std::max((int)(1000*exp((-matsize + matsize_min)/3100.0)), 4);
        std::vector<float> elapsed_time_ms(n_iter, 0);

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        for (int j = 0; j < n_iter; j++) {
            checkCudaErrors(cudaEventRecord(start));
#if CUBLAS == 1
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        m,
                        n,
                        k,
                        &alpha,
                        B_device,
                        ldb,
                        A_device,
                        lda,
                        &beta,
                        C_device,
                        ldc);
#else
            sgemm(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
#endif
            checkCudaErrors(cudaEventRecord(stop));
            checkCudaErrors(cudaEventSynchronize(stop));
            checkCudaErrors(cudaEventElapsedTime(&elapsed_time_ms[j], start, stop));
            checkCudaErrors(cudaGetLastError());
            // Flush L2 cache
            l2flush();
        }

        float avg_elapsed_time_ms = 0;
        int midpoint_idx = n_iter / 2;
        for (int j = midpoint_idx; j < n_iter; j++) {
            avg_elapsed_time_ms += elapsed_time_ms[j];
        }
        avg_elapsed_time_ms = avg_elapsed_time_ms / (n_iter - midpoint_idx);

        avg_gflops[i] = (int)(GFLOP / (avg_elapsed_time_ms * 1e-3));

        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));

        printf("m=n=k=%i:\n", matsize);
        printf("%s %*i GFLOP/s\n", file_name.c_str(), 8, avg_gflops[i]);
        printf("\n");
    }

    fs::path work_dir_path = fs::current_path();
    fs::path store_sgemm_path = work_dir_path / save_dir / (file_name + ".txt");
    std::ofstream sgemm_file(store_sgemm_path);
    for (int i = 0; i < npts; i++) {
        sgemm_file << matsizes[i] << " " << avg_gflops[i] << '\n';
    }
    printf("Benchmark data stored in %s\n", store_sgemm_path.c_str());
    sgemm_file.close();
    printf("%.*s\n", sep_len, "===================================================");
    return 0;
}