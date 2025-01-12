#include <helper_matrix.h>
#include <helper_string.h>
#include <sgemm.cuh>
#include <string>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using string = std::string;

#define MATSIZE_MAX_DEFAULT  1024
#define MATSIZE_MIN_DEFAULT  2
#define MATSIZE_STEP_DEFAULT 1
#define SAVEDIR_DEFAULT      "test_results"

int
main(int argc, char** argv) {
    srand(time(NULL));

    std::vector<string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(string{argv[i]});
    }

    int matsize_max = get_cmd_line_arg_int(args, "mmax", MATSIZE_MAX_DEFAULT);
    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);

    int sep_len = 25;
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing...\n");
    printf("%.*s\n", sep_len, "===================================================");

    string test_summary{};
    string test_full_info{};
    string failed_tests{};
    const int n_tests = (matsize_max - matsize_min) / matsize_step + 1;
    std::vector<cmp_result> test_results(n_tests, cmp_result{});
    int n_failed = 0;

    for (int i = 0; i < n_tests; i += 1) {
        size_t matsize = matsize_min + i * matsize_step;
        size_t m = matsize, n = matsize, k = matsize;
        size_t lda = k;
        size_t ldb = n;
        size_t ldc = n;

        float* A_host = alloc_mat_host(m * lda * sizeof(float));
        float* B_host = alloc_mat_host(k * ldb * sizeof(float));
        float* C_host = alloc_mat_host(m * ldc * sizeof(float));
        float* C_ref_host = alloc_mat_host(m * ldc * sizeof(float));

        init_random(A_host, m * lda);
        init_random(B_host, k * ldb);
        init_random(C_host, m * ldc);
        checkCudaErrors(
            cudaMemcpy(C_ref_host, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        float* A_device = alloc_mat_device(m * lda * sizeof(float));
        float* B_device = alloc_mat_device(k * ldb * sizeof(float));
        float* C_device = alloc_mat_device(m * ldc * sizeof(float));
        float* C_ref_device = alloc_mat_device(m * ldc * sizeof(float));

        float alpha = 1.5;
        float beta = 0.5;

        checkCudaErrors(
            cudaMemcpy(A_device, A_host, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(B_device, B_host, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_device, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_ref_device, C_ref_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        sgemm(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
        sgemm_basic(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_ref_device, ldc);
        cudaMemcpy(C_host, C_device, m * ldc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref_host, C_ref_device, m * ldc * sizeof(float), cudaMemcpyDeviceToHost);

        cmp_result result = compare_mats(C_ref_host, C_host, m * ldc, 1e-4, false, true);
        string result_string{};
        if (!result.equal) {
            n_failed += 1;
            failed_tests += to_string(i);
            failed_tests += " ";
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Test #%i | matsize = %lu | %s\n", i, matsize, result_string.c_str());

        test_full_info += "Test #" + to_string(i);
        test_full_info += " Matsize = " + to_string(matsize) + ": " + result.debug_info;
        test_results[i] = result;

        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFreeHost(C_ref_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
        checkCudaErrors(cudaGetLastError());
    }
    test_summary += "\n=============== SUMMARY ===============\n";
    test_summary += "PASSED: " + to_string(n_tests - n_failed) + " / " + to_string(n_tests) + "\n";
    test_summary += "FAILED: " + ((failed_tests.size() > 0) ? failed_tests : "0") + "\n";
    test_full_info += test_summary;
    printf("%s", test_summary.c_str());

    fs::path work_dir_path = fs::current_path();
    fs::path store_path = work_dir_path / save_dir / "sgemm.cu.txt";
    std::ofstream test_results_file(store_path);
    test_results_file << test_full_info.c_str();
    test_results_file.close();
    printf("Test results stored in %s\n", store_path.c_str());
    return 0;
}