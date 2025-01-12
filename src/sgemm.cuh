#include "kernels/128x128x8.cuh"
#include "kernels/basic.cuh"

#if GPUCC >= 80
#include "kernels/128x256x8.cuh"
#else
#include "kernels/128x128x8_texld.cuh"
cudaResourceDesc resDesc;
cudaTextureDesc texDesc;
cudaTextureObject_t tex_a = 0;
cudaTextureObject_t tex_b = 0;
#endif

int
div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

void
sgemm(int m,
      int n,
      int k,
      const float* alpha,
      float* A,
      int lda,
      float* B,
      int ldb,
      const float* beta,
      float* C,
      int ldc) {
    // C := alpha*A*B + beta*C
    // Operands A, B, C: row-major format
    // For column-major order, swap (A, lda) and (B, ldb) because C^T = B^T * A^T.

#if GPUCC >= 80
    static_assert(__CUDACC_VER_MAJOR__ > 11
                      || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 5)),
                  "For devices with compute compatibility >= 80, install CUDA >= 11.5");
    dim3 grid;
    dim3 threads(256);
    grid.y = div_ceil(m, 128);
    if (m > 2500 || n > 2500) {
        grid.x = div_ceil(n, 256);
        sgemm_128x256x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    } else {
        grid.x = div_ceil(n, 128);
        sgemm_128x128x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    }
#else
    dim3 grid;
    dim3 threads(256);
    grid.x = div_ceil(n, 128);
    grid.y = div_ceil(m, 128);
    bool is_aligned = (((unsigned)lda & 3u) == 0) && (((unsigned)ldb & 3u) == 0)
                      && (((unsigned long)A & 15u) == 0) && (((unsigned long)B & 15u) == 0);
    if (is_aligned) {
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.desc.y = 32;
        resDesc.res.linear.desc.z = 32;
        resDesc.res.linear.desc.w = 32;
        resDesc.res.linear.devPtr = A;
        resDesc.res.linear.sizeInBytes = m * lda * sizeof(float);
        cudaCreateTextureObject(&tex_a, &resDesc, &texDesc, NULL);
        resDesc.res.linear.devPtr = B;
        resDesc.res.linear.sizeInBytes = k * ldb * sizeof(float);
        cudaCreateTextureObject(&tex_b, &resDesc, &texDesc, NULL);
        sgemm_texld_128x128x8<<<grid, threads>>>(m,
                                                 n,
                                                 k,
                                                 *alpha,
                                                 tex_a,
                                                 lda,
                                                 tex_b,
                                                 ldb,
                                                 *beta,
                                                 C,
                                                 ldc);
        cudaDestroyTextureObject(tex_a);
        cudaDestroyTextureObject(tex_b);
    } else {
        sgemm_128x128x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    }
#endif
}

void
sgemm_basic(int m,
            int n,
            int k,
            const float* alpha,
            float* A,
            int lda,
            float* B,
            int ldb,
            const float* beta,
            float* C,
            int ldc) {
    // C := alpha*A*B + beta*C
    // Operands A, B, C: row-major format
    // For column-major order, swap (A, lda) and (B, ldb) because C^T = B^T * A^T.

    dim3 grid;
    grid.x = div_ceil(n, 16);
    grid.y = div_ceil(m, 16);
    dim3 threads(16, 16);
    kernel_basic<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
}