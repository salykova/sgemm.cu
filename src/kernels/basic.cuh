__global__ void
kernel_basic(int m,
             int n,
             int k,
             const float alpha,
             const float* A,
             int lda,
             const float* B,
             int ldb,
             const float beta,
             float* C,
             int ldc) {
    // Operands A, B, C: row-major format

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= n || y >= m) return;
    float acc = 0;
    for (int p = 0; p < k; p++) {
        acc += A[y * lda + p] * B[x + p * ldb];
    }
    acc *= alpha;
    float c = C[x + y * ldc];
    acc += c * beta;
    C[x + y * ldc] = acc;
}
