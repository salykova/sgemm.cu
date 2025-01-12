#include "helper_cuda ptx.h"
#include <cstdint>

__global__
__launch_bounds__(256, 2) void sgemm_128x128x8(int m,
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

    // Abbreviations:
    // ldg - load global
    // lds - load shared
    // stg - store global
    // sts - store shared
    // cvta - convert address

    const int smem_a_padding = 256;
    const int smem_a_size = smem_a_padding * 8;
    const int smem_a_ld = 132; // leading dimension
    const int smem_b_padding = 128;
    const int smem_b_size = smem_b_padding * 8;
    const int smem_b_ld = 128; // leading dimension

    __shared__ float __align__(2 * smem_a_size * sizeof(float))
        smem_ptr[2 * (smem_a_size + smem_b_size)];

    // C accumulator
    float accumulator[8][8]{};

    // Registers for (global memory -> shared memory) transfers
    float ldg_a_buffer[4];
    float ldg_b_buffer[4];

    // Bitmasks to track in-bounds and out-of-bounds global memory reads
    unsigned ldg_a_bitmask = 0x0;
    unsigned ldg_b_bitmask = 0x0;

    float* smem_a_ptr = smem_ptr;
    float* smem_b_ptr = smem_ptr + 2 * smem_a_size;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int ldg_a_start_x = threadIdx.x % 8;
    int ldg_a_start_y = blockIdx.y * 128 + 4 * (threadIdx.x / 8);
    int ldg_a_start = ldg_a_start_x + ldg_a_start_y * lda;
    const float* ldg_a_ptr = A + ldg_a_start;
    int ldg_a_offsets_y[4];
    int ldg_a_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_a_offsets_y[i] = i;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
        // if global memory access is in-bounds, flip corresponding bit
        if (m_idx < m) { ldg_a_bitmask ^= (0x1 << i); }
    }

    int ldg_b_start_x = blockIdx.x * 128 + threadIdx.x % 32;
    int ldg_b_start_y = threadIdx.x / 32;
    int ldg_b_start = ldg_b_start_x + ldg_b_start_y * ldb;
    const float* ldg_b_ptr = B + ldg_b_start;
    int ldg_b_offsets_x[4];
    int ldg_b_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_b_offsets_x[i] = 32 * i;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_b_offsets[i] = ldg_b_offsets_x[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = ldg_b_start_x + ldg_b_offsets_x[i];
        // if global memory access is in-bounds, flip corresponding bit
        if (n_idx < n) { ldg_b_bitmask ^= (0x1 << i); }
    }

    int sts_a_start_x = 4 * (threadIdx.x / 8);
    int sts_a_start_y = threadIdx.x % 8;
    int sts_a_start = sts_a_start_x + sts_a_start_y * smem_a_ld;
    float* sts_a_ptr = smem_a_ptr + sts_a_start;

    int sts_b_start_x = threadIdx.x % 32;
    int sts_b_start_y = threadIdx.x / 32;
    int sts_b_start = sts_b_start_x + sts_b_start_y * smem_b_ld;
    float* sts_b_ptr = smem_b_ptr + sts_b_start;
    int sts_b_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        sts_b_offsets[i] = 32 * i;
    }

    uint64_t sts_a_addr;
    uint64_t sts_b_addr;

    // Convert from generic to .shared state space
    CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
    CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

    // if (k % 8 == 0) {n_blocks_k = k/8 - 1} else {n_blocks_k = k/8;}
    int n_blocks_k = (k + 7) / 8 - 1;
    int first_block_k_size = k - 8 * n_blocks_k;

    // Load first blocks from global memory to shared memory
    // {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = ldg_a_start_x < first_block_k_size;
        bool guard_m = ldg_a_bitmask & (0x1 << i);
        bool guard = guard_k && guard_m;
        LDG32_GUARD_MOV0_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i], (unsigned)guard);
    }
    STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2], ldg_a_buffer[3], sts_a_addr);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = ldg_b_start_y < first_block_k_size;
        bool guard_n = ldg_b_bitmask & (0x1 << i);
        bool guard = guard_k && guard_n;
        LDG32_GUARD_MOV0_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i], (unsigned)guard);
    }
#pragma unroll
    for (int i = 0; i < 4; i += 1) {
        STS32_PTX(ldg_b_buffer[i], sts_b_addr + sts_b_offsets[i] * sizeof(float));
    }
    __syncthreads();
    // }

    float frag_a[2][8];
    float frag_b[2][8];

    uint64_t lds_a_addr;
    uint64_t lds_b_addr;

    int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
    int lane_id_mapped_y = (lane_id / 2) % 4;
    int warp_id_mapped_x = 64 * (warp_id % 2);
    int warp_id_mapped_y = 32 * (warp_id / 2);

    int lds_a_start = 4 * lane_id_mapped_y + warp_id_mapped_y;
    int lds_b_start = 4 * lane_id_mapped_x + warp_id_mapped_x;
    float* lds_a_ptr = smem_a_ptr + lds_a_start;
    float* lds_b_ptr = smem_b_ptr + lds_b_start;

    // Convert from generic to .shared state space
    CVTA_TO_SHARED_PTX(lds_a_addr, lds_a_ptr);
    CVTA_TO_SHARED_PTX(lds_b_addr, lds_b_ptr);

    // Load first fragments from shared memory
    // {
    LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
    LDS128_PTX(frag_a[0][4],
               frag_a[0][5],
               frag_a[0][6],
               frag_a[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
    LDS128_PTX(frag_b[0][4],
               frag_b[0][5],
               frag_b[0][6],
               frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));
    // }

    // Move pointers to next blocks
    ldg_a_ptr += first_block_k_size;
    ldg_b_ptr += first_block_k_size * ldb;

    // Switch shared memory buffers
    sts_a_addr ^= 8192;
    sts_b_addr ^= 4096;

    // Iterate over k and divide into ks blocks
    for (int block_k = 0; block_k < n_blocks_k; block_k++) {

        // Prefetch next blocks from global memory
        // {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            bool guard_m = (ldg_a_bitmask & (0x1 << i));
            LDG32_GUARD_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i], (unsigned)guard_m);

            bool guard_n = (ldg_b_bitmask & (0x1 << i));
            LDG32_GUARD_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i], (unsigned)guard_n);
        }
        // }
#pragma unroll
        for (int warp_k = 0; warp_k < 8; warp_k += 1) {
            int prefetch = (warp_k + 1) % 8;
            int frag_idx = warp_k & 1;
            int frag_next_idx = (warp_k + 1) & 1;

            // Prefetch next fragments from shared memory
            // {
            LDS128_PTX(frag_a[frag_next_idx][0],
                       frag_a[frag_next_idx][1],
                       frag_a[frag_next_idx][2],
                       frag_a[frag_next_idx][3],
                       lds_a_addr + prefetch * smem_a_ld * sizeof(float));
            LDS128_PTX(frag_a[frag_next_idx][4],
                       frag_a[frag_next_idx][5],
                       frag_a[frag_next_idx][6],
                       frag_a[frag_next_idx][7],
                       lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
            LDS128_PTX(frag_b[frag_next_idx][0],
                       frag_b[frag_next_idx][1],
                       frag_b[frag_next_idx][2],
                       frag_b[frag_next_idx][3],
                       lds_b_addr + prefetch * smem_b_ld * sizeof(float));
            LDS128_PTX(frag_b[frag_next_idx][4],
                       frag_b[frag_next_idx][5],
                       frag_b[frag_next_idx][6],
                       frag_b[frag_next_idx][7],
                       lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));
            // }

            // Update the accumulator
            // {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
                }
            }
            // }
        }

        // Store prefetched blocks to shared memory
        // {
        STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2], ldg_a_buffer[3], sts_a_addr);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            STS32_PTX(ldg_b_buffer[i], sts_b_addr + i * 32 * sizeof(float));
        }
        __syncthreads();
        // }

        // Switch shared memory buffers
        sts_a_addr ^= 8192;
        sts_b_addr ^= 4096;
        lds_a_addr ^= 8192;
        lds_b_addr ^= 4096;

        // Move pointers to next blocks
        ldg_a_ptr += 8;
        ldg_b_ptr += 8 * n;

        // Load first fragments from shared memory
        // {
        LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
        LDS128_PTX(frag_a[0][4],
                   frag_a[0][5],
                   frag_a[0][6],
                   frag_a[0][7],
                   lds_a_addr + 16 * sizeof(float));
        LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
        LDS128_PTX(frag_b[0][4],
                   frag_b[0][5],
                   frag_b[0][6],
                   frag_b[0][7],
                   lds_b_addr + 32 * sizeof(float));
        // }
    }

    // Compute last block
    // {
#pragma unroll
    for (int warp_k = 0; warp_k < 8; warp_k += 1) {
        int prefetch = (warp_k + 1) % 8;
        int frag_idx = warp_k & 1;
        int frag_next_idx = (warp_k + 1) & 1;

        LDS128_PTX(frag_a[frag_next_idx][0],
                   frag_a[frag_next_idx][1],
                   frag_a[frag_next_idx][2],
                   frag_a[frag_next_idx][3],
                   lds_a_addr + prefetch * smem_a_ld * sizeof(float));
        LDS128_PTX(frag_a[frag_next_idx][4],
                   frag_a[frag_next_idx][5],
                   frag_a[frag_next_idx][6],
                   frag_a[frag_next_idx][7],
                   lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
        LDS128_PTX(frag_b[frag_next_idx][0],
                   frag_b[frag_next_idx][1],
                   frag_b[frag_next_idx][2],
                   frag_b[frag_next_idx][3],
                   lds_b_addr + prefetch * smem_b_ld * sizeof(float));
        LDS128_PTX(frag_b[frag_next_idx][4],
                   frag_b[frag_next_idx][5],
                   frag_b[frag_next_idx][6],
                   frag_b[frag_next_idx][7],
                   lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));

#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
            }
        }
    }
    // }

    // Calculate alpha * A * B
    // {
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] *= alpha;
        }
    }
    // }

    // Store the accumulator to global memory
    // {
    uint64_t sts_c_addr;
    int sts_c_offset = 512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
    CVTA_TO_SHARED_PTX(sts_c_addr, smem_ptr + sts_c_offset);

    float* lds_c_ptr = (float*)((smem_ptr + 512 * warp_id + lane_id));

    int m_idx = blockIdx.y * 128 + warp_id_mapped_y;
    int n_idx = blockIdx.x * 128 + warp_id_mapped_x + lane_id;
    float* stg_c_ptr = C + m_idx * ldc + n_idx;

    if (m_idx < m) {
#pragma unroll 1
        for (int i = 0; i < 2; ++i) {
#pragma unroll 1
            for (int j = 0; j < 2; ++j) {
                __syncthreads();
#pragma unroll 2
                for (int p = 0; p < 4; ++p) {
                    STS128_PTX(accumulator[i * 4 + p][j * 4],
                               accumulator[i * 4 + p][j * 4 + 1],
                               accumulator[i * 4 + p][j * 4 + 2],
                               accumulator[i * 4 + p][j * 4 + 3],
                               sts_c_addr + p * 8 * sizeof(float4));
                }
                __syncthreads();
#pragma unroll 4
                for (int p = 0; p < 16; ++p) {
                    int m_edge = m - (m_idx + i * 16);
                    int n_pos = n_idx + j * 32;
                    bool guard = p < m_edge && n_pos < n;
                    // if (beta != 0.0) {compute (beta*C + accumulator) and write to global memory}
                    if (beta != 0) {
                        float c;
                        LDG32_GUARD_MOV0_PTX(c,
                                             stg_c_ptr + (i * 16 + p) * n + j * 32,
                                             (unsigned)guard);
                        c *= beta;
                        STG32_GUARD_PTX(c + lds_c_ptr[p * 32],
                                        stg_c_ptr + (i * 16 + p) * n + j * 32,
                                        (unsigned)guard);
                    }
                    // if (beta == 0.0) {directly store the accumulator to global memory}
                    else {
                        STG32_GUARD_PTX(lds_c_ptr[p * 32],
                                        stg_c_ptr + (i * 16 + p) * n + j * 32,
                                        (unsigned)guard);
                    }
                }
            }
        }
    }
    // }
}