#include "helper_cuda ptx.h"
#include <cstdint>

__global__
__launch_bounds__(256, 1) void sgemm_128x256x8(int m,
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

    const unsigned smem_a_padding = 256;
    const unsigned smem_a_size = smem_a_padding * 8;
    const unsigned smem_a_ld = 132;
    const unsigned smem_b_padding = 256;
    const unsigned smem_b_size = smem_b_padding * 8;
    const unsigned smem_b_ld = 256;

    __shared__ float __align__(2 * smem_a_size * sizeof(float))
        smem_ptr[2 * (smem_a_size + smem_b_size)];

    float accumulator[16][8]{};

    unsigned ldg_a_bitmask = 0x0;
    unsigned ldg_b_bitmask = 0x0;

    float* smem_a_ptr = smem_ptr;
    float* smem_b_ptr = smem_ptr + 2 * smem_a_size;

    unsigned warp_id = threadIdx.x / 32;
    unsigned lane_id = threadIdx.x % 32;

    unsigned ldg_a_start_x = threadIdx.x % 8;
    unsigned ldg_a_start_y = blockIdx.y * 128 + threadIdx.x / 8;
    unsigned ldg_a_start = ldg_a_start_x + ldg_a_start_y * lda;
    const float* ldg_a_ptr = A + ldg_a_start;
    unsigned ldg_a_offsets_y[4];
    unsigned ldg_a_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_a_offsets_y[i] = i * 32;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
        if (m_idx < m) { ldg_a_bitmask ^= (0x1 << i); }
    }

    unsigned ldg_b_start_x = blockIdx.x * 256 + threadIdx.x % 128;
    unsigned ldg_b_start_y = threadIdx.x / 128;
    unsigned ldg_b_start = ldg_b_start_x + ldg_b_start_y * ldb;
    const float* ldg_b_ptr = B + ldg_b_start;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n_idx = ldg_b_start_x + i * 128;
        if (n_idx < n) { ldg_b_bitmask ^= (0x1 << i); }
    }

    unsigned sts_a_start_x = threadIdx.x / 8;
    unsigned sts_a_start_y = threadIdx.x % 8;
    unsigned sts_a_start = sts_a_start_x + sts_a_start_y * smem_a_ld;
    float* sts_a_ptr = smem_a_ptr + sts_a_start;

    unsigned sts_b_start_x = threadIdx.x % 128;
    unsigned sts_b_start_y = threadIdx.x / 128;
    unsigned sts_b_start = sts_b_start_x + sts_b_start_y * smem_b_ld;
    float* sts_b_ptr = smem_b_ptr + sts_b_start;

    uint64_t sts_a_addr;
    uint64_t sts_b_addr;

    CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
    CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

    unsigned n_blocks_k = (k + 7) / 8 - 1;
    unsigned first_block_k_size = k - 8 * n_blocks_k;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = ldg_a_start_x < first_block_k_size;
        bool guard_m = ldg_a_bitmask & (0x1 << i);
        bool guard = guard_k && guard_m;
        CP_ASYNC_IGNORE_SRC_PTX(sts_a_addr + i * 32 * sizeof(float),
                                ldg_a_ptr + ldg_a_offsets[i],
                                (unsigned)guard)
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = (2 * i + threadIdx.x / 128) < first_block_k_size;
        bool guard_n = ldg_b_bitmask & (0x1 << 0);
        bool guard = guard_k && guard_n;
        CP_ASYNC_IGNORE_SRC_PTX(sts_b_addr + i * 2 * 256 * sizeof(float),
                                ldg_b_ptr + i * 2 * ldb,
                                (unsigned)guard)
        guard_n = ldg_b_bitmask & (0x1 << 1);
        guard = guard_k && guard_n;
        CP_ASYNC_IGNORE_SRC_PTX(sts_b_addr + (i * 2 * 256 + 128) * sizeof(float),
                                ldg_b_ptr + i * 2 * ldb + 128,
                                (unsigned)guard)
    }

    WAIT_ALL_PTX;
    __syncthreads();

    ldg_a_ptr += first_block_k_size;
    ldg_b_ptr += first_block_k_size * ldb;
    sts_a_addr ^= 8192;
    sts_b_addr ^= 8192;

    float frag_a[2][16];
    float frag_b[2][8];

    uint64_t lds_a_addr;
    uint64_t lds_b_addr;

    unsigned lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
    unsigned lane_id_mapped_y = (lane_id / 2) % 4;
    unsigned warp_id_mapped_x = 64 * (warp_id % 4);
    unsigned warp_id_mapped_y = 64 * (warp_id / 4);

    unsigned lds_a_offset = 4 * lane_id_mapped_y + warp_id_mapped_y;
    unsigned lds_b_offset = 4 * lane_id_mapped_x + warp_id_mapped_x;
    float* lds_a_ptr = smem_a_ptr + lds_a_offset;
    float* lds_b_ptr = smem_b_ptr + lds_b_offset;

    CVTA_TO_SHARED_PTX(lds_a_addr, lds_a_ptr);
    CVTA_TO_SHARED_PTX(lds_b_addr, lds_b_ptr);

    LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
    LDS128_PTX(frag_a[0][4],
               frag_a[0][5],
               frag_a[0][6],
               frag_a[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_a[0][8],
               frag_a[0][9],
               frag_a[0][10],
               frag_a[0][11],
               lds_a_addr + 32 * sizeof(float));
    LDS128_PTX(frag_a[0][12],
               frag_a[0][13],
               frag_a[0][14],
               frag_a[0][15],
               lds_a_addr + 48 * sizeof(float));
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
    LDS128_PTX(frag_b[0][4],
               frag_b[0][5],
               frag_b[0][6],
               frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));


    for (int block_ks = 0; block_ks < n_blocks_k; block_ks++) {
#pragma unroll
        for (int warp_k = 0; warp_k < 7; warp_k += 1) {
            unsigned prefetch = warp_k + 1;
            unsigned frag_idx = warp_k & 1;
            unsigned frag_next_idx = (warp_k + 1) & 1;

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

            LDS128_PTX(frag_a[frag_next_idx][8],
                       frag_a[frag_next_idx][9],
                       frag_a[frag_next_idx][10],
                       frag_a[frag_next_idx][11],
                       lds_a_addr + (prefetch * smem_a_ld + 32) * sizeof(float));

            LDS128_PTX(frag_a[frag_next_idx][12],
                       frag_a[frag_next_idx][13],
                       frag_a[frag_next_idx][14],
                       frag_a[frag_next_idx][15],
                       lds_a_addr + (prefetch * smem_a_ld + 48) * sizeof(float));

            if (warp_k < 4) {
                bool guard = ldg_a_bitmask & (0x1 << warp_k);
                CP_ASYNC_GUARD_PTX(sts_a_addr + warp_k * 32 * sizeof(float),
                                   ldg_a_ptr + ldg_a_offsets[warp_k],
                                   (unsigned)guard)

                guard = ldg_b_bitmask & (0x1 << 0);
                CP_ASYNC_GUARD_PTX(sts_b_addr + warp_k * 2 * 256 * sizeof(float),
                                   ldg_b_ptr + warp_k * 2 * ldb,
                                   (unsigned)guard)
                guard = ldg_b_bitmask & (0x1 << 1);
                CP_ASYNC_GUARD_PTX(sts_b_addr + (warp_k * 2 * 256 + 128) * sizeof(float),
                                   ldg_b_ptr + warp_k * 2 * ldb + 128,
                                   (unsigned)guard)
            }


#pragma unroll
            for (int i = 0; i < 16; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 16; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
            }
        }

        WAIT_ALL_PTX;
        __syncthreads();

        sts_a_addr ^= 8192;
        sts_b_addr ^= 8192;
        lds_a_addr ^= 8192;
        lds_b_addr ^= 8192;

        LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
        LDS128_PTX(frag_a[0][4],
                   frag_a[0][5],
                   frag_a[0][6],
                   frag_a[0][7],
                   lds_a_addr + 16 * sizeof(float));
        LDS128_PTX(frag_a[0][8],
                   frag_a[0][9],
                   frag_a[0][10],
                   frag_a[0][11],
                   lds_a_addr + 32 * sizeof(float));
        LDS128_PTX(frag_a[0][12],
                   frag_a[0][13],
                   frag_a[0][14],
                   frag_a[0][15],
                   lds_a_addr + 48 * sizeof(float));
        LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
        LDS128_PTX(frag_b[0][4],
                   frag_b[0][5],
                   frag_b[0][6],
                   frag_b[0][7],
                   lds_b_addr + 32 * sizeof(float));

        ldg_a_ptr += 8;
        ldg_b_ptr += 8 * n;
    }

#pragma unroll
    for (int warp_k = 0; warp_k < 7; warp_k += 1) {
        unsigned prefetch = warp_k + 1;
        unsigned frag_idx = warp_k & 1;
        unsigned frag_next_idx = (warp_k + 1) & 1;

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
        LDS128_PTX(frag_a[frag_next_idx][8],
                   frag_a[frag_next_idx][9],
                   frag_a[frag_next_idx][10],
                   frag_a[frag_next_idx][11],
                   lds_a_addr + (prefetch * smem_a_ld + 32) * sizeof(float));
        LDS128_PTX(frag_a[frag_next_idx][12],
                   frag_a[frag_next_idx][13],
                   frag_a[frag_next_idx][14],
                   frag_a[frag_next_idx][15],
                   lds_a_addr + (prefetch * smem_a_ld + 48) * sizeof(float));

#pragma unroll
        for (int i = 0; i < 16; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 16; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] *= alpha;
        }
    }

    uint64_t sts_c_addr;
    unsigned sts_c_offset = 512 * warp_id + 4 * 4 * 8 * lane_id_mapped_y + 4 * lane_id_mapped_x;
    CVTA_TO_SHARED_PTX(sts_c_addr, smem_ptr + sts_c_offset);

    const float* lds_c_ptr = (float*)((smem_ptr + 512 * warp_id + lane_id));

    unsigned m_idx = blockIdx.y * 128 + warp_id_mapped_y;
    unsigned n_idx = blockIdx.x * 256 + warp_id_mapped_x + lane_id;
    float* stg_c_ptr = C + m_idx * ldc + n_idx;

    if (m_idx < m) {
#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
#pragma unroll 2
            for (int j = 0; j < 2; ++j) {
                __syncthreads();
#pragma unroll 4
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
                    if (beta != 0) {
                        float c;
                        LDG32_GUARD_MOV0_PTX(c,
                                             stg_c_ptr + (i * 16 + p) * n + j * 32,
                                             (unsigned)guard);
                        c *= beta;
                        STG32_GUARD_PTX(c + lds_c_ptr[p * 32],
                                        stg_c_ptr + (i * 16 + p) * n + j * 32,
                                        (unsigned)guard);
                    } else {
                        STG32_GUARD_PTX(lds_c_ptr[p * 32],
                                        stg_c_ptr + (i * 16 + p) * n + j * 32,
                                        (unsigned)guard);
                    }
                }
            }
        }
    }
}