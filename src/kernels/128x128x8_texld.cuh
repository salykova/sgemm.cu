#include "helper_cuda ptx.h"
#include <cstdint>

__global__
__launch_bounds__(256, 2) void sgemm_texld_128x128x8(int m,
                                                     int n,
                                                     int k,
                                                     const float alpha,
                                                     cudaTextureObject_t tex_a,
                                                     int lda,
                                                     cudaTextureObject_t tex_b,
                                                     int ldb,
                                                     const float beta,
                                                     float* C,
                                                     int ldc) {
    // Operands A, B, C: row-major format

    const int smem_a_padding = 128;
    const int smem_a_size = smem_a_padding * 8;
    const int smem_a_ld = 128;
    const int smem_b_padding = 128;
    const int smem_b_size = smem_b_padding * 8;
    const int smem_b_ld = 128;

    __shared__ float __align__(2 * smem_a_size * sizeof(float))
        smem_ptr[2 * (smem_a_size + smem_b_size)];

    float accumulator[8][8]{};

    float4 texld_a_buffer;
    float4 texld_b_buffer;

    float* smem_a_ptr = smem_ptr;
    float* smem_b_ptr = smem_ptr + 2 * smem_a_size;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int texld_a_offset_x = threadIdx.x % 2;
    int texld_a_offset_y = blockIdx.y * 128 + threadIdx.x / 2;
    int texld_a_offset = texld_a_offset_x + texld_a_offset_y * lda / 4;

    int texld_b_offset_x = (blockIdx.x * 128) / 4 + threadIdx.x % 32;
    int texld_b_offset_y = threadIdx.x / 32;
    int texld_b_offset = texld_b_offset_x + texld_b_offset_y * ldb / 4;

    int sts_a_offset_x = threadIdx.x / 2;
    int sts_a_offset_y = 4 * (threadIdx.x % 2);
    int sts_a_offset = sts_a_offset_x + sts_a_offset_y * smem_a_ld;
    float* sts_a_ptr = smem_a_ptr + sts_a_offset;

    int sts_b_offset_x = 4 * (threadIdx.x % 32);
    int sts_b_offset_y = threadIdx.x / 32;
    int sts_b_offset = sts_b_offset_x + sts_b_offset_y * smem_b_ld;
    float* sts_b_ptr = smem_b_ptr + sts_b_offset;

    uint64_t sts_a_addr;
    uint64_t sts_b_addr;

    CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
    CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

    int n_blocks_k = (k + 7) / 8 - 1;

    texld_a_buffer = tex1Dfetch<float4>(tex_a, texld_a_offset);
    STS32_PTX(texld_a_buffer.x, sts_a_addr);
    STS32_PTX(texld_a_buffer.y, sts_a_addr + sizeof(float) * smem_a_ld);
    STS32_PTX(texld_a_buffer.z, sts_a_addr + 2 * sizeof(float) * smem_a_ld);
    STS32_PTX(texld_a_buffer.w, sts_a_addr + 3 * sizeof(float) * smem_a_ld);

    texld_b_buffer = tex1Dfetch<float4>(tex_b, texld_b_offset);
    STS128_PTX(texld_b_buffer.x, texld_b_buffer.y, texld_b_buffer.z, texld_b_buffer.w, sts_b_addr);
    __syncthreads();

    float frag_a[2][8];
    float frag_b[2][8];

    uint64_t lds_a_addr;
    uint64_t lds_b_addr;

    int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
    int lane_id_mapped_y = (lane_id / 2) % 4;
    int warp_id_mapped_x = 64 * (warp_id % 2);
    int warp_id_mapped_y = 32 * (warp_id / 2);

    int lds_a_offset = 4 * lane_id_mapped_y + warp_id_mapped_y;
    int lds_b_offset = 4 * lane_id_mapped_x + warp_id_mapped_x;
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
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
    LDS128_PTX(frag_b[0][4],
               frag_b[0][5],
               frag_b[0][6],
               frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));

    texld_a_offset += 2;
    texld_b_offset += 2 * n;
    sts_a_addr ^= 4096;
    sts_b_addr ^= 4096;

    for (int block_ks = 0; block_ks < n_blocks_k; block_ks++) {
        texld_a_buffer = tex1Dfetch<float4>(tex_a, texld_a_offset);
        texld_b_buffer = tex1Dfetch<float4>(tex_b, texld_b_offset);

#pragma unroll
        for (int warp_k = 0; warp_k < 7; warp_k += 1) {
            int prefetch = warp_k + 1;
            int frag_idx = warp_k & 1;
            int frag_next_idx = (warp_k + 1) & 1;
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
                }
            }
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
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
            }
        }
        STS32_PTX(texld_a_buffer.x, sts_a_addr);
        STS32_PTX(texld_a_buffer.y, sts_a_addr + sizeof(float) * smem_a_ld);
        STS32_PTX(texld_a_buffer.z, sts_a_addr + 2 * sizeof(float) * smem_a_ld);
        STS32_PTX(texld_a_buffer.w, sts_a_addr + 3 * sizeof(float) * smem_a_ld);

        STS128_PTX(texld_b_buffer.x,
                   texld_b_buffer.y,
                   texld_b_buffer.z,
                   texld_b_buffer.w,
                   sts_b_addr);
        __syncthreads();

        sts_a_addr ^= 4096;
        sts_b_addr ^= 4096;
        lds_a_addr ^= 4096;
        lds_b_addr ^= 4096;

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

        texld_a_offset += 2;
        texld_b_offset += 2 * n;
    }

    // Compute last block
#pragma unroll
    for (int warp_k = 0; warp_k < 7; warp_k += 1) {
        int prefetch = warp_k + 1;
        int frag_idx = warp_k & 1;
        int frag_next_idx = (warp_k + 1) & 1;

#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
            }
        }

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
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] *= alpha;
        }
    }

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