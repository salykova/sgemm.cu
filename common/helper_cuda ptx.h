#pragma once

#define CVTA_TO_SHARED_PTX(addr, smem_ptr) \
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr));

#define LDG32_GUARD_PTX(reg, ptr, guard)               \
    {                                                  \
        asm volatile("{.reg .pred p;\n\t"              \
                     "setp.ne.u32 p, %2, 0;\n\t"       \
                     "@p ld.global.f32 %0, [%1];}\n\t" \
                     : "=f"(reg)                       \
                     : "l"(ptr), "r"(guard));          \
    }

#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard)          \
    {                                                  \
        asm volatile("{.reg .pred p;\n\t"              \
                     "setp.ne.u32 p, %2, 0;\n\t"       \
                     "@!p mov.b32 %0, 0;\n\t"          \
                     "@p ld.global.f32 %0, [%1];}\n\t" \
                     : "=f"(reg)                       \
                     : "l"(ptr), "r"(guard));          \
    }

#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
    {                                                                          \
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t"            \
                     :                                                         \
                     : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)); \
    }

#define LDS128_PTX(reg0, reg1, reg2, reg3, addr)                      \
    {                                                                 \
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"   \
                     : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3) \
                     : "l"(addr));                                    \
    }

#define STS32_PTX(reg, addr) \
    { asm volatile("st.shared.f32 [%0], %1;\n" : : "l"(addr), "f"(reg)); }

#define STG32_GUARD_PTX(reg, ptr, guard)                \
    {                                                   \
        asm volatile("{.reg .pred p;\n\t"               \
                     "setp.ne.u32 p, %2, 0;\n\t"        \
                     "@p st.global.f32 [%0], %1;}\n\t"  \
                     :                                  \
                     : "l"(ptr), "f"(reg), "r"(guard)); \
    }

#define COMMIT_GROUP_PTX asm volatile("cp.async.commit_group;");

#define WAIT_GROUP_PTX(N) asm volatile("cp.async.wait_group %0;" : : "n"(N))

#define WAIT_ALL_PTX asm volatile("cp.async.wait_all ;")

#define CP_ASYNC_GUARD_PTX(addr, ptr, guard)                          \
    {                                                                 \
        asm volatile("{.reg .pred p;\n\t"                             \
                     "setp.ne.u32 p, %2, 0;\n\t"                      \
                     "@p cp.async.ca.shared.global [%0], [%1], 4;}\n" \
                     :                                                \
                     : "l"(addr), "l"(ptr), "r"(guard));              \
    }

#define CP_ASYNC_IGNORE_SRC_PTX(addr, ptr, guard)                     \
    {                                                                 \
        asm volatile("{.reg .pred p;\n\t"                             \
                     "setp.eq.u32 p, %2, 0;\n\t"                      \
                     "cp.async.ca.shared.global [%0], [%1], 4, p;}\n" \
                     :                                                \
                     : "l"(addr), "l"(ptr), "r"(guard));              \
    }
