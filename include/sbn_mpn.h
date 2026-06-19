#pragma once
// GMP zen-tuned scalar mpn kernels, m4-extracted to ELF/SysV and namespaced
// `sbn_` (so they coexist with -lgmp). Source: src/x86_64/*.s, preprocessed
// from mpn/x86_64/zen/{aorsmul_1,mul_1}.asm and mpn/x86_64/aors_n.asm via a
// linux config.m4. Same ABI/semantics as the GMP originals.
#include <stdint.h>
typedef uint64_t sbn_limb;
#ifdef __cplusplus
extern "C" {
#endif
sbn_limb sbn_submul_1(sbn_limb* rp, const sbn_limb* sp, long n, sbn_limb b); // rp -= b*sp
sbn_limb sbn_addmul_1(sbn_limb* rp, const sbn_limb* sp, long n, sbn_limb b); // rp += b*sp
sbn_limb sbn_mul_1   (sbn_limb* rp, const sbn_limb* sp, long n, sbn_limb b); // rp  = b*sp
sbn_limb sbn_add_n   (sbn_limb* rp, const sbn_limb* ap, const sbn_limb* bp, long n);
sbn_limb sbn_sub_n   (sbn_limb* rp, const sbn_limb* ap, const sbn_limb* bp, long n);
#ifdef __cplusplus
}
#endif
