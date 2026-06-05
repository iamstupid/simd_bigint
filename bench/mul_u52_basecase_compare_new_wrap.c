#include "../include/mul_new.h"

#ifdef __cplusplus
extern "C"
#endif
void
bench_mul_u52_basecase_new(unsigned long long *rp,
                           const unsigned long long *ap,
                           const unsigned long long *bp,
                           unsigned long long an,
                           unsigned long long bn)
{
  mul_u52_basecase((pvec)rp, (pvec)ap, (pvec)bp, an, bn);
}
