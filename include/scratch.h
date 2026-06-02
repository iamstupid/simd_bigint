/* scratch.h - two-tier bump scratch allocator for a bignum library.
 *
 * One mechanism (a bump arena over an mmap reservation), configured twice:
 *
 *   STACK tier  - small/local allocations (size < page_size).
 *                 64-byte aligned, raw increment. Default reserve 256 MiB.
 *   HEAP  tier  - large transform buffers (size >= page_size).
 *                 page (2 MiB) aligned, raw increment, THP-backed.
 *                 Default reserve 64 GiB (configurable).
 *
 * salloc() routes by size; everything else is shared. Memory is never freed
 * per-allocation: a scope marks both bump pointers on entry and restores them
 * on exit (SCRATCH guard), which frees everything allocated in that scope from
 * either tier in two pointer writes. Reservations are kernel-placed and
 * MAP_NORESERVE, so untouched space costs only address space; exhaustion is a
 * hard stop, not a silent degrade.
 *
 * USAGE
 *   scratch *s = scratch_thread();          // or scratch_create_ex(...)
 *   void karatsuba(..., scratch *s) {
 *       SCRATCH(s);                          // marks stack+heap, restores on return
 *       limb *t = SALLOC(s, limb, half);     // small -> stack tier, 64-aligned
 *       karatsuba(..., s);                   // child marks nest below ours
 *   }
 *   void ntt(..., scratch *s) {
 *       SCRATCH(s);
 *       limb *xf = SALLOC(s, limb, N);       // large -> heap tier, 2-MiB aligned + THP
 *   }
 *
 * Header-only. Define SCRATCH_IMPLEMENTATION in exactly one TU (for the
 * thread-local default arena's storage).
 *
 * Compile-time configuration (define before include):
 *   SCRATCH_PAGE            routing threshold + heap alignment. Default 2 MiB.
 *   SCRATCH_STACK_RESERVE   stack tier reservation. Default 256 MiB.
 *   SCRATCH_HEAP_RESERVE    heap tier reservation.  Default 64 GiB.
 *   SCRATCH_ALIGN           stack tier alignment.   Default 64 (AVX-512 line).
 *   SCRATCH_EXPLICIT_COMMIT reserve PROT_NONE + mprotect on growth, so true OOM
 *                           is a catchable error at alloc time, not SIGBUS on
 *                           first touch. Default is lazy (NORESERVE) commit.
 *   SCRATCH_OOM(msg)        override the abort-on-exhaustion handler.
 *
 * For a 4 KiB-page configuration:  -DSCRATCH_PAGE=4096 -DSCRATCH_STACK_RESERVE=(16u<<20)
 */
#ifndef SCRATCH_H
#define SCRATCH_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#ifndef SCRATCH_PAGE
#define SCRATCH_PAGE ((size_t)2 << 20)            /* 2 MiB */
#endif
#ifndef SCRATCH_STACK_RESERVE
#define SCRATCH_STACK_RESERVE ((size_t)256 << 20) /* 256 MiB */
#endif
#ifndef SCRATCH_HEAP_RESERVE
#define SCRATCH_HEAP_RESERVE ((size_t)64 << 30)   /* 64 GiB */
#endif
#ifndef SCRATCH_ALIGN
#define SCRATCH_ALIGN 64u
#endif

#ifndef SCRATCH_OOM
#include <stdio.h>
#include <stdlib.h>
#define SCRATCH_OOM(msg) (fprintf(stderr, "scratch: %s\n", (msg)), abort())
#endif

#ifdef __cplusplus
#define SCRATCH_TLS thread_local
extern "C" {
#else
#define SCRATCH_TLS _Thread_local
#endif

/* ----- ASan: mark/restore makes use-after-free silent, so poison on restore -- */
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define SCRATCH__ASAN 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#  define SCRATCH__ASAN 1
#endif
#ifdef SCRATCH__ASAN
void __asan_poison_memory_region(void const volatile *, size_t);
void __asan_unpoison_memory_region(void const volatile *, size_t);
#define SCRATCH_POISON(p, n)   __asan_poison_memory_region((p), (n))
#define SCRATCH_UNPOISON(p, n) __asan_unpoison_memory_region((p), (n))
#else
#define SCRATCH_POISON(p, n)   ((void)0)
#define SCRATCH_UNPOISON(p, n) ((void)0)
#endif

/* =============================== region (one bump tier) ===================== */
typedef struct region {
    uint8_t *at;       /* bump pointer                                          */
    uint8_t *cap;      /* end of usable reservation                             */
    uint8_t *base;     /* aligned start; restore floor                          */
    uint8_t *dirty;    /* high-water ever written: [dirty,cap) is kernel-zero   */
    uint8_t *map;      /* raw mmap base (pre-alignment) for munmap              */
    size_t   maplen;   /* raw mmap length for munmap                            */
#ifdef SCRATCH_EXPLICIT_COMMIT
    uint8_t *commit;   /* end of RW-committed region                            */
#endif
} region;

static inline uintptr_t scratch__align_up(uintptr_t p, size_t a) {
    return (p + (a - 1)) & ~(uintptr_t)(a - 1);
}
static inline void scratch__oom(const char *m) { SCRATCH_OOM(m); }

/* Backing-page policy. SCRATCH_PAGE is only a software routing/alignment
 * threshold; THIS chooses what the kernel actually backs the region with. */
typedef enum scratch_pagemode {
    SCRATCH_PAGES_DEFAULT,   /* no hint: leave THP up to the system            */
    SCRATCH_PAGES_4K,        /* force small pages: MADV_NOHUGEPAGE (opt OUT)    */
    SCRATCH_PAGES_THP,       /* hint huge pages: MADV_HUGEPAGE (best-effort)    */
    SCRATCH_PAGES_HUGETLB    /* guaranteed 2 MiB hugetlb, or fall back to THP   */
} scratch_pagemode;

/* Reserve `reserve` bytes usable, base aligned to `base_align`, backed per
 * `mode`. Anonymous modes are lazy (NORESERVE, or PROT_NONE under explicit
 * commit). HUGETLB is eager/pinned and falls back to THP if no huge pages are
 * available, so the lazy reservation model degrades gracefully. */
static inline region region_create(size_t reserve, size_t base_align, scratch_pagemode mode) {
    region r;
    void  *m = MAP_FAILED;

    /* ---- explicit hugetlb: deterministic 2 MiB, eager + pinned ---- */
    if (mode == SCRATCH_PAGES_HUGETLB) {
#if defined(MAP_HUGETLB)
        size_t hp = (size_t)2 << 20;
        size_t hlen = scratch__align_up(reserve, hp);
        int hflags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
#ifdef MAP_HUGE_2MB
                     | MAP_HUGE_2MB
#endif
                     ;
        m = mmap(NULL, hlen, PROT_READ | PROT_WRITE, hflags, -1, 0);
        if (m != MAP_FAILED) {
            r.map = (uint8_t *)m; r.maplen = hlen;
            r.base = (uint8_t *)m;          /* hugetlb is huge-page aligned */
            r.at = r.dirty = r.base;
            r.cap = r.base + reserve;
#ifdef SCRATCH_EXPLICIT_COMMIT
            r.commit = r.cap;               /* hugetlb is fully committed */
#endif
            SCRATCH_POISON(r.base, reserve);
            return r;
        }
#endif
        mode = SCRATCH_PAGES_THP;           /* no huge pages -> fall through to THP */
    }

    /* ---- anonymous: DEFAULT / 4K / THP (and the HUGETLB fallback) ---- */
    size_t maplen = reserve + base_align;          /* slack to align the base */
    int prot  = PROT_READ | PROT_WRITE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#ifdef SCRATCH_EXPLICIT_COMMIT
    prot  = PROT_NONE;
    flags = MAP_PRIVATE | MAP_ANONYMOUS;
#endif
    m = mmap(NULL, maplen, prot, flags, -1, 0);
    if (m == MAP_FAILED) scratch__oom("mmap reserve failed");
    r.map = (uint8_t *)m; r.maplen = maplen;
    r.base = (uint8_t *)scratch__align_up((uintptr_t)m, base_align);
    r.at = r.dirty = r.base;
    r.cap = r.base + reserve;
#ifdef SCRATCH_EXPLICIT_COMMIT
    r.commit = r.base;
#endif
    switch (mode) {
    case SCRATCH_PAGES_4K:
#ifdef MADV_NOHUGEPAGE
        madvise(r.base, reserve, MADV_NOHUGEPAGE);     /* force 4 KiB, no THP collapse */
#endif
        break;
    case SCRATCH_PAGES_THP:
#ifdef MADV_HUGEPAGE
        madvise(r.base, reserve, MADV_HUGEPAGE);       /* best-effort 2 MiB */
#endif
        break;
    default:
        break;                                          /* DEFAULT: system decides */
    }
    SCRATCH_POISON(r.base, reserve);
    return r;
}

static inline void region_destroy(region *r) {
    if (r->map) munmap(r->map, r->maplen);
    memset(r, 0, sizeof *r);
}

#ifdef SCRATCH_EXPLICIT_COMMIT
static inline void scratch__commit(region *r, uint8_t *upto) {
    uint8_t *newc = (uint8_t *)scratch__align_up((uintptr_t)upto, 4096u);
    if (newc > r->cap) newc = r->cap;
    if (mprotect(r->commit, (size_t)(newc - r->commit), PROT_READ | PROT_WRITE) != 0)
        scratch__oom("mprotect commit failed");
    r->commit = newc;
}
#endif

/* The hot path: align the bump pointer up, raw-increment, one cap check. */
static inline void *region_alloc(region *r, size_t n, size_t align) {
    uintptr_t p    = scratch__align_up((uintptr_t)r->at, align);
    uint8_t  *next = (uint8_t *)p + n;
    if (__builtin_expect(next > r->cap, 0)) scratch__oom("scratch tier exhausted");
#ifdef SCRATCH_EXPLICIT_COMMIT
    if (__builtin_expect(next > r->commit, 0)) scratch__commit(r, next);
#endif
    r->at = next;
    if (next > r->dirty) r->dirty = next;
    SCRATCH_UNPOISON((void *)p, n);
    return (void *)p;
}

/* Zeroed alloc that skips memset on bytes never touched since create/trim
 * (anonymous pages fault in zeroed): free zero-padding for NTT buffers. */
static inline void *region_alloc_zeroed(region *r, size_t n, size_t align) {
    uint8_t *d0  = r->dirty;
    uint8_t *p   = (uint8_t *)region_alloc(r, n, align);
    if (p < d0) {
        uint8_t *end = (p + n < d0) ? p + n : d0;
        memset(p, 0, (size_t)(end - p));
    }
    return p;
}

static inline void region_trim(region *r) {
#ifdef MADV_DONTNEED
    uint8_t *from = (uint8_t *)scratch__align_up((uintptr_t)r->at, 4096u);
    if (r->dirty > from) {
        madvise(from, (size_t)(r->dirty - from), MADV_DONTNEED);
        r->dirty = from;
    }
#else
    (void)r;
#endif
}

/* =============================== scratch (both tiers) ======================= */
typedef struct scratch {
    region stack;       /* size <  page_size : 64-aligned                       */
    region heap;        /* size >= page_size : page-aligned + THP               */
    size_t page_size;   /* routing threshold and heap alignment                 */
} scratch;

typedef struct scratch_mark {
    scratch *s;
    uint8_t *stack_at;
    uint8_t *heap_at;
} scratch_mark;

/* Full control: pick the backing-page policy for each tier explicitly. */
static inline scratch scratch_create_full(size_t page_size,
                                          size_t stack_reserve, scratch_pagemode stack_mode,
                                          size_t heap_reserve,  scratch_pagemode heap_mode) {
    scratch s;
    s.page_size = page_size;
    s.stack = region_create(stack_reserve, SCRATCH_ALIGN, stack_mode);
    s.heap  = region_create(heap_reserve,  page_size,     heap_mode);
    return s;
}

/* Page mode follows page_size: >= 2 MiB hints THP, otherwise forces 4 KiB.
 * The stack tier is always 4 KiB (small dense allocations gain nothing from
 * huge pages and would waste RSS, since a huge page faults in 2 MiB at once). */
static inline scratch scratch_create_ex(size_t page_size,
                                        size_t stack_reserve,
                                        size_t heap_reserve) {
    scratch_pagemode heap_mode =
        (page_size >= ((size_t)2 << 20)) ? SCRATCH_PAGES_THP : SCRATCH_PAGES_4K;
    return scratch_create_full(page_size,
                               stack_reserve, SCRATCH_PAGES_4K,
                               heap_reserve,  heap_mode);
}
static inline scratch scratch_create(void) {
    return scratch_create_ex(SCRATCH_PAGE, SCRATCH_STACK_RESERVE, SCRATCH_HEAP_RESERVE);
}
static inline void scratch_destroy(scratch *s) {
    region_destroy(&s->stack);
    region_destroy(&s->heap);
    s->page_size = 0;
}

/* Route by size: small -> stack (64B), large -> heap (page-aligned). */
static inline void *salloc(scratch *s, size_t n) {
    return (n < s->page_size) ? region_alloc(&s->stack, n, SCRATCH_ALIGN)
                              : region_alloc(&s->heap,  n, s->page_size);
}
static inline void *salloc_zeroed(scratch *s, size_t n) {
    return (n < s->page_size) ? region_alloc_zeroed(&s->stack, n, SCRATCH_ALIGN)
                              : region_alloc_zeroed(&s->heap,  n, s->page_size);
}
#define SALLOC(s, T, n)  ((T *)salloc((s), sizeof(T) * (size_t)(n)))
#define SALLOC0(s, T, n) ((T *)salloc_zeroed((s), sizeof(T) * (size_t)(n)))

/* ------------------------------ mark / restore ------------------------------ */
static inline scratch_mark scratch_save(scratch *s) {
    scratch_mark m = { s, s->stack.at, s->heap.at };
    return m;
}
static inline void scratch_restore(scratch_mark *m) {
    region *st = &m->s->stack, *hp = &m->s->heap;
    SCRATCH_POISON(m->stack_at, (size_t)(st->at - m->stack_at));
    SCRATCH_POISON(m->heap_at,  (size_t)(hp->at - m->heap_at));
    st->at = m->stack_at;
    hp->at = m->heap_at;
}

#define SCRATCH__CAT2(x, y) x##y
#define SCRATCH__CAT(x, y)  SCRATCH__CAT2(x, y)
/* Scope guard: marks both tiers here, restores both on scope exit. */
#define SCRATCH(s)                                                  \
    scratch_mark SCRATCH__CAT(scratch__m_, __COUNTER__)             \
        __attribute__((cleanup(scratch_restore))) = scratch_save(s)
#define SCRATCH_NAMED(name, s) \
    scratch_mark name __attribute__((cleanup(scratch_restore))) = scratch_save(s)

/* ------------------------------ utilities ----------------------------------- */
static inline size_t scratch_stack_used(const scratch *s) { return (size_t)(s->stack.at - s->stack.base); }
static inline size_t scratch_heap_used(const scratch *s)  { return (size_t)(s->heap.at  - s->heap.base);  }

/* Return physical pages above both bump pointers to the OS (between big ops). */
static inline void scratch_trim(scratch *s) { region_trim(&s->stack); region_trim(&s->heap); }

/* Pin a specific hot buffer to keep the kernel out during a long multiply.
 * Bounded by RLIMIT_MEMLOCK / cgroup memory.max; pin the buffer, not the arena. */
static inline int scratch_pin(void *p, size_t n)   { return mlock(p, n); }
static inline int scratch_unpin(void *p, size_t n) { return munlock(p, n); }

/* ------------------------- per-thread default scratch ----------------------- */
extern SCRATCH_TLS scratch scratch__tls;
extern SCRATCH_TLS int     scratch__tls_ready;

static inline scratch *scratch_thread(void) {
    if (__builtin_expect(!scratch__tls_ready, 0)) {
        scratch__tls = scratch_create();
        scratch__tls_ready = 1;
    }
    return &scratch__tls;
}
static inline void scratch_thread_release(void) {
    if (scratch__tls_ready) { scratch_destroy(&scratch__tls); scratch__tls_ready = 0; }
}

#ifdef SCRATCH_IMPLEMENTATION
SCRATCH_TLS scratch scratch__tls;
SCRATCH_TLS int     scratch__tls_ready;
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SCRATCH_H */