#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <vector>

namespace simd_bigint::reference {

struct alignas(64) Block52 {
  std::uint64_t v[8];
};

struct Decoded52 {
  std::vector<Block52> blocks;
  std::size_t nblocks = 0;
  std::size_t digits = 0;
  std::uint8_t tail_len = 0;
};

struct Ifma52Workspace {
  Decoded52 a;
  Decoded52 b;
};

struct ToomScratchStats {
  std::size_t allocation_count = 0;
  std::size_t allocated_bytes = 0;
  std::size_t current_live_bytes = 0;
  std::size_t peak_live_bytes = 0;
};

class CountingMemoryResource final : public std::pmr::memory_resource {
public:
  CountingMemoryResource() = default;
  explicit CountingMemoryResource(std::pmr::memory_resource *upstream);

  void reset_stats();
  ToomScratchStats stats() const;

private:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void *p, std::size_t bytes,
                     std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource &other) const
    noexcept override;

  std::pmr::memory_resource *upstream_ = std::pmr::null_memory_resource();
  ToomScratchStats stats_;
};

struct ToomWorkspace {
  explicit ToomWorkspace(std::size_t initial_bytes = 256 * 1024);
  ToomWorkspace(const ToomWorkspace &) = delete;
  ToomWorkspace &operator=(const ToomWorkspace &) = delete;

  void reset();
  ToomScratchStats scratch_stats() const;

  std::vector<std::byte> storage;
  std::pmr::monotonic_buffer_resource arena;
  CountingMemoryResource resource;
  Ifma52Workspace point_ws;
};

void decode52(Decoded52 &dst, const std::uint64_t *xp, std::size_t n);

void mul_predecoded_u64(std::uint64_t *rp, const Decoded52 &ap,
                        const Decoded52 &bp, std::size_t out_limbs,
                        Ifma52Workspace &ws);

void mul_basecase_u64(std::uint64_t *rp, const std::uint64_t *ap,
                      std::size_t an, const std::uint64_t *bp,
                      std::size_t bn, Ifma52Workspace &ws);

void mul_basecase_u64(std::uint64_t *rp, const std::uint64_t *ap,
                      std::size_t an, const std::uint64_t *bp,
                      std::size_t bn);

void mul_auto_u64(std::uint64_t *rp, const std::uint64_t *ap,
                  std::size_t an, const std::uint64_t *bp,
                  std::size_t bn);

void mul_auto_u64(std::uint64_t *rp, const std::uint64_t *ap,
                  std::size_t an, const std::uint64_t *bp,
                  std::size_t bn, ToomWorkspace &workspace);

void mul_toom22_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn);

void mul_toom22_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn, ToomWorkspace &workspace);

void mul_toom22_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn);

void mul_toom22_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn, ToomWorkspace &workspace);

void mul_toom32_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn);

void mul_toom32_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn, ToomWorkspace &workspace);

void mul_toom32_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn);

void mul_toom32_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn, ToomWorkspace &workspace);

void mul_toom33_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn);

void mul_toom33_u64(std::uint64_t *rp, const std::uint64_t *ap,
                    std::size_t an, const std::uint64_t *bp,
                    std::size_t bn, ToomWorkspace &workspace);

void mul_toom33_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn);

void mul_toom33_raw_u64(std::uint64_t *rp, const std::uint64_t *ap,
                        std::size_t an, const std::uint64_t *bp,
                        std::size_t bn, ToomWorkspace &workspace);

} // namespace simd_bigint::reference
