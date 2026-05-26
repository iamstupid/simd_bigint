# simd_bigint

`simd_bigint` is an experimental C big-integer library aimed at modern
wide-SIMD machines, with AVX-512 as the first-class target.

The library is intended to start as an `mpn`-style arithmetic layer rather
than a high-level object system.  The public binary format is a little-endian
array of 64-bit limbs plus an `int64_t limb_size` field whose top bit encodes
the sign.  Internal kernels may freely use temporary SIMD-friendly formats
such as `u52x8`, packed complex FFT arrays, and `p50` NTT layouts.

See [docs/design.md](docs/design.md) for the current design draft and the
open choices that need to be solved before the API and kernel contracts are
frozen.
