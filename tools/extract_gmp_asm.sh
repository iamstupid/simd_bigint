#!/bin/bash
# Extract a GMP m4 .asm kernel to namespaced ELF/SysV .s for simd_bigint.
#   extract_gmp_asm.sh <outname> <OPERATION_flag> <asm-path-rel-to-gmp-root>
# e.g. extract_gmp_asm.sh submul_1 OPERATION_submul_1 mpn/x86_64/zen/aorsmul_1.asm
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"          # absolute (m4 runs from /tmp)
GMP_ROOT="$(cd "$HERE/../../.." && pwd)"
CFG="$HERE/gmp_linux_config.m4"
OUT="$HERE/../src/x86_64/$1.s"
( cd /tmp && m4 -D"$2" "$CFG" "$GMP_ROOT/$3" ) | sed 's/__gmpn_/sbn_/g' > "$OUT"
gcc -c "$OUT" -o /tmp/_chk.o && echo "$1: $(nm /tmp/_chk.o | grep ' T ' | awk '{print $3}' | tr '\n' ' ')"
