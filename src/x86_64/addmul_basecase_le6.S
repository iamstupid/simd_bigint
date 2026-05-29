/*
 * simd_mpn_addmul_basecase_le6
 *
 * Tiny add-on-buffer multiplication kernel for 1..6 limbs by 1..6 limbs.
 *
 * ABI: System V x86-64
 *   rdi = rp
 *   rsi = ap
 *   rdx = an
 *   rcx = bp
 *   r8  = bn
 *
 * Contract:
 *   1 <= an,bn <= 6
 *   rp has an + bn limbs of live input/output space
 *   rp does not overlap ap or bp
 *
 * Semantics:
 *   rp[0 .. an + bn - 1] += ap[0 .. an - 1] * bp[0 .. bn - 1]
 *   return carry_out, either 0 or 1, beyond limb an + bn - 1
 *
 * Shape:
 *   - Make ap the longer operand so the fixed inner length is an.
 *   - Bias ap/rp by -(6 - an) limbs, matching mul_basecase_le6.S.
 *   - Every row is an addmul row; unlike the store-only multiply kernel there
 *     is no special first row.
 *   - ADCX carries product-high propagation through CF.
 *   - ADOX carries old-rp addition through OF.
 *   - The high limb of each row is added to memory immediately.  Only the
 *     overflow from that high-limb add is carried to the next row's high limb.
 */

    .text
    .p2align 5
    .globl simd_mpn_addmul_basecase_le6
    .type simd_mpn_addmul_basecase_le6, @function
simd_mpn_addmul_basecase_le6:
    /*
     * r12: addmul row entry address for the fixed an.
     * r13: pending carry into this row's high-limb add.
     */
    pushq   %r12
    pushq   %r13

    /*
     * Normalize so an >= bn.  The product is commutative and the result length
     * is unchanged, so this is invisible to the caller.
     */
    cmpq    %r8, %rdx
    jae     .Lnoswap
    xchgq   %rsi, %rcx
    xchgq   %rdx, %r8
.Lnoswap:

    /*
     * Bias by -(6 - an) limbs.  Entering .Ladd_eN then starts at the first
     * active ap/rp limb and falls through to offset 40; the row high limb is
     * always at offset 48.
     */
    leaq    -48(%rsi,%rdx,8), %rsi
    leaq    -48(%rdi,%rdx,8), %rdi

    /*
     * Resolve the only row-entry table once.  All rows are addmul rows.
     */
    leaq    .Ladd_tab(%rip), %r10
    movslq  (%r10,%rdx,4), %r12
    addq    %r10, %r12

    xorl    %r13d, %r13d

/* ------------------------------------------------------------------------- */
/* Outer loop: add shifted rows ap * bp[j] into the existing result.          */
/* ------------------------------------------------------------------------- */

.Louter:
    movq    (%rcx), %rdx
    xorl    %r11d, %r11d          /* zero register; also clears CF and OF */
    jmp     *%r12

/* ------------------------------------------------------------------------- */
/* Addmul row entries selected by an.                                         */
/* ------------------------------------------------------------------------- */

.Ladd_e6:
    mulxq   0(%rsi), %rax, %r10
    adoxq   0(%rdi), %rax
    movq    %rax, 0(%rdi)
    jmp     .Ladd_b5

.Ladd_e5:
    mulxq   8(%rsi), %rax, %r10
    adoxq   8(%rdi), %rax
    movq    %rax, 8(%rdi)
    jmp     .Ladd_b4

.Ladd_e4:
    mulxq   16(%rsi), %rax, %r10
    adoxq   16(%rdi), %rax
    movq    %rax, 16(%rdi)
    jmp     .Ladd_b3

.Ladd_e3:
    mulxq   24(%rsi), %rax, %r10
    adoxq   24(%rdi), %rax
    movq    %rax, 24(%rdi)
    jmp     .Ladd_b2

.Ladd_e2:
    mulxq   32(%rsi), %rax, %r10
    adoxq   32(%rdi), %rax
    movq    %rax, 32(%rdi)
    jmp     .Ladd_b1

.Ladd_e1:
    mulxq   40(%rsi), %rax, %r10
    adoxq   40(%rdi), %rax
    movq    %rax, 40(%rdi)
    jmp     .Ladd_done

/*
 * Row continuation.  r10 is previous product high; r9 receives the current
 * product high.  The last active limb still reads memory because this is an
 * add-on-buffer kernel; there is no pending high-limb value replacing memory.
 */
.Ladd_b5:
    mulxq   8(%rsi), %rax, %r9
    adcxq   %r10, %rax
    adoxq   8(%rdi), %rax
    movq    %rax, 8(%rdi)
    movq    %r9, %r10
.Ladd_b4:
    mulxq   16(%rsi), %rax, %r9
    adcxq   %r10, %rax
    adoxq   16(%rdi), %rax
    movq    %rax, 16(%rdi)
    movq    %r9, %r10
.Ladd_b3:
    mulxq   24(%rsi), %rax, %r9
    adcxq   %r10, %rax
    adoxq   24(%rdi), %rax
    movq    %rax, 24(%rdi)
    movq    %r9, %r10
.Ladd_b2:
    mulxq   32(%rsi), %rax, %r9
    adcxq   %r10, %rax
    adoxq   32(%rdi), %rax
    movq    %rax, 32(%rdi)
    movq    %r9, %r10
.Ladd_b1:
    mulxq   40(%rsi), %rax, %r9
    adcxq   %r10, %rax
    adoxq   40(%rdi), %rax
    movq    %rax, 40(%rdi)
    movq    %r9, %r10

.Ladd_done:
    /*
     * Fold both carry chains into this row's product high limb.
     */
    adoxq   %r11, %r10
    adcxq   %r11, %r10

    /*
     * Add the previous row's high-limb overflow, then add the high limb to
     * rp[an].  r13 becomes the carry-out from this high-limb update.
     *
     * MOV does not affect flags, so it preserves the carry from add r13,r10.
     */
    addq    %r13, %r10
    movl    $0, %r13d
    adcq    %r13, %r13
    addq    48(%rdi), %r10
    adcq    $0, %r13
    movq    %r10, 48(%rdi)

    decq    %r8
    jz      .Ldone
    leaq    8(%rcx), %rcx
    leaq    8(%rdi), %rdi
    jmp     .Louter

.Ldone:
    movq    %r13, %rax
    popq    %r13
    popq    %r12
    ret

/* ------------------------------------------------------------------------- */
/* PC-relative entry table indexed by an.  Entry 0 is invalid and unused.    */
/* ------------------------------------------------------------------------- */

    .p2align 2
.Ladd_tab:
    .long   .Ldone   - .Ladd_tab
    .long   .Ladd_e1 - .Ladd_tab
    .long   .Ladd_e2 - .Ladd_tab
    .long   .Ladd_e3 - .Ladd_tab
    .long   .Ladd_e4 - .Ladd_tab
    .long   .Ladd_e5 - .Ladd_tab
    .long   .Ladd_e6 - .Ladd_tab

    .size simd_mpn_addmul_basecase_le6, .-simd_mpn_addmul_basecase_le6

    .section .note.GNU-stack,"",@progbits
