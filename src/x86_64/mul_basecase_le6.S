/*
 * simd_mpn_mul_basecase_le6
 *
 * Tiny unsigned multiplication kernel for 1..6 limbs by 1..6 limbs.
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
 *   rp has an + bn limbs of writable space
 *   rp does not overlap ap or bp
 *
 * Shape:
 *   - Make ap the longer operand so the fixed inner length is an.
 *   - Bias ap/rp by -(6 - an) limbs.  This lets every row use the same
 *     displacements: 0,8,16,24,32,40 and the final high limb at 48.
 *   - First row is store-only, matching GMP's "mul_1 first, addmul_1 later"
 *     trick.  This avoids clearing rp.
 *   - Keep each row's high carry pending in r13.  The next shifted row consumes
 *     that value directly as its last overlapping result limb, saving one
 *     high-limb store/load pair per outer row.
 *   - Remaining rows are addmul rows.  They use MULX plus the two ADX carry
 *     chains:
 *        ADCX propagates product-high/carry through CF.
 *        ADOX propagates the old-rp addition carry through OF.
 *   - A tiny jump table chooses the entry label for the fixed an, so there is
 *     no per-limb tail branch inside a row.
 */

    .text
    .p2align 5
    .globl simd_mpn_mul_basecase_le6
    .type simd_mpn_mul_basecase_le6, @function
simd_mpn_mul_basecase_le6:
    /*
     * Keep r12 as the addmul row entry address across outer rows.
     * Keep r13 as the pending high limb from the previous row.
     * All other live registers are caller-saved.
     */
    pushq   %r12
    pushq   %r13

    /*
     * Normalize so an >= bn.  For small multiplication this reduces the number
     * of outer addmul rows and gives one fixed inner length.
     */
    cmpq    %r8, %rdx
    jae     .Lnoswap
    xchgq   %rsi, %rcx
    xchgq   %rdx, %r8
.Lnoswap:

    /*
     * Bias ap and rp by -(6 - an) limbs:
     *   an=6 -> +0 bytes
     *   an=5 -> -8 bytes
     *   ...
     *   an=1 -> -40 bytes
     *
     * After this bias, entering label Ean makes the first active limb appear at
     * offset 8*(6 - an), and the row high limb is always at 48(rp).
     */
    leaq    -48(%rsi,%rdx,8), %rsi
    leaq    -48(%rdi,%rdx,8), %rdi

    /*
     * Resolve the addmul row entry once.  r12 is callee-saved, so it survives
     * all rows.  The table stores 32-bit PC-relative offsets to stay PIE-safe.
     */
    leaq    .Ladd_tab(%rip), %r10
    movslq  (%r10,%rdx,4), %r12
    addq    %r10, %r12

    /*
     * Jump to the store-only row for bp[0].
     * r11 is our zero register; xor also clears CF and OF before ADX use.
     */
    leaq    .Lmul_tab(%rip), %r10
    movslq  (%r10,%rdx,4), %rax
    addq    %r10, %rax
    movq    (%rcx), %rdx
    xorl    %r11d, %r11d
    jmp     *%rax

/* ------------------------------------------------------------------------- */
/* First row: rp = ap * bp[0].  No old rp values are read.                   */
/* ------------------------------------------------------------------------- */

.Lmul_e6:
    mulxq   0(%rsi), %rax, %r10
    movq    %rax, 0(%rdi)
    jmp     .Lmul_b5

.Lmul_e5:
    mulxq   8(%rsi), %rax, %r10
    movq    %rax, 8(%rdi)
    jmp     .Lmul_b4

.Lmul_e4:
    mulxq   16(%rsi), %rax, %r10
    movq    %rax, 16(%rdi)
    jmp     .Lmul_b3

.Lmul_e3:
    mulxq   24(%rsi), %rax, %r10
    movq    %rax, 24(%rdi)
    jmp     .Lmul_b2

.Lmul_e2:
    mulxq   32(%rsi), %rax, %r10
    movq    %rax, 32(%rdi)
    jmp     .Lmul_b1

.Lmul_e1:
    mulxq   40(%rsi), %rax, %r10
    movq    %rax, 40(%rdi)
    jmp     .Lmul_done

/*
 * Store row continuation.  r10 is the previous product high limb.  ADCX adds
 * that high limb plus CF into the next product low limb.
 */
.Lmul_b5:
    mulxq   8(%rsi), %rax, %r9
    adcxq   %r10, %rax
    movq    %rax, 8(%rdi)
    movq    %r9, %r10
.Lmul_b4:
    mulxq   16(%rsi), %rax, %r9
    adcxq   %r10, %rax
    movq    %rax, 16(%rdi)
    movq    %r9, %r10
.Lmul_b3:
    mulxq   24(%rsi), %rax, %r9
    adcxq   %r10, %rax
    movq    %rax, 24(%rdi)
    movq    %r9, %r10
.Lmul_b2:
    mulxq   32(%rsi), %rax, %r9
    adcxq   %r10, %rax
    movq    %rax, 32(%rdi)
    movq    %r9, %r10
.Lmul_b1:
    mulxq   40(%rsi), %rax, %r9
    adcxq   %r10, %rax
    movq    %rax, 40(%rdi)
    movq    %r9, %r10
.Lmul_done:
    adcxq   %r11, %r10
    movq    %r10, %r13

    /*
     * If bp had only one limb, store the pending high and finish.  Otherwise,
     * advance the row window by one output limb and enter the addmul loop.
     */
    decq    %r8
    jz      .Lstore_pending_done
    leaq    8(%rcx), %rcx
    leaq    8(%rdi), %rdi

/* ------------------------------------------------------------------------- */
/* Outer loop: add shifted rows ap * bp[j] into the existing result.          */
/* ------------------------------------------------------------------------- */

.Louter:
    movq    (%rcx), %rdx
    xorl    %r11d, %r11d
    jmp     *%r12

/* ------------------------------------------------------------------------- */
/* Addmul rows: rp += ap * bp[j].                                             */
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
    adoxq   %r13, %rax
    movq    %rax, 40(%rdi)
    jmp     .Ladd_done

/*
 * Addmul continuation.  r10 is previous product high.  r9 receives the current
 * product high.  CF and OF are independent carry streams.
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
    adoxq   %r13, %rax
    movq    %rax, 40(%rdi)
    movq    %r9, %r10
.Ladd_done:
    /*
     * Final row carry is current product high plus both carry chains.  Keep it
     * pending.  If another row follows, it will be consumed as that row's last
     * overlapping limb; otherwise it is the final product high limb.
     */
    adoxq   %r11, %r10
    adcxq   %r11, %r10
    movq    %r10, %r13

    decq    %r8
    jz      .Lstore_pending_done
    leaq    8(%rcx), %rcx
    leaq    8(%rdi), %rdi
    jmp     .Louter

.Lstore_pending_done:
    movq    %r13, 48(%rdi)

.Ldone:
    popq    %r13
    popq    %r12
    ret

/* ------------------------------------------------------------------------- */
/* PC-relative entry tables indexed by an.  Entry 0 is invalid and unused.   */
/* ------------------------------------------------------------------------- */

    .p2align 2
.Lmul_tab:
    .long   .Ldone   - .Lmul_tab
    .long   .Lmul_e1 - .Lmul_tab
    .long   .Lmul_e2 - .Lmul_tab
    .long   .Lmul_e3 - .Lmul_tab
    .long   .Lmul_e4 - .Lmul_tab
    .long   .Lmul_e5 - .Lmul_tab
    .long   .Lmul_e6 - .Lmul_tab

    .p2align 2
.Ladd_tab:
    .long   .Ldone   - .Ladd_tab
    .long   .Ladd_e1 - .Ladd_tab
    .long   .Ladd_e2 - .Ladd_tab
    .long   .Ladd_e3 - .Ladd_tab
    .long   .Ladd_e4 - .Ladd_tab
    .long   .Ladd_e5 - .Ladd_tab
    .long   .Ladd_e6 - .Ladd_tab

    .size simd_mpn_mul_basecase_le6, .-simd_mpn_mul_basecase_le6

    .section .note.GNU-stack,"",@progbits
