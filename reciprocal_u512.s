	.text
# void recip_u512(const u64* D (rdi, 8 limbs, normalized), u64* xout (rsi, 8 limbs))
# Negate-form Newton, schedule 1->3->4->8 (iter1 carries a guard limb -> worst-case
# precision >=503 bit, verified in the mpz model).  Optimized accumulation: triangular
# rows are summed SHORT->LONG; each new top limb is introduced by `xor` (which also
# clears CF/OF), so every adcx/adox row needs just one `adcx %rbp` drain.
# 4->8 last step uses a 9-limb accumulator (8 kept + 1 discarded low) -> rcx is the 9th.
#
# stack:  x[k] at 8*k(%rsp) k=0..7 -> 0..56 ;  t[k] at 64+8*k(%rsp) -> 64..120
	.globl	recip_u512
	.type	recip_u512,@function
recip_u512:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$128, %rsp
	xor	%ebp, %ebp			# permanent zero register for adcx/adox drains

	# ===== seed: x[7] = (2^127 - 1) / D[7] =====
	mov	$0x7FFFFFFFFFFFFFFF, %rdx
	mov	$0xFFFFFFFFFFFFFFFF, %rax
	divq	56(%rdi)
	mov	%rax, 56(%rsp)

	# ===== step1: 1->3.  x_hi=x[7], D=D[5..7] -> x[5..7] (single row) =====
	mov	56(%rsp), %rdx
	mulx	40(%rdi), %r8,  %r9		# *D[5]
	mulx	48(%rdi), %rax, %r10		# *D[6]
	add	%rax, %r9
	mulx	56(%rdi), %rax, %r11		# *D[7]
	adc	%rax, %r10
	adc	$0,   %r11
	not	%r9
	not	%r10
	not	%r11
	mov	%r9,  64(%rsp)
	mov	%r10, 72(%rsp)
	mov	%r11, 80(%rsp)
	mov	56(%rsp), %rdx
	mulx	64(%rsp), %r8,  %r9
	mulx	72(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	80(%rsp), %rax, %r11
	adc	%rax, %r10
	adc	$0,   %r11
	shld	$1, %r10, %r11			# x[7]
	shld	$1, %r9,  %r10			# x[6]
	shld	$1, %r8,  %r9			# x[5]
	mov	%r9,  40(%rsp)
	mov	%r10, 48(%rsp)
	mov	%r11, 56(%rsp)

	# ===== step2: 3->4.  x_hi=x[5..7], D=D[4..7] -> x[4..7] =====
	# mul1: rows short->long x[5](D6..7) x[6](D5..7) x[7](D4..7); acc r8(disc)..r12
	mov	40(%rsp), %rdx			# x[5], D[6..7]
	mulx	48(%rdi), %r8,  %r9
	mulx	56(%rdi), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	mov	48(%rsp), %rdx			# x[6], D[5..7]
	xor	%r11d, %r11d			# new top
	xor	%r12d, %r12d			# next top; clears CF/OF
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	56(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	adcx	%rbp, %r11
	mov	56(%rsp), %rdx			# x[7], D[4..7]
	xor	%ebp, %ebp
	mulx	32(%rdi), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	56(%rdi), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	adcx	%rbp, %r12
	not	%r9
	not	%r10
	not	%r11
	not	%r12
	mov	%r9,  64(%rsp)
	mov	%r10, 72(%rsp)
	mov	%r11, 80(%rsp)
	mov	%r12, 88(%rsp)
	# mul2: rows x[5](t2..3) x[6](t1..3) x[7](t0..3); x[4..7]=hi4 after <<1
	mov	40(%rsp), %rdx			# x[5], t[2..3]
	mulx	80(%rsp), %r8,  %r9
	mulx	88(%rsp), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	mov	48(%rsp), %rdx			# x[6], t[1..3]
	xor	%r11d, %r11d
	xor	%r12d, %r12d
	mulx	72(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	80(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	88(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	adcx	%rbp, %r11
	mov	56(%rsp), %rdx			# x[7], t[0..3]
	xor	%ebp, %ebp
	mulx	64(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	72(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	80(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	88(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	adcx	%rbp, %r12
	shld	$1, %r11, %r12			# x[7]
	shld	$1, %r10, %r11			# x[6]
	shld	$1, %r9,  %r10			# x[5]
	shld	$1, %r8,  %r9			# x[4]
	mov	%r9,  32(%rsp)
	mov	%r10, 40(%rsp)
	mov	%r11, 48(%rsp)
	mov	%r12, 56(%rsp)

	# ===== step3: 4->8.  x_hi=x[4..7], D=D[0..7] -> x[0..7] =====
	# rows short->long x[4](D3..7) x[5](D2..7) x[6](D1..7) x[7](D0..7)
	# acc r8(disc),r9..r15,rcx (8 kept)
	# --- mul1 ---
	mov	32(%rsp), %rdx			# x[4], D[3..7]  (5 mulx)
	mulx	24(%rdi), %r8,  %r9
	mulx	32(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	40(%rdi), %rax, %r11
	adc	%rax, %r10
	mulx	48(%rdi), %rax, %r12
	adc	%rax, %r11
	mulx	56(%rdi), %rax, %r13
	adc	%rax, %r12
	adc	$0,   %r13
	mov	40(%rsp), %rdx			# x[5], D[2..7]  (6 mulx)
	xor	%r14d, %r14d
	xor	%r15d, %r15d
	mulx	16(%rdi), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	24(%rdi), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	32(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	56(%rdi), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	adcx	%rbp, %r14
	mov	48(%rsp), %rdx			# x[6], D[1..7]  (7 mulx)
	xor	%ecx, %ecx
	mulx	8(%rdi),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	16(%rdi), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	24(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	32(%rdi), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	56(%rdi), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	adcx	%rbp, %r15
	mov	56(%rsp), %rdx			# x[7], D[0..7]  (8 mulx)
	xor	%ebp, %ebp
	mulx	0(%rdi),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	8(%rdi),  %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	16(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	24(%rdi), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	32(%rdi), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	56(%rdi), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	adcx	%rbp, %rcx
	not	%r9
	mov	%r9,  64(%rsp)
	not	%r10
	mov	%r10, 72(%rsp)
	not	%r11
	mov	%r11, 80(%rsp)
	not	%r12
	mov	%r12, 88(%rsp)
	not	%r13
	mov	%r13, 96(%rsp)
	not	%r14
	mov	%r14, 104(%rsp)
	not	%r15
	mov	%r15, 112(%rsp)
	not	%rcx
	mov	%rcx, 120(%rsp)
	# --- mul2 ---
	mov	32(%rsp), %rdx			# x[4], t[3..7]  (5 mulx)
	mulx	88(%rsp),  %r8,  %r9
	mulx	96(%rsp),  %rax, %r10
	add	%rax, %r9
	mulx	104(%rsp), %rax, %r11
	adc	%rax, %r10
	mulx	112(%rsp), %rax, %r12
	adc	%rax, %r11
	mulx	120(%rsp), %rax, %r13
	adc	%rax, %r12
	adc	$0,   %r13
	mov	40(%rsp), %rdx			# x[5], t[2..7]  (6 mulx)
	xor	%r14d, %r14d
	xor	%r15d, %r15d
	mulx	80(%rsp),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	88(%rsp),  %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	96(%rsp),  %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	adcx	%rbp, %r14
	mov	48(%rsp), %rdx			# x[6], t[1..7]  (7 mulx)
	xor	%ecx, %ecx
	mulx	72(%rsp),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	80(%rsp),  %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	88(%rsp),  %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	96(%rsp),  %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	adcx	%rbp, %r15
	mov	56(%rsp), %rdx			# x[7], t[0..7]  (8 mulx)
	xor	%ebp, %ebp
	mulx	64(%rsp),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	72(%rsp),  %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	80(%rsp),  %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	88(%rsp),  %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	96(%rsp),  %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	adcx	%rbp, %rcx
	shld	$1, %r15, %rcx			# x[7]
	shld	$1, %r14, %r15			# x[6]
	shld	$1, %r13, %r14			# x[5]
	shld	$1, %r12, %r13			# x[4]
	shld	$1, %r11, %r12			# x[3]
	shld	$1, %r10, %r11			# x[2]
	shld	$1, %r9,  %r10			# x[1]
	shld	$1, %r8,  %r9			# x[0]
	mov	%r9,  0(%rsi)
	mov	%r10, 8(%rsi)
	mov	%r11, 16(%rsi)
	mov	%r12, 24(%rsi)
	mov	%r13, 32(%rsi)
	mov	%r14, 40(%rsi)
	mov	%r15, 48(%rsi)
	mov	%rcx, 56(%rsi)

	add	$128, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_u512, .-recip_u512
	.section	.note.GNU-stack,"",@progbits
