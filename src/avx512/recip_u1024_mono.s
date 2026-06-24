	.text
	.globl	recip_u1024
	.type	recip_u1024,@function
recip_u1024:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$448, %rsp
	xor	%ebp, %ebp
	mov	%rsi, 352(%rsp)		# save out
	mov	0(%rdi), %rax
	mov	%rax, 224(%rsp)
	mov	8(%rdi), %rax
	mov	%rax, 232(%rsp)
	mov	16(%rdi), %rax
	mov	%rax, 240(%rsp)
	mov	24(%rdi), %rax
	mov	%rax, 248(%rsp)
	mov	32(%rdi), %rax
	mov	%rax, 256(%rsp)
	mov	40(%rdi), %rax
	mov	%rax, 264(%rsp)
	mov	48(%rdi), %rax
	mov	%rax, 272(%rsp)
	mov	56(%rdi), %rax
	mov	%rax, 280(%rsp)
	mov	64(%rdi), %rax
	mov	%rax, 288(%rsp)
	mov	72(%rdi), %rax
	mov	%rax, 296(%rsp)
	mov	80(%rdi), %rax
	mov	%rax, 304(%rsp)
	mov	88(%rdi), %rax
	mov	%rax, 312(%rsp)
	mov	96(%rdi), %rax
	mov	%rax, 320(%rsp)
	mov	104(%rdi), %rax
	mov	%rax, 328(%rsp)
	mov	112(%rdi), %rax
	mov	%rax, 336(%rsp)
	mov	120(%rdi), %rax
	mov	%rax, 344(%rsp)
	lea	288(%rsp), %rdi		# rdi = &D[8] (top half)
	# ---- inlined recip_u512_mp (writes x8 to XHI@160) ----

	# ===== seed: x[7] = (2^127-1)/D[7] =====
	mov	$0x7FFFFFFFFFFFFFFF, %rdx
	mov	$0xFFFFFFFFFFFFFFFF, %rax
	divq	56(%rdi)
	mov	%rax, 56(%rsp)
	# ===== step1: 1->3 =====
	mov	56(%rsp), %rdx
	mulx	40(%rdi), %r8,  %r9
	mulx	48(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	56(%rdi), %rax, %r11
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
	shld	$1, %r10, %r11
	shld	$1, %r9,  %r10
	shld	$1, %r8,  %r9
	mov	%r9,  40(%rsp)
	mov	%r10, 48(%rsp)
	mov	%r11, 56(%rsp)
	# ===== step2: 3->4 (opt) =====
	mov	40(%rsp), %rdx
	mulx	48(%rdi), %r8,  %r9
	mulx	56(%rdi), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	mov	48(%rsp), %rdx
	xor	%r11d, %r11d
	xor	%r12d, %r12d
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
	mov	56(%rsp), %rdx
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
	mov	40(%rsp), %rdx
	mulx	80(%rsp), %r8,  %r9
	mulx	88(%rsp), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	mov	48(%rsp), %rdx
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
	mov	56(%rsp), %rdx
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
	shld	$1, %r11, %r12
	shld	$1, %r10, %r11
	shld	$1, %r9,  %r10
	shld	$1, %r8,  %r9
	mov	%r9,  32(%rsp)		# x[4]
	mov	%r10, 40(%rsp)		# x[5]
	mov	%r11, 48(%rsp)		# x[6]
	mov	%r12, 56(%rsp)		# x[7]

	# ===== step3: 4->8 MIDDLE PRODUCT (x_hi-direct: D*Xh = 2*D*x_hi) =====
	# parallelogram W = D*x_hi window [3,8] (x_hi = x[4..7]) -> r8..r13 ; r14 guard
	mov	32(%rsp), %rdx		# x[4], D[3..7]
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
	mov	40(%rsp), %rdx		# x[5], D[2..7]
	xor	%r14d, %r14d
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
	mov	48(%rsp), %rdx		# x[6], D[1..6]
	xor	%r15d, %r15d
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
	adcx	%rbp, %r14
	mov	56(%rsp), %rdx		# x[7], D[0..5]
	xor	%r15d, %r15d
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
	adcx	%rbp, %r14
	# W = r8..r13 (limb9 r14 discarded).  V_Xh = 2*W (the x2; leading 1 = x_hi top
	# bit, doubled -> D*B^4).  top carry dropped, 0 in at bottom.
	shld	$1, %r12, %r13
	shld	$1, %r11, %r12
	shld	$1, %r10, %r11
	shld	$1, %r9,  %r10
	shld	$1, %r8,  %r9
	shl	$1, %r8
	# M = -V_Xh nearest-0 ; sign = V_Xh top bit (r13) ; |M|[1..5] KEPT IN r9..r13 (no spill)
	mov	%r13, %rax
	sar	$63,  %rax		# m = V_Xh top bit broadcast (0 or -1)
	mov	%rax, %rbx
	not	%rbx
	mov	%rbx, 144(%rsp)		# neg_mask = ~m  (-1 iff M<0)
	xor	%rax, %r9		# |M|[1..5] = V_Xh[1..5] ^ m  (|M|[0] unneeded; skip +1)
	xor	%rax, %r10
	xor	%rax, %r11
	xor	%rax, %r12
	xor	%rax, %r13

	# Delta = high5(x_hi*|M|) ; |M|[1..5]=r9..r13 (REGS, no reload) ; acc r8,r14,r15,rcx,rdi,rsi
	mov	32(%rsp), %rdx		# x[4], |M|[4..5]=r12,r13
	mulx	%r12, %r8,  %r14
	mulx	%r13, %rax, %r15
	add	%rax, %r14
	adc	$0,   %r15
	mov	40(%rsp), %rdx		# x[5], |M|[3..5]=r11,r12,r13
	xor	%ecx, %ecx
	mulx	%r11, %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r14
	mulx	%r12, %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	%r13, %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	adcx	%rbp, %rcx
	mov	48(%rsp), %rdx		# x[6], |M|[2..5]=r10..r13
	xor	%edi, %edi
	mulx	%r10, %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r14
	mulx	%r11, %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	%r12, %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	%r13, %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	adcx	%rbp, %rdi
	mov	56(%rsp), %rdx		# x[7], |M|[1..5]=r9..r13
	xor	%esi, %esi
	mulx	%r9,  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r14
	mulx	%r10, %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	%r11, %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	%r12, %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	%r13, %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	# Delta = r14,r15,rcx,rdi,rsi (limbs[5,9]; r8=limb4 discarded)
	# x_out = x_hi*B^4 +/- Delta : x_hi@[4,7], Delta@[0,4].  reload ptr(r8)+neg(r9).
	lea	160(%rsp), %r8		# -> XHI
	mov	144(%rsp), %r9		# neg_mask
	test	%r9, %r9
	jnz	.Lneg
	mov	%r14, 0(%r8)
	mov	%r15, 8(%r8)
	mov	%rcx, 16(%r8)
	mov	%rdi, 24(%r8)
	mov	32(%rsp), %rax
	add	%rsi, %rax
	mov	%rax, 32(%r8)
	mov	40(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 40(%r8)
	mov	48(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 48(%r8)
	mov	56(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 56(%r8)
	jmp	.Ldone
.Lneg:
	xor	%eax, %eax
	sub	%r14, %rax		# x_out[0] = 0 - Delta[0], CF
	mov	%rax, 0(%r8)
	mov	%rbp, %rax		# 0 (preserves CF)
	sbb	%r15, %rax
	mov	%rax, 8(%r8)
	mov	%rbp, %rax
	sbb	%rcx, %rax
	mov	%rax, 16(%r8)
	mov	%rbp, %rax
	sbb	%rdi, %rax
	mov	%rax, 24(%r8)
	mov	32(%rsp), %rax
	sbb	%rsi, %rax		# x_out[4] = x[4] - Delta[4] - b
	mov	%rax, 32(%r8)
	mov	40(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 40(%r8)
	mov	48(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 48(%r8)
	mov	56(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 56(%r8)
.Ldone:

	# ---- inlined 8->16 step (reads XHI@160, D@224) ----
	mov	160(%rsp), %rdx	# x_hi[0]
	mulx	280(%rsp), %r8, %r9
	mulx	288(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	296(%rsp), %rax, %r11
	adc	%rax, %r10
	mulx	304(%rsp), %rax, %r12
	adc	%rax, %r11
	mulx	312(%rsp), %rax, %r13
	adc	%rax, %r12
	mulx	320(%rsp), %rax, %r14
	adc	%rax, %r13
	mulx	328(%rsp), %rax, %r15
	adc	%rax, %r14
	mulx	336(%rsp), %rax, %rcx
	adc	%rax, %r15
	mulx	344(%rsp), %rax, %rdi
	adc	%rax, %rcx
	adc	$0, %rdi
	mov	168(%rsp), %rdx	# x_hi[1]
	xor	%esi, %esi
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	312(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	320(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	328(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	336(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	344(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	176(%rsp), %rdx	# x_hi[2]
	xor	%eax, %eax
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	312(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	320(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	328(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	336(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	184(%rsp), %rdx	# x_hi[3]
	xor	%eax, %eax
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	312(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	320(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	328(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	192(%rsp), %rdx	# x_hi[4]
	xor	%eax, %eax
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	312(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	320(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	200(%rsp), %rdx	# x_hi[5]
	xor	%eax, %eax
	mulx	240(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	312(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	208(%rsp), %rdx	# x_hi[6]
	xor	%eax, %eax
	mulx	232(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	240(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	304(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	216(%rsp), %rdx	# x_hi[7]
	xor	%eax, %eax
	mulx	224(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	232(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	240(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	280(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	288(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	296(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	shld	$1, %rcx, %rdi
	shld	$1, %r15, %rcx
	shld	$1, %r14, %r15
	shld	$1, %r13, %r14
	shld	$1, %r12, %r13
	shld	$1, %r11, %r12
	shld	$1, %r10, %r11
	shld	$1, %r9, %r10
	shld	$1, %r8, %r9
	shl	$1, %r8
	mov	%rdi, %rax
	sar	$63, %rax
	mov	%rax, %rbx
	not	%rbx
	mov	%rbx, 360(%rsp)
	xor	%rax, %r9
	xor	%rax, %r10
	xor	%rax, %r11
	xor	%rax, %r12
	xor	%rax, %r13
	xor	%rax, %r14
	xor	%rax, %r15
	xor	%rax, %rcx
	xor	%rax, %rdi
	mov	%r9, 368(%rsp)
	mov	%r10, 376(%rsp)
	mov	%r11, 384(%rsp)
	mov	%r12, 392(%rsp)
	mov	%r13, 400(%rsp)
	mov	%r14, 408(%rsp)
	mov	%r15, 416(%rsp)
	mov	%rcx, 424(%rsp)
	mov	%rdi, 432(%rsp)
	mov	160(%rsp), %rdx	# x_hi[0]
	mulx	424(%rsp), %r8, %r9
	mulx	432(%rsp), %rax, %r10
	add	%rax, %r9
	adc	$0, %r10
	mov	168(%rsp), %rdx	# x_hi[1]
	xor	%r11d, %r11d
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	adcx	%rbp, %r11
	mov	176(%rsp), %rdx	# x_hi[2]
	xor	%r12d, %r12d
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	adcx	%rbp, %r12
	mov	184(%rsp), %rdx	# x_hi[3]
	xor	%r13d, %r13d
	mulx	400(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	adcx	%rbp, %r13
	mov	192(%rsp), %rdx	# x_hi[4]
	xor	%r14d, %r14d
	mulx	392(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	400(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	adcx	%rbp, %r14
	mov	200(%rsp), %rdx	# x_hi[5]
	xor	%r15d, %r15d
	mulx	384(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	392(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	400(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	adcx	%rbp, %r15
	mov	208(%rsp), %rdx	# x_hi[6]
	xor	%ecx, %ecx
	mulx	376(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	384(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	392(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	400(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	adcx	%rbp, %rcx
	mov	216(%rsp), %rdx	# x_hi[7]
	xor	%edi, %edi
	mulx	368(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	376(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	384(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	392(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	400(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	408(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	416(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	424(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	432(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	adcx	%rbp, %rdi
	mov	352(%rsp), %rsi
	mov	360(%rsp), %rax
	test	%rax, %rax
	jnz	.Lsub1024
	mov	%r9, 0(%rsi)
	mov	%r10, 8(%rsi)
	mov	%r11, 16(%rsi)
	mov	%r12, 24(%rsi)
	mov	%r13, 32(%rsi)
	mov	%r14, 40(%rsi)
	mov	%r15, 48(%rsi)
	mov	%rcx, 56(%rsi)
	mov	160(%rsp), %rax
	add	%rdi, %rax
	mov	%rax, 64(%rsi)
	mov	168(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 72(%rsi)
	mov	176(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 80(%rsi)
	mov	184(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 88(%rsi)
	mov	192(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 96(%rsi)
	mov	200(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 104(%rsi)
	mov	208(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 112(%rsi)
	mov	216(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 120(%rsi)
	jmp	.Ldone1024
.Lsub1024:
	xor	%eax, %eax
	sub	%r9, %rax
	mov	%rax, 0(%rsi)
	mov	%rbp, %rax
	sbb	%r10, %rax
	mov	%rax, 8(%rsi)
	mov	%rbp, %rax
	sbb	%r11, %rax
	mov	%rax, 16(%rsi)
	mov	%rbp, %rax
	sbb	%r12, %rax
	mov	%rax, 24(%rsi)
	mov	%rbp, %rax
	sbb	%r13, %rax
	mov	%rax, 32(%rsi)
	mov	%rbp, %rax
	sbb	%r14, %rax
	mov	%rax, 40(%rsi)
	mov	%rbp, %rax
	sbb	%r15, %rax
	mov	%rax, 48(%rsi)
	mov	%rbp, %rax
	sbb	%rcx, %rax
	mov	%rax, 56(%rsi)
	mov	160(%rsp), %rax
	sbb	%rdi, %rax
	mov	%rax, 64(%rsi)
	mov	168(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 72(%rsi)
	mov	176(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 80(%rsi)
	mov	184(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 88(%rsi)
	mov	192(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 96(%rsi)
	mov	200(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 104(%rsi)
	mov	208(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 112(%rsi)
	mov	216(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 120(%rsi)
.Ldone1024:
	add	$448, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_u1024, .-recip_u1024
	.section	.note.GNU-stack,"",@progbits
