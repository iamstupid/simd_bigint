	.text
# void recip_u512_mp(const u64* D (rdi,8 limbs norm), u64* xout (rsi,8 limbs))
# Schedule 1->3->4 negate-form (opt shape) then the LAST 4->8 step as a residue-class
# MIDDLE PRODUCT doubling (validated /tmp/mp48_limb.c, min 503 bit):
#   iph = x_hi<<1 ; V = coeffs[3,8] of D*iph (23-mulx parallelogram) ;
#   M = [0,(-D)[0..4]] - V (6-limb signed, neg=borrow) ; |M| ;
#   Delta = |M|>>1limb + (iph*|M|)>>5limb ; I = X0 +/- Delta (X0=2*x_hi*B^4) ; out = I>>1.
# stack: x[k]@8k(0..56) iph[k]@64+8k(64..88) |M|[k]@96+8k(96..136)
	.globl	recip_u512_mp
	.type	recip_u512_mp,@function
recip_u512_mp:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	push	%rsi			# save output ptr (need rsi/regs)
	sub	$160, %rsp
	xor	%ebp, %ebp		# permanent zero register

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
	mov	160(%rsp), %r8		# output ptr
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

	add	$160, %rsp
	pop	%rsi
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_u512_mp, .-recip_u512_mp
	.section	.note.GNU-stack,"",@progbits
