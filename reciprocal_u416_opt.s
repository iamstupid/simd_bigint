	.text
# void recip_mono_opt(const u64* D (rdi, 7 limbs, normalized), u64* xout (rsi, 7 limbs))
# Negate-form Newton, tapered 2->4->7, all inlined.
# Same arithmetic as recip_mono, but triangular products are accumulated short->long
# (3->4 and 4->5->6->7 rows) to reduce high-limb carry drains.
	.globl	recip_mono_opt
	.type	recip_mono_opt,@function
recip_mono_opt:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$128, %rsp
	xor	%ebp, %ebp			# permanent zero register for adcx/adox drains

	# ===== seed: x[6] = (2^127 - 1) / D[6] =====
	mov	$0x7FFFFFFFFFFFFFFF, %rdx
	mov	$0xFFFFFFFFFFFFFFFF, %rax
	divq	48(%rdi)
	mov	%rax, 48(%rsp)

	# ===== step1: N=2,pw=1. x_hi=x[6], D=D[5..6] -> x[5..6] =====
	# mul1: x[6]*D[5..6] -> out[0..2]
	mov	48(%rsp), %rdx
	mulx	40(%rdi), %r8,  %r9
	mulx	48(%rdi), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	not	%r9
	not	%r10
	# mul2: x[6]*t[0..1] -> out2[0..2]
	mov	48(%rsp), %rdx
	mulx	%r9,  %r11, %r12
	mulx	%r10, %rax, %r13
	add	%rax, %r12
	adc	$0,   %r13
	shld	$1, %r12, %r13
	shld	$1, %r11, %r12
	mov	%r12, 40(%rsp)
	mov	%r13, 48(%rsp)

	# ===== step2: N=4,pw=2. x_hi=x[5..6], D=D[3..6] -> x[3..6] =====
	# mul1, reversed: first x[5]*D[4..6], then add x[6]*D[3..6]
	mov	40(%rsp), %rdx			# x[5], row length 3
	mulx	32(%rdi), %r8,  %r9
	mulx	40(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	48(%rdi), %rax, %r11
	adc	%rax, %r10
	adc	$0,   %r11
	mov	48(%rsp), %rdx			# x[6], row length 4
	xor	%r12d, %r12d			# new top limb; clears CF/OF
	mulx	24(%rdi), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	32(%rdi), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	40(%rdi), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	48(%rdi), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	adcx	%rbp, %r12			# discard carry beyond out[4]

	# t[0..3]=~out[1..4]
	not	%r9
	not	%r10
	not	%r11
	not	%r12
	mov	%r9,  56(%rsp)
	mov	%r10, 64(%rsp)
	mov	%r11, 72(%rsp)
	mov	%r12, 80(%rsp)

	# mul2, reversed: first x[5]*t[1..3], then add x[6]*t[0..3]
	mov	40(%rsp), %rdx			# x[5], row length 3
	mulx	64(%rsp), %r8,  %r9
	mulx	72(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	80(%rsp), %rax, %r11
	adc	%rax, %r10
	adc	$0,   %r11
	mov	48(%rsp), %rdx			# x[6], row length 4
	xor	%r12d, %r12d			# new top limb; clears CF/OF
	mulx	56(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	64(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	72(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	80(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	adcx	%rbp, %r12			# discard carry beyond out2[4]

	shld	$1, %r11, %r12			# x[6]
	shld	$1, %r10, %r11			# x[5]
	shld	$1, %r9,  %r10			# x[4]
	shld	$1, %r8,  %r9			# x[3]
	mov	%r9,  24(%rsp)
	mov	%r10, 32(%rsp)
	mov	%r11, 40(%rsp)
	mov	%r12, 48(%rsp)

	# ===== step3: N=7,pw=4. x_hi=x[3..6], D=D[0..6] -> x[0..6] =====
	# mul1, reversed rows: x[3]*D[3..6], x[4]*D[2..6], x[5]*D[1..6], x[6]*D[0..6]
	mov	24(%rsp), %rdx			# x[3], row length 4
	mulx	24(%rdi), %r8,  %r9
	mulx	32(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	40(%rdi), %rax, %r11
	adc	%rax, %r10
	mulx	48(%rdi), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12

	mov	32(%rsp), %rdx			# x[4], row length 5
	xor	%r13d, %r13d			# new top col5
	xor	%r14d, %r14d			# future carry col6; also clears CF/OF
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
	adcx	%rbp, %r13

	mov	40(%rsp), %rdx			# x[5], row length 6
	xor	%r15d, %r15d			# future carry col7; clears CF/OF
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

	mov	48(%rsp), %rdx			# x[6], row length 7
	xor	%ebp, %ebp			# clear CF/OF; keep zero register zero
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
	adcx	%rbp, %r15			# discard carry beyond out[7]

	# t[0..6]=~out[1..7]
	not	%r9
	mov	%r9,  56(%rsp)
	not	%r10
	mov	%r10, 64(%rsp)
	not	%r11
	mov	%r11, 72(%rsp)
	not	%r12
	mov	%r12, 80(%rsp)
	not	%r13
	mov	%r13, 88(%rsp)
	not	%r14
	mov	%r14, 96(%rsp)
	not	%r15
	mov	%r15, 104(%rsp)

	# mul2, reversed rows: x[3]*t[3..6], x[4]*t[2..6], x[5]*t[1..6], x[6]*t[0..6]
	mov	24(%rsp), %rdx			# x[3], row length 4
	mulx	80(%rsp),  %r8,  %r9
	mulx	88(%rsp),  %rax, %r10
	add	%rax, %r9
	mulx	96(%rsp),  %rax, %r11
	adc	%rax, %r10
	mulx	104(%rsp), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12

	mov	32(%rsp), %rdx			# x[4], row length 5
	xor	%r13d, %r13d			# new top col5
	xor	%r14d, %r14d			# future carry col6; also clears CF/OF
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
	adcx	%rbp, %r13

	mov	40(%rsp), %rdx			# x[5], row length 6
	xor	%r15d, %r15d			# future carry col7; clears CF/OF
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
	adcx	%rbp, %r14

	mov	48(%rsp), %rdx			# x[6], row length 7
	xor	%ebp, %ebp			# clear CF/OF; keep zero register zero
	mulx	56(%rsp),  %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	64(%rsp),  %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	72(%rsp),  %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	80(%rsp),  %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	88(%rsp),  %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	96(%rsp),  %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	adcx	%rbp, %r15			# discard carry beyond out2[7]

	# x[0..6] = shld over out2[0..7]
	shld	$1, %r14, %r15
	shld	$1, %r13, %r14
	shld	$1, %r12, %r13
	shld	$1, %r11, %r12
	shld	$1, %r10, %r11
	shld	$1, %r9,  %r10
	shld	$1, %r8,  %r9
	mov	%r9,  0(%rsi)
	mov	%r10, 8(%rsi)
	mov	%r11, 16(%rsi)
	mov	%r12, 24(%rsi)
	mov	%r13, 32(%rsi)
	mov	%r14, 40(%rsi)
	mov	%r15, 48(%rsi)

	add	$128, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_mono_opt, .-recip_mono_opt
	.section	.note.GNU-stack,"",@progbits
