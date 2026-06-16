	.text
# void recip_mono(const u64* D (rdi, 7 limbs, normalized), u64* xout (rsi, 7 limbs))
# Negate-form Newton, tapered 2->4->7, all inlined. x kept in regs/stack; only
# step3's t spills once. 62 mulx total + seed div.
#
# stack (after 6 callee pushes + sub 128):
#   x[k]  at  8*k (%rsp),  k=0..6   ->  0,8,16,24,32,40,48
#   t[k]  at  56+8*k(%rsp),k=0..6   ->  56..104
	.globl	recip_mono
	.type	recip_mono,@function
recip_mono:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$128, %rsp

	# ===== seed: x[6] = (2^127 - 1) / D[6]  (no #DE since quotient < 2^64) =====
	mov	$0x7FFFFFFFFFFFFFFF, %rdx
	mov	$0xFFFFFFFFFFFFFFFF, %rax
	divq	48(%rdi)			# rax = quotient = x[6]
	mov	%rax, 48(%rsp)

	# ===== step1: N=2,pw=1.  x_hi=x[6], D=D[5..6] -> x[5..6] =====
	# mul1: x[6]*D[5..6] -> out[0..2]
	mov	48(%rsp), %rdx
	mulx	40(%rdi), %r8,  %r9		# *D[5]
	mulx	48(%rdi), %rax, %r10		# *D[6]
	add	%rax, %r9
	adc	$0,   %r10
	# t[0]=~out[1]=~r9, t[1]=~out[2]=~r10
	not	%r9
	not	%r10
	# mul2: x[6]*t[0..1] -> out2[0..2]
	mov	48(%rsp), %rdx
	mulx	%r9,  %r11, %r12		# *t[0]: out2[0]=r11, col2=r12
	mulx	%r10, %rax, %r13		# *t[1]: col2=rax, out2[2]=r13
	add	%rax, %r12			# out2[1]=r12
	adc	$0,   %r13			# out2[2]=r13
	# x[6]=shld(out2[2],out2[1]); x[5]=shld(out2[1],out2[0])
	shld	$1, %r12, %r13			# x[6]
	shld	$1, %r11, %r12			# x[5]
	mov	%r12, 40(%rsp)
	mov	%r13, 48(%rsp)

	# ===== step2: N=4,pw=2. x_hi=x[5..6], D=D[3..6] -> x[3..6] =====
	# mul1: x2(x[5],x[6]) * D4(D[3..6]) -> out[0..4]
	mov	48(%rsp), %rdx			# x[6] (row a=1, full)
	mulx	24(%rdi), %r8,  %r9
	mulx	32(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	40(%rdi), %rax, %r11
	adc	%rax, %r10
	mulx	48(%rdi), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12
	mov	40(%rsp), %rdx			# x[5] (row a=0, j=1..3)
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
	adcx	%rbp, %r11
	adox	%rbp, %r12
	adcx	%rbp, %r12
	# t[0..3]=~out[1..4]=~r9,~r10,~r11,~r12 ; spill to t[0..3]
	not	%r9
	not	%r10
	not	%r11
	not	%r12
	mov	%r9,  56(%rsp)
	mov	%r10, 64(%rsp)
	mov	%r11, 72(%rsp)
	mov	%r12, 80(%rsp)
	# mul2: x2(x[5],x[6]) * t[0..3] -> out2[0..4]
	mov	48(%rsp), %rdx			# x[6] full
	mulx	56(%rsp), %r8,  %r9
	mulx	64(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	72(%rsp), %rax, %r11
	adc	%rax, %r10
	mulx	80(%rsp), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12
	mov	40(%rsp), %rdx			# x[5] j=1..3
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
	adcx	%rbp, %r11
	adox	%rbp, %r12
	adcx	%rbp, %r12
	# x[3..6] = shld over out2[0..4] (r8..r12), high to low
	shld	$1, %r11, %r12			# x[6]
	shld	$1, %r10, %r11			# x[5]
	shld	$1, %r9,  %r10			# x[4]
	shld	$1, %r8,  %r9			# x[3]
	mov	%r9,  24(%rsp)
	mov	%r10, 32(%rsp)
	mov	%r11, 40(%rsp)
	mov	%r12, 48(%rsp)

	# ===== step3: N=7,pw=4. x_hi=x[3..6], D=D[0..6] -> x[0..6] =====
	# mul1: x4(x[3..6]) * D7(D[0..6]) -> out[0..7]
	mov	48(%rsp), %rdx			# x[6] row a=3 full
	mulx	0(%rdi),  %r8,  %r9
	mulx	8(%rdi),  %rax, %r10
	add	%rax, %r9
	mulx	16(%rdi), %rax, %r11
	adc	%rax, %r10
	mulx	24(%rdi), %rax, %r12
	adc	%rax, %r11
	mulx	32(%rdi), %rax, %r13
	adc	%rax, %r12
	mulx	40(%rdi), %rax, %r14
	adc	%rax, %r13
	mulx	48(%rdi), %rax, %r15
	adc	%rax, %r14
	adc	$0,   %r15
	mov	40(%rsp), %rdx			# x[5] row a=2 j=1..6
	xor	%ebp, %ebp
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
	adox	%rbp, %r15
	adcx	%rbp, %r15
	mov	32(%rsp), %rdx			# x[4] row a=1 j=2..6
	xor	%ebp, %ebp
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
	adox	%rbp, %r14
	adcx	%rbp, %r14
	adox	%rbp, %r15
	adcx	%rbp, %r15
	mov	24(%rsp), %rdx			# x[3] row a=0 j=3..6
	xor	%ebp, %ebp
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
	adcx	%rbp, %r12
	adox	%rbp, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r14
	adcx	%rbp, %r14
	adox	%rbp, %r15
	adcx	%rbp, %r15
	# t[0..6]=~out[1..7]=~r9..~r15 ; spill
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
	# mul2: x4(x[3..6]) * t(t[0..6]) -> out2[0..7]
	mov	48(%rsp), %rdx			# x[6] row a=3 full
	mulx	56(%rsp),  %r8,  %r9
	mulx	64(%rsp),  %rax, %r10
	add	%rax, %r9
	mulx	72(%rsp),  %rax, %r11
	adc	%rax, %r10
	mulx	80(%rsp),  %rax, %r12
	adc	%rax, %r11
	mulx	88(%rsp),  %rax, %r13
	adc	%rax, %r12
	mulx	96(%rsp),  %rax, %r14
	adc	%rax, %r13
	mulx	104(%rsp), %rax, %r15
	adc	%rax, %r14
	adc	$0,   %r15
	mov	40(%rsp), %rdx			# x[5] row a=2 j=1..6
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
	adcx	%rbp, %r14
	adox	%rbp, %r15
	adcx	%rbp, %r15
	mov	32(%rsp), %rdx			# x[4] row a=1 j=2..6
	xor	%ebp, %ebp
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
	adox	%rbp, %r14
	adcx	%rbp, %r14
	adox	%rbp, %r15
	adcx	%rbp, %r15
	mov	24(%rsp), %rdx			# x[3] row a=0 j=3..6
	xor	%ebp, %ebp
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
	adcx	%rbp, %r12
	adox	%rbp, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r14
	adcx	%rbp, %r14
	adox	%rbp, %r15
	adcx	%rbp, %r15
	# x[0..6] = shld over out2[0..7] (r8..r15), high to low
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

	add	$128, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_mono, .-recip_mono
	.section	.note.GNU-stack,"",@progbits
