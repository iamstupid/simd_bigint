	.text
	.globl	recip_step_8_16
	.type	recip_step_8_16,@function
recip_step_8_16:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$320, %rsp
	xor	%ebp, %ebp
	mov	%rdx, 192(%rsp)
	mov	0(%rdi), %rax
	mov	%rax, 0(%rsp)
	mov	8(%rdi), %rax
	mov	%rax, 8(%rsp)
	mov	16(%rdi), %rax
	mov	%rax, 16(%rsp)
	mov	24(%rdi), %rax
	mov	%rax, 24(%rsp)
	mov	32(%rdi), %rax
	mov	%rax, 32(%rsp)
	mov	40(%rdi), %rax
	mov	%rax, 40(%rsp)
	mov	48(%rdi), %rax
	mov	%rax, 48(%rsp)
	mov	56(%rdi), %rax
	mov	%rax, 56(%rsp)
	mov	0(%rsi), %rax
	mov	%rax, 64(%rsp)
	mov	8(%rsi), %rax
	mov	%rax, 72(%rsp)
	mov	16(%rsi), %rax
	mov	%rax, 80(%rsp)
	mov	24(%rsi), %rax
	mov	%rax, 88(%rsp)
	mov	32(%rsi), %rax
	mov	%rax, 96(%rsp)
	mov	40(%rsi), %rax
	mov	%rax, 104(%rsp)
	mov	48(%rsi), %rax
	mov	%rax, 112(%rsp)
	mov	56(%rsi), %rax
	mov	%rax, 120(%rsp)
	mov	64(%rsi), %rax
	mov	%rax, 128(%rsp)
	mov	72(%rsi), %rax
	mov	%rax, 136(%rsp)
	mov	80(%rsi), %rax
	mov	%rax, 144(%rsp)
	mov	88(%rsi), %rax
	mov	%rax, 152(%rsp)
	mov	96(%rsi), %rax
	mov	%rax, 160(%rsp)
	mov	104(%rsi), %rax
	mov	%rax, 168(%rsp)
	mov	112(%rsi), %rax
	mov	%rax, 176(%rsp)
	mov	120(%rsi), %rax
	mov	%rax, 184(%rsp)
	# ===== parallelogram W = D*x_hi window [7,16]+guard17 -> a[0..10] =====
	mov	0(%rsp), %rdx	# x_hi[0]
	mulx	120(%rsp), %r8, %r9
	mulx	128(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	136(%rsp), %rax, %r11
	adc	%rax, %r10
	mulx	144(%rsp), %rax, %r12
	adc	%rax, %r11
	mulx	152(%rsp), %rax, %r13
	adc	%rax, %r12
	mulx	160(%rsp), %rax, %r14
	adc	%rax, %r13
	mulx	168(%rsp), %rax, %r15
	adc	%rax, %r14
	mulx	176(%rsp), %rax, %rcx
	adc	%rax, %r15
	mulx	184(%rsp), %rax, %rdi
	adc	%rax, %rcx
	adc	$0, %rdi
	mov	8(%rsp), %rdx	# x_hi[1]
	xor	%esi, %esi
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	152(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	160(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	168(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	176(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	184(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	16(%rsp), %rdx	# x_hi[2]
	xor	%eax, %eax
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	152(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	160(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	168(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	176(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	24(%rsp), %rdx	# x_hi[3]
	xor	%eax, %eax
	mulx	96(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	152(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	160(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	168(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	32(%rsp), %rdx	# x_hi[4]
	xor	%eax, %eax
	mulx	88(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	96(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	104(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	112(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	120(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	152(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	160(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	40(%rsp), %rdx	# x_hi[5]
	xor	%eax, %eax
	mulx	80(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	88(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	96(%rsp), %rax, %rbx
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
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	152(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	48(%rsp), %rdx	# x_hi[6]
	xor	%eax, %eax
	mulx	72(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	80(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	88(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	96(%rsp), %rax, %rbx
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
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	144(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	mov	56(%rsp), %rdx	# x_hi[7]
	xor	%eax, %eax
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
	mulx	96(%rsp), %rax, %rbx
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
	mulx	128(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	mulx	136(%rsp), %rax, %rbx
	adcx	%rax, %rdi
	adox	%rbx, %rsi
	adcx	%rbp, %rsi
	# W = a[0..9] (positions 7..16); guard a[10]=%rsi discarded. V=2W.
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
	# m=sign-extend(V[9]=%rdi top); |M|[1..9]=a[1..9]^m (drop |M|[0]=a[0]); spill |M| to MR
	mov	%rdi, %rax
	sar	$63, %rax
	mov	%rax, %rbx
	not	%rbx
	mov	%rbx, 200(%rsp)
	xor	%rax, %r9
	xor	%rax, %r10
	xor	%rax, %r11
	xor	%rax, %r12
	xor	%rax, %r13
	xor	%rax, %r14
	xor	%rax, %r15
	xor	%rax, %rcx
	xor	%rax, %rdi
	mov	%r9, 208(%rsp)
	mov	%r10, 216(%rsp)
	mov	%r11, 224(%rsp)
	mov	%r12, 232(%rsp)
	mov	%r13, 240(%rsp)
	mov	%r14, 248(%rsp)
	mov	%r15, 256(%rsp)
	mov	%rcx, 264(%rsp)
	mov	%rdi, 272(%rsp)
	# ===== Delta HP = x_hi*|M|[1..9], keep [8,17] -> AH[0..9]; Delta=AH[1..9] =====
	mov	0(%rsp), %rdx	# x_hi[0]
	mulx	264(%rsp), %r8, %r9
	mulx	272(%rsp), %rax, %r10
	add	%rax, %r9
	adc	$0, %r10
	mov	8(%rsp), %rdx	# x_hi[1]
	xor	%r11d, %r11d
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	adcx	%rbp, %r11
	mov	16(%rsp), %rdx	# x_hi[2]
	xor	%r12d, %r12d
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
	adcx	%rbp, %r12
	mov	24(%rsp), %rdx	# x_hi[3]
	xor	%r13d, %r13d
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
	adcx	%rbp, %r13
	mov	32(%rsp), %rdx	# x_hi[4]
	xor	%r14d, %r14d
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
	adcx	%rbp, %r14
	mov	40(%rsp), %rdx	# x_hi[5]
	xor	%r15d, %r15d
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
	adcx	%rbp, %r15
	mov	48(%rsp), %rdx	# x_hi[6]
	xor	%ecx, %ecx
	mulx	216(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	224(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	232(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	240(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	adcx	%rbp, %rcx
	mov	56(%rsp), %rdx	# x_hi[7]
	xor	%edi, %edi
	mulx	208(%rsp), %rax, %rbx
	adcx	%rax, %r8
	adox	%rbx, %r9
	mulx	216(%rsp), %rax, %rbx
	adcx	%rax, %r9
	adox	%rbx, %r10
	mulx	224(%rsp), %rax, %rbx
	adcx	%rax, %r10
	adox	%rbx, %r11
	mulx	232(%rsp), %rax, %rbx
	adcx	%rax, %r11
	adox	%rbx, %r12
	mulx	240(%rsp), %rax, %rbx
	adcx	%rax, %r12
	adox	%rbx, %r13
	mulx	248(%rsp), %rax, %rbx
	adcx	%rax, %r13
	adox	%rbx, %r14
	mulx	256(%rsp), %rax, %rbx
	adcx	%rax, %r14
	adox	%rbx, %r15
	mulx	264(%rsp), %rax, %rbx
	adcx	%rax, %r15
	adox	%rbx, %rcx
	mulx	272(%rsp), %rax, %rbx
	adcx	%rax, %rcx
	adox	%rbx, %rdi
	adcx	%rbp, %rdi
	# Delta = AH[1..9] (positions [9,17]); AH[0] discarded. out = x_hi*B^8 +/- Delta.
	mov	192(%rsp), %rsi
	mov	200(%rsp), %rax
	test	%rax, %rax
	jnz	.Lsub816
	mov	%r9, 0(%rsi)
	mov	%r10, 8(%rsi)
	mov	%r11, 16(%rsi)
	mov	%r12, 24(%rsi)
	mov	%r13, 32(%rsi)
	mov	%r14, 40(%rsi)
	mov	%r15, 48(%rsi)
	mov	%rcx, 56(%rsi)
	mov	0(%rsp), %rax
	add	%rdi, %rax
	mov	%rax, 64(%rsi)
	mov	8(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 72(%rsi)
	mov	16(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 80(%rsi)
	mov	24(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 88(%rsi)
	mov	32(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 96(%rsi)
	mov	40(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 104(%rsi)
	mov	48(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 112(%rsi)
	mov	56(%rsp), %rax
	adc	$0, %rax
	mov	%rax, 120(%rsi)
	jmp	.Ldone816
.Lsub816:
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
	mov	0(%rsp), %rax
	sbb	%rdi, %rax
	mov	%rax, 64(%rsi)
	mov	8(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 72(%rsi)
	mov	16(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 80(%rsi)
	mov	24(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 88(%rsi)
	mov	32(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 96(%rsi)
	mov	40(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 104(%rsi)
	mov	48(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 112(%rsi)
	mov	56(%rsp), %rax
	sbb	$0, %rax
	mov	%rax, 120(%rsi)
.Ldone816:
	add	$320, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_step_8_16, .-recip_step_8_16
	.section	.note.GNU-stack,"",@progbits
