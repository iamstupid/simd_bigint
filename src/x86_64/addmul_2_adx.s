// sbn_addmul_2_adx: rp[0..n) += up[0..n)*<v0,v1>.  rdi=rp rsi=up rdx=n rcx=vp.
// 4x-unrolled flag-carried dual chain. Carries live in c0(r12),c1(r13) registers;
// CF/OF only carry WITHIN a limb-body (reset to 0 at body end), so loop control
// may clobber flags as long as each group re-clears with xor.  writes rp[0..n],
// returns rp[n+1] in rax.
	.text
	.globl	sbn_addmul_2_adx
	.type	sbn_addmul_2_adx,@function
sbn_addmul_2_adx:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	mov	(%rcx), %r8		# v0
	mov	8(%rcx), %r9		# v1
	mov	%rdx, %rcx		# n
	mov	%rcx, %r14
	and	$3, %r14d		# tail = n & 3
	shr	$2, %rcx		# groups = n >> 2
	xor	%r12d, %r12d		# c0 = 0
	xor	%r13d, %r13d		# c1 = 0
	test	%rcx, %rcx
	jz	.Ltailprep
.Lmain:
	xor	%ebp, %ebp		# z=0, CF=0, OF=0
	# --- limb 0 ---
	mov	0(%rsi), %rdx
	mulx	%r8, %rax, %r10
	mulx	%r9, %rbx, %r11
	mov	0(%rdi), %r15
	adcx	%r12, %r15
	adox	%rax, %r15
	mov	%r15, 0(%rdi)
	mov	%r10, %r12
	adcx	%rbx, %r12
	adox	%r13, %r12
	mov	%r11, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r13
	# --- limb 1 ---
	mov	8(%rsi), %rdx
	mulx	%r8, %rax, %r10
	mulx	%r9, %rbx, %r11
	mov	8(%rdi), %r15
	adcx	%r12, %r15
	adox	%rax, %r15
	mov	%r15, 8(%rdi)
	mov	%r10, %r12
	adcx	%rbx, %r12
	adox	%r13, %r12
	mov	%r11, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r13
	# --- limb 2 ---
	mov	16(%rsi), %rdx
	mulx	%r8, %rax, %r10
	mulx	%r9, %rbx, %r11
	mov	16(%rdi), %r15
	adcx	%r12, %r15
	adox	%rax, %r15
	mov	%r15, 16(%rdi)
	mov	%r10, %r12
	adcx	%rbx, %r12
	adox	%r13, %r12
	mov	%r11, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r13
	# --- limb 3 ---
	mov	24(%rsi), %rdx
	mulx	%r8, %rax, %r10
	mulx	%r9, %rbx, %r11
	mov	24(%rdi), %r15
	adcx	%r12, %r15
	adox	%rax, %r15
	mov	%r15, 24(%rdi)
	mov	%r10, %r12
	adcx	%rbx, %r12
	adox	%r13, %r12
	mov	%r11, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r13
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	dec	%rcx
	jnz	.Lmain
.Ltailprep:
	test	%r14, %r14
	jz	.Lflush
.Ltail:
	xor	%ebp, %ebp
	mov	0(%rsi), %rdx
	mulx	%r8, %rax, %r10
	mulx	%r9, %rbx, %r11
	mov	0(%rdi), %r15
	adcx	%r12, %r15
	adox	%rax, %r15
	mov	%r15, 0(%rdi)
	mov	%r10, %r12
	adcx	%rbx, %r12
	adox	%r13, %r12
	mov	%r11, %r13
	adcx	%rbp, %r13
	adox	%rbp, %r13
	lea	8(%rsi), %rsi
	lea	8(%rdi), %rdi
	dec	%r14
	jnz	.Ltail
.Lflush:
	mov	%r12, 0(%rdi)
	mov	%r13, %rax
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	sbn_addmul_2_adx,.-sbn_addmul_2_adx
