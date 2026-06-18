	.text
# __m512i recip_3by2_seed(const uint64_t* I (rdi, 13 limbs, normalized))
#
# Stripped reciprocal: inlined recip_mono on the top 448 bits of I, then the
# bit[31..446] seed load -> zmm0.  NO verify tail.  Output is the 3/2 block
# reciprocal V to within +-1 (one-sided: seed in {V-1, V}); see 3by2_recip.md
# Sec.6 for the proof.  OurDiv3by2 absorbs the +-1 in its own quotient fixup.
#
# Stack frame (after 6 pushes, rsp 16-aligned):
#   0(%rsp)  : recip_buf[7]   (recip_mono output)
#   64(%rsp) : dd[16]         (i0 lanes 0..7, i1 lanes 8..15) for broadcast
#   192(%rsp): recip_mono's own 128B scratch  -> placed ABOVE so it can't clash
# Total sub = 320.  recip_mono body uses 0(%rsp_inner)..; we give it rsp+192.
	.globl	recip_3by2_seed
	.type	recip_3by2_seed,@function
recip_3by2_seed:
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	sub	$320, %rsp
	mov	%rdi, 304(%rsp)			# save I (recip_mono clobbers all GPRs incl rbx; 304 is above its 192..303 scratch)

	# ===================== inlined recip_mono(I+6 -> recip_buf) ============
	# Operates with rdi = D = I+6, rsi = recip_buf, scratch base = rsp+192.
	# (verbatim recip_mono_opt body, with 48(%rsp)->240(%rsp) etc: +192, and
	#  output stores to 0(%rsi)..48(%rsi) with rsi=recip_buf=rsp+0.)
	lea	48(%rdi), %rdi			# D = I + 6 limbs  (rdi still = I here)
	mov	%rsp, %rsi			# recip_buf = rsp+0
	xor	%ebp, %ebp			# permanent zero reg for adcx/adox drains

	# seed: x[6] = (2^127-1)/D[6]
	mov	$0x7FFFFFFFFFFFFFFF, %rdx
	mov	$0xFFFFFFFFFFFFFFFF, %rax
	divq	48(%rdi)
	mov	%rax, 240(%rsp)

	# step1
	mov	240(%rsp), %rdx
	mulx	40(%rdi), %r8,  %r9
	mulx	48(%rdi), %rax, %r10
	add	%rax, %r9
	adc	$0,   %r10
	not	%r9
	not	%r10
	mov	240(%rsp), %rdx
	mulx	%r9,  %r11, %r12
	mulx	%r10, %rax, %r13
	add	%rax, %r12
	adc	$0,   %r13
	shld	$1, %r12, %r13
	shld	$1, %r11, %r12
	mov	%r12, 232(%rsp)
	mov	%r13, 240(%rsp)

	# step2 mul1
	mov	232(%rsp), %rdx
	mulx	32(%rdi), %r8,  %r9
	mulx	40(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	48(%rdi), %rax, %r11
	adc	%rax, %r10
	adc	$0,   %r11
	mov	240(%rsp), %rdx
	xor	%r12d, %r12d
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
	not	%r9
	not	%r10
	not	%r11
	not	%r12
	mov	%r9,  248(%rsp)
	mov	%r10, 256(%rsp)
	mov	%r11, 264(%rsp)
	mov	%r12, 272(%rsp)
	# step2 mul2
	mov	232(%rsp), %rdx
	mulx	256(%rsp), %r8,  %r9
	mulx	264(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	272(%rsp), %rax, %r11
	adc	%rax, %r10
	adc	$0,   %r11
	mov	240(%rsp), %rdx
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
	shld	$1, %r11, %r12
	shld	$1, %r10, %r11
	shld	$1, %r9,  %r10
	shld	$1, %r8,  %r9
	mov	%r9,  216(%rsp)
	mov	%r10, 224(%rsp)
	mov	%r11, 232(%rsp)
	mov	%r12, 240(%rsp)

	# step3 mul1
	mov	216(%rsp), %rdx
	mulx	24(%rdi), %r8,  %r9
	mulx	32(%rdi), %rax, %r10
	add	%rax, %r9
	mulx	40(%rdi), %rax, %r11
	adc	%rax, %r10
	mulx	48(%rdi), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12
	mov	224(%rsp), %rdx
	xor	%r13d, %r13d
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
	adcx	%rbp, %r13
	mov	232(%rsp), %rdx
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
	mov	240(%rsp), %rdx
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
	adcx	%rbp, %r15
	not	%r9
	mov	%r9,  248(%rsp)
	not	%r10
	mov	%r10, 256(%rsp)
	not	%r11
	mov	%r11, 264(%rsp)
	not	%r12
	mov	%r12, 272(%rsp)
	not	%r13
	mov	%r13, 280(%rsp)
	not	%r14
	mov	%r14, 288(%rsp)
	not	%r15
	mov	%r15, 296(%rsp)
	# step3 mul2
	mov	216(%rsp), %rdx
	mulx	272(%rsp), %r8,  %r9
	mulx	280(%rsp), %rax, %r10
	add	%rax, %r9
	mulx	288(%rsp), %rax, %r11
	adc	%rax, %r10
	mulx	296(%rsp), %rax, %r12
	adc	%rax, %r11
	adc	$0,   %r12
	mov	224(%rsp), %rdx
	xor	%r13d, %r13d
	xor	%r14d, %r14d
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
	adcx	%rbp, %r13
	mov	232(%rsp), %rdx
	xor	%r15d, %r15d
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
	adcx	%rbp, %r14
	mov	240(%rsp), %rdx
	xor	%ebp, %ebp
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
	adcx	%rbp, %r15
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
	# ===================== end inlined recip_mono =========================

	# ----- seed load: bits[31..446] of recip_buf -> seed (zmm0, return) -----
	mov	$0xFFFFFFFFFFFFF, %rax
	vpbroadcastq	%rax, %zmm19		# MASK52
	mov	$0x7f, %eax
	kmovb	%eax, %k1
	vmovdqu64	0(%rsp), %zmm21{%k1}{z}	# w = recip_buf, lane7=0
	vmovdqa64	seed_byte_perm(%rip), %zmm18
	vpermb	%zmm21, %zmm18, %zmm0		# seed = perm(w)
	vmovdqa64	seed_vshift(%rip), %zmm18
	vpsrlvq	%zmm18, %zmm0, %zmm0
	vpandq	%zmm19, %zmm0, %zmm0		# seed &= MASK52
	add	$320, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_3by2_seed, .-recip_3by2_seed

	.section	.rodata
	.align	64
seed_byte_perm:
	.byte	3,4,5,6,7,8,9,10, 10,11,12,13,14,15,16,17
	.byte	16,17,18,19,20,21,22,23, 23,24,25,26,27,28,29,30
	.byte	29,30,31,32,33,34,35,36, 36,37,38,39,40,41,42,43
	.byte	42,43,44,45,46,47,48,49, 49,50,51,52,53,54,55,56
seed_vshift:
	.quad	7,3,7,3,7,3,7,3
	.section	.note.GNU-stack,"",@progbits
