	.text
# __m512i recip_3by2(const uint64_t* I (rdi, 13 limbs, normalized 2^831<=I<2^832))
#
# Returns V = floor((2^1248-1)/I) - 2^416 as one zmm (8 u52 lanes), the 3/2
# block reciprocal.  Monolithic / fused:  recip_mono inlined -> seed load ->
# decode I -> seed*I (madd52) -> alignr diagonal reduce + canonize -> compare
# vs T=[all1,~i0,~i1] -> inc/dec.  Validated against bench/recip_3by2_ref.c.
#
# Stack frame (after 6 pushes, rsp 16-aligned):
#   0(%rsp)  : recip_buf[7]   (recip_mono output)
#   64(%rsp) : dd[16]         (i0 lanes 0..7, i1 lanes 8..15) for broadcast
#   192(%rsp): recip_mono's own 128B scratch  -> placed ABOVE so it can't clash
# Total sub = 320.  recip_mono body uses 0(%rsp_inner)..; we give it rsp+192.
	.globl	recip_3by2
	.type	recip_3by2,@function
recip_3by2:
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

	mov	304(%rsp), %rdi			# restore I

	# ----- constants -----
	mov	$0xFFFFFFFFFFFFF, %rax
	vpbroadcastq	%rax, %zmm19		# MASK52
	vpxorq	%zmm17, %zmm17, %zmm17		# Z = 0  (also dec/inc "zero" source)

	# ----- seed load: bits[31..446] of recip_buf -> 8x u52  (zmm20) -----
	mov	$0x7f, %eax
	kmovb	%eax, %k1
	vmovdqu64	0(%rsp), %zmm21{%k1}{z}	# w = recip_buf, lane7=0
	vmovdqa64	seed_byte_perm(%rip), %zmm18
	vpermb	%zmm21, %zmm18, %zmm20		# seed = perm(w)
	vmovdqa64	seed_vshift(%rip), %zmm18
	vpsrlvq	%zmm18, %zmm20, %zmm20
	vpandq	%zmm19, %zmm20, %zmm20		# seed &= MASK52

	# ----- decode I -> i0 (zmm dd[0..7]), i1 (dd[8..15]) -----
	vmovdqa64	u52_dec_perm(%rip), %zmm18	# idx
	vmovdqa64	dec_vshift(%rip), %zmm16		# {0,4,..}
	vmovdqu8	0(%rdi), %zmm21			# 64B (I has 104B; safe)
	vpermb	%zmm21, %zmm18, %zmm21
	vpsrlvq	%zmm16, %zmm21, %zmm21
	vpandq	%zmm19, %zmm21, %zmm21		# i0
	vmovdqu64	%zmm21, 64(%rsp)
	mov	$0xFFFFFFFFFFFFF, %rax		# 52-byte mask (52 bits)
	kmovq	%rax, %k2
	vmovdqu8	52(%rdi), %zmm22{%k2}{z}
	vpermb	%zmm22, %zmm18, %zmm22
	vpsrlvq	%zmm16, %zmm22, %zmm22
	vpandq	%zmm19, %zmm22, %zmm22		# i1
	vmovdqu64	%zmm22, 128(%rsp)

	# ----- multiply P = seed * D : temp[0..16] in zmm0..16 -----
	# b broadcast in zmm18; seed in zmm20.
	vpxorq	%zmm0, %zmm0, %zmm0
	.macro MUL i tl th
	vpbroadcastq	(64+8*\i)(%rsp), %zmm18
	vpmadd52luq	%zmm18, %zmm20, %zmm\tl
	vpxorq	%zmm\th, %zmm\th, %zmm\th
	vpmadd52huq	%zmm18, %zmm20, %zmm\th
	.endm
	MUL 0  0 1
	MUL 1  1 2
	MUL 2  2 3
	MUL 3  3 4
	MUL 4  4 5
	MUL 5  5 6
	MUL 6  6 7
	MUL 7  7 8
	MUL 8  8 9
	MUL 9  9 10
	MUL 10 10 11
	MUL 11 11 12
	MUL 12 12 13
	MUL 13 13 14
	MUL 14 14 15
	MUL 15 15 16

	# ----- reduction + canonize : P0=zmm29, P1=zmm30, P2=zmm31 -----
	# carry=zmm28, sig=r8d.  reduction scratch zmm21..27.
	vpxorq	%zmm28, %zmm28, %zmm28		# carry = 0
	xor	%r8d, %r8d			# sig = 0

	# --- canonize macro: vec=%1 (in/out), uses carry zmm28, sig r8d ---
	.macro CANON v
	vpsrlq	$52, %zmm\v, %zmm26		# hi
	valignq	$7, %zmm28, %zmm26, %zmm27	# his = alignr(hi, carry, 7)
	vpandq	%zmm19, %zmm\v, %zmm25		# lo
	vpaddq	%zmm27, %zmm25, %zmm25		# ct = lo + his
	vmovdqa64	%zmm26, %zmm28		# carry = hi
	vpcmpuq	$6, %zmm19, %zmm25, %k3		# g = ct > MASK
	vpcmpuq	$0, %zmm19, %zmm25, %k4		# p = ct == MASK
	kmovb	%k4, %eax			# p
	kmovb	%k3, %ecx			# g
	lea	(%rax,%rcx,2), %edx		# p + 2g
	add	%r8d, %edx			# chain = p + 2g + sig
	mov	%edx, %r8d
	shr	$8, %r8d			# sig = chain>>8
	xor	%edx, %eax			# cin = p ^ chain
	kmovb	%eax, %k5
	vpsubq	%zmm19, %zmm25, %zmm25{%k5}	# ct{cin} -= MASK
	vpandq	%zmm19, %zmm25, %zmm\v		# vec = ct & MASK
	.endm

	# block 0: temp0 + sum_{i=1..7} lsh(temp[i],i)
	valignq	$7, %zmm17, %zmm1, %zmm21
	vpaddq	%zmm0, %zmm21, %zmm21
	valignq	$6, %zmm17, %zmm2, %zmm22
	valignq	$5, %zmm17, %zmm3, %zmm23
	vpaddq	%zmm23, %zmm22, %zmm22
	valignq	$4, %zmm17, %zmm4, %zmm23
	valignq	$3, %zmm17, %zmm5, %zmm24
	vpaddq	%zmm24, %zmm23, %zmm23
	valignq	$2, %zmm17, %zmm6, %zmm24
	valignq	$1, %zmm17, %zmm7, %zmm25
	vpaddq	%zmm25, %zmm24, %zmm24
	vpaddq	%zmm22, %zmm21, %zmm21
	vpaddq	%zmm24, %zmm23, %zmm23
	vpaddq	%zmm23, %zmm21, %zmm29
	CANON	29

	# block 1: temp8 + alignr(temp[i+8],temp[i],8-i)
	valignq	$7, %zmm1, %zmm9, %zmm21
	vpaddq	%zmm8, %zmm21, %zmm21
	valignq	$6, %zmm2, %zmm10, %zmm22
	valignq	$5, %zmm3, %zmm11, %zmm23
	vpaddq	%zmm23, %zmm22, %zmm22
	valignq	$4, %zmm4, %zmm12, %zmm23
	valignq	$3, %zmm5, %zmm13, %zmm24
	vpaddq	%zmm24, %zmm23, %zmm23
	valignq	$2, %zmm6, %zmm14, %zmm24
	valignq	$1, %zmm7, %zmm15, %zmm25
	vpaddq	%zmm25, %zmm24, %zmm24
	vpaddq	%zmm22, %zmm21, %zmm21
	vpaddq	%zmm24, %zmm23, %zmm23
	vpaddq	%zmm23, %zmm21, %zmm30
	CANON	30

	# block 2: temp16 + alignr(Z,temp[i+8],8-i)
	valignq	$7, %zmm9, %zmm17, %zmm21
	vpaddq	%zmm16, %zmm21, %zmm21
	valignq	$6, %zmm10, %zmm17, %zmm22
	valignq	$5, %zmm11, %zmm17, %zmm23
	vpaddq	%zmm23, %zmm22, %zmm22
	valignq	$4, %zmm12, %zmm17, %zmm23
	valignq	$3, %zmm13, %zmm17, %zmm24
	vpaddq	%zmm24, %zmm23, %zmm23
	valignq	$2, %zmm14, %zmm17, %zmm24
	valignq	$1, %zmm15, %zmm17, %zmm25
	vpaddq	%zmm25, %zmm24, %zmm24
	vpaddq	%zmm22, %zmm21, %zmm21
	vpaddq	%zmm24, %zmm23, %zmm23
	vpaddq	%zmm23, %zmm21, %zmm31
	CANON	31

	# ----- T high blocks: c1=~i0 (zmm2), c2=~i1 (zmm3) -----
	vmovdqu64	64(%rsp), %zmm0		# i0
	vmovdqu64	128(%rsp), %zmm1		# i1
	vpxorq	%zmm19, %zmm0, %zmm2		# c1 = ~i0
	vpxorq	%zmm19, %zmm1, %zmm3		# c2 = ~i1

	# gt2(A0,A1 ; B0,B1): macro -> sets ZF such that JA == (A>B). Returns in CF/ZF
	# via final cmp; caller uses 'ja'. Implemented with explicit compares.
	# DEC test:  [P1:P2] > [c1:c2] ?
	# A1=P2(zmm31) A0=P1(zmm30) ; B1=c2(zmm3) B0=c1(zmm2)
	vpcmpuq	$4, %zmm3, %zmm31, %k3		# ne_hi
	kmovb	%k3, %eax
	test	%al, %al
	jnz	1f
	vpcmpuq	$4, %zmm2, %zmm30, %k3		# ne_lo
	kmovb	%k3, %eax
	test	%al, %al
	jz	2f				# fully equal -> not >
	vpcmpuq	$1, %zmm2, %zmm30, %k4		# lt_lo
	kmovb	%k4, %ecx
	mov	%eax, %edx
	xor	%ecx, %edx			# gt
	cmp	%ecx, %edx
	ja	3f				# P>T -> dec
	jmp	2f
1:	vpcmpuq	$1, %zmm3, %zmm31, %k4		# lt_hi
	kmovb	%k4, %ecx
	mov	%eax, %edx
	xor	%ecx, %edx
	cmp	%ecx, %edx
	ja	3f
2:	# not greater -> try INC
	# Q = P + I  (carry_prop), carry in r9d; only need Q1,Q2 (zmm4,zmm5)
	xor	%r9d, %r9d
	# carry_prop dst, P, addend  (res scratch zmm7)
	.macro CPROP dst pp add
	vpaddq	%zmm\add, %zmm\pp, %zmm7
	vpcmpuq	$0, %zmm19, %zmm7, %k3		# prop
	vpcmpuq	$6, %zmm19, %zmm7, %k4		# carr
	kmovw	%k3, %eax			# prop
	kmovw	%k4, %ecx			# carr
	shl	$1, %ecx
	or	%ecx, %r9d			# carry |= carr<<1
	add	%eax, %r9d			# carry += prop
	xor	%r9d, %eax			# eax = prop ^ carry = cin
	shr	$8, %r9d			# carry >>= 8
	kmovb	%eax, %k5
	vpsubq	%zmm19, %zmm7, %zmm7{%k5}
	vpandq	%zmm19, %zmm7, %zmm\dst
	.endm
	CPROP	6  29 0			# Q0 = P0 + i0  (scratch zmm6, discarded)
	CPROP	4  30 1			# Q1 = P1 + i1
	CPROP	5  31 17		# Q2 = P2 + 0
	# if [Q1:Q2] > [c1:c2] -> exact, keep seed; else INC
	vpcmpuq	$4, %zmm3, %zmm5, %k3		# ne_hi
	kmovb	%k3, %eax
	test	%al, %al
	jnz	4f
	vpcmpuq	$4, %zmm2, %zmm4, %k3
	kmovb	%k3, %eax
	test	%al, %al
	jz	5f				# Q==T -> not > -> inc
	vpcmpuq	$1, %zmm2, %zmm4, %k4
	kmovb	%k4, %ecx
	mov	%eax, %edx
	xor	%ecx, %edx
	cmp	%ecx, %edx
	ja	6f				# Q>T -> keep seed
	jmp	5f
4:	vpcmpuq	$1, %zmm3, %zmm5, %k4
	kmovb	%k4, %ecx
	mov	%eax, %edx
	xor	%ecx, %edx
	cmp	%ecx, %edx
	ja	6f
5:	# INC seed
	vpcmpuq	$0, %zmm19, %zmm20, %k3		# p = seed==MASK
	kmovb	%k3, %eax
	lea	1(%rax), %ecx			# chain
	mov	%eax, %edx
	xor	%ecx, %edx			# flip
	mov	%edx, %esi
	and	%eax, %esi			# flip & p  -> zero lanes
	not	%eax
	and	%eax, %edx			# flip & ~p -> +1 lane
	kmovb	%edx, %k4
	kmovb	%esi, %k5
	mov	$1, %eax
	vpbroadcastq	%rax, %zmm6
	vpaddq	%zmm6, %zmm20, %zmm20{%k4}
	vmovdqa64	%zmm17, %zmm20{%k5}
	jmp	6f
3:	# DEC seed
	vpcmpuq	$0, %zmm17, %zmm20, %k3		# z = seed==0
	kmovb	%k3, %eax
	lea	1(%rax), %ecx
	mov	%eax, %edx
	xor	%ecx, %edx			# flip
	mov	%edx, %esi
	and	%eax, %esi			# flip & z -> set-MASK lanes
	not	%eax
	and	%eax, %edx			# flip & ~z -> -1 lane
	kmovb	%edx, %k4
	kmovb	%esi, %k5
	mov	$1, %eax
	vpbroadcastq	%rax, %zmm6
	vpsubq	%zmm6, %zmm20, %zmm20{%k4}
	vmovdqa64	%zmm19, %zmm20{%k5}
6:	# return seed in zmm0
	vmovdqa64	%zmm20, %zmm0
	add	$320, %rsp
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	.size	recip_3by2, .-recip_3by2

	.section	.rodata
	.align	64
seed_byte_perm:
	.byte	3,4,5,6,7,8,9,10, 10,11,12,13,14,15,16,17
	.byte	16,17,18,19,20,21,22,23, 23,24,25,26,27,28,29,30
	.byte	29,30,31,32,33,34,35,36, 36,37,38,39,40,41,42,43
	.byte	42,43,44,45,46,47,48,49, 49,50,51,52,53,54,55,56
seed_vshift:
	.quad	7,3,7,3,7,3,7,3
u52_dec_perm:
	.byte	0,1,2,3,4,5,6,7, 6,7,8,9,10,11,12,13
	.byte	13,14,15,16,17,18,19,20, 19,20,21,22,23,24,25,26
	.byte	26,27,28,29,30,31,32,33, 32,33,34,35,36,37,38,39
	.byte	39,40,41,42,43,44,45,46, 45,46,47,48,49,50,51,52
dec_vshift:
	.quad	0,4,0,4,0,4,0,4
	.section	.note.GNU-stack,"",@progbits
