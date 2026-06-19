






















































   
   
   
   








	.text
	.align	16, 0x90
	.globl	sbn_mul_1c
		.type	sbn_mul_1c,@function
	
sbn_mul_1c:

	

	jmp	.Lent
		.size	sbn_mul_1c,.-sbn_mul_1c
	.align	16, 0x90
	.globl	sbn_mul_1
		.type	sbn_mul_1,@function
	
sbn_mul_1:

	
	xor	%r8d, %r8d	
.Lent:	mov	(%rsi), %r9

	push	%rbx
	push	%r12
	push	%r13

	lea	(%rsi,%rdx,8), %rsi
	lea	-32(%rdi,%rdx,8), %rdi
	mov	%edx, %eax
	xchg	%rcx, %rdx		

	neg	%rcx

	and	$3, %al
	jz	.Lb0
	cmp	$2, %al
	jz	.Lb2
	jg	.Lb3

.Lb1:	mov	%r8, %r12
	.byte	0xc4,194,227,0xf6,193
	sub	$-1, %rcx
	jz	.Lwd1
	.byte	0xc4,0x62,0xb3,0xf6,0x04,0xce		
	.byte	0xc4,0x62,0xa3,0xf6,0x54,0xce,0x08	
	add	%r12, %rbx
	jmp	.Llo1

.Lb3:	.byte	0xc4,66,163,0xf6,209
	.byte	0xc4,0x62,0x93,0xf6,0x64,0xce,0x08	
	.byte	0xc4,0xe2,0xe3,0xf6,0x44,0xce,0x10	
	sub	$-3, %rcx
	jz	.Lwd3
	add	%r8, %r11
	jmp	.Llo3

.Lb2:	mov	%r8, %r10		
	.byte	0xc4,66,147,0xf6,225
	.byte	0xc4,0xe2,0xe3,0xf6,0x44,0xce,0x08	
	sub	$-2, %rcx
	jz	.Lwd2
	.byte	0xc4,0x62,0xb3,0xf6,0x04,0xce		
	add	%r10, %r13
	jmp	.Llo2

.Lb0:	mov	%r8, %rax		
	.byte	0xc4,66,179,0xf6,193
	.byte	0xc4,0x62,0xa3,0xf6,0x54,0xce,0x08	
	.byte	0xc4,0x62,0x93,0xf6,0x64,0xce,0x10	
	add	%rax, %r9
	jmp	.Llo0

.Ltop:	jrcxz	.Lend
	adc	%r8, %r11
	mov	%r9, (%rdi,%rcx,8)
.Llo3:	.byte	0xc4,0x62,0xb3,0xf6,0x04,0xce		
	adc	%r10, %r13
	mov	%r11, 8(%rdi,%rcx,8)
.Llo2:	.byte	0xc4,0x62,0xa3,0xf6,0x54,0xce,0x08	
	adc	%r12, %rbx
	mov	%r13, 16(%rdi,%rcx,8)
.Llo1:	.byte	0xc4,0x62,0x93,0xf6,0x64,0xce,0x10	
	adc	%rax, %r9
	mov	%rbx, 24(%rdi,%rcx,8)
.Llo0:	.byte	0xc4,0xe2,0xe3,0xf6,0x44,0xce,0x18	
	lea	4(%rcx), %rcx
	jmp	.Ltop

.Lend:	mov	%r9, (%rdi)
.Lwd3:	adc	%r8, %r11
	mov	%r11, 8(%rdi)
.Lwd2:	adc	%r10, %r13
	mov	%r13, 16(%rdi)
.Lwd1:	adc	%r12, %rbx
	adc	$0, %rax
	mov	%rbx, 24(%rdi)

	pop	%r13
	pop	%r12
	pop	%rbx
	
	ret
		.size	sbn_mul_1,.-sbn_mul_1

