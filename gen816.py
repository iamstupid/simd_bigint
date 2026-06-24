#!/usr/bin/env python3
# Register-resident 8->16, scaling reciprocal_u512_mp.s step3 (faithful: V=2W, xor-abs,
# drop|M|[0], Delta HP). Parallelogram acc + HP acc held in registers; |M| spilled (HP mult).
#   args: rdi=x_hi(8), rsi=D(16), rdx=out
h=8; n=16
XHI=0; DD=64; OUT=192; NEG=200; MR=208   # MR: |M|[1..9] (9 limbs)
FRAME=320
L=[]
def e(s): L.append('\t'+s if not s.endswith(':') and not s.startswith('.') else s)
def lbl(s): L.append(s)
def Doff(j): return DD+8*j
def Moff(k): return MR+8*(k-1)         # |M|[k], k=1..9
r32={'%r8':'%r8d','%r9':'%r9d','%r10':'%r10d','%r11':'%r11d','%r12':'%r12d','%r13':'%r13d',
     '%r14':'%r14d','%r15':'%r15d','%rcx':'%ecx','%rdi':'%edi','%rsi':'%esi','%rax':'%eax'}
A=['%r8','%r9','%r10','%r11','%r12','%r13','%r14','%r15','%rcx','%rdi','%rsi']  # parallelogram a[0..10]
L.append('\t.text'); L.append('\t.globl\trecip_step_8_16'); L.append('\t.type\trecip_step_8_16,@function')
lbl('recip_step_8_16:')
for r in ('%rbx','%rbp','%r12','%r13','%r14','%r15'): e('push\t'+r)
e('sub\t$%d, %%rsp'%FRAME)
e('xor\t%ebp, %ebp')
e('mov\t%%rdx, %d(%%rsp)'%OUT)
for i in range(8): e('mov\t%d(%%rdi), %%rax'%(8*i)); e('mov\t%%rax, %d(%%rsp)'%(XHI+8*i))
for i in range(16): e('mov\t%d(%%rsi), %%rax'%(8*i)); e('mov\t%%rax, %d(%%rsp)'%(Doff(i)-DD+DD))
lbl('\t# ===== parallelogram W = D*x_hi window [7,16]+guard17 -> a[0..10] =====')
# row 0 (i=0): D[7..15], single chain. a[0..9].
e('mov\t%d(%%rsp), %%rdx\t# x_hi[0]'%XHI)
e('mulx\t%d(%%rsp), %s, %s'%(Doff(7),A[0],A[1]))
for j in range(8,16):
    k=j-7
    e('mulx\t%d(%%rsp), %%rax, %s'%(Doff(j),A[k+1]))
    e(('add' if j==8 else 'adc')+'\t%%rax, %s'%A[k])
e('adc\t$0, %s'%A[9])
# rows 1..7 (adcx/adox)
for i in range(1,8):
    e('mov\t%d(%%rsp), %%rdx\t# x_hi[%d]'%(XHI+8*i,i))
    if i==1: e('xor\t%s, %s'%(r32[A[10]],r32[A[10]]))   # introduce guard a[10], clear CF/OF
    else:    e('xor\t%eax, %eax')                        # clear CF/OF (rax overwritten)
    for p in range(7,17):
        j=p-i
        if j<0 or j>=16: continue
        k=p-7
        e('mulx\t%d(%%rsp), %%rax, %%rbx'%Doff(j))
        e('adcx\t%%rax, %s'%A[k])
        e('adox\t%%rbx, %s'%A[k+1])
    e('adcx\t%%rbp, %s'%A[10])      # drain CF into guard
lbl('\t# W = a[0..9] (positions 7..16); guard a[10]=%s discarded. V=2W.'%A[10])
# V = 2*W : shld over a[0..9]
for k in range(9,0,-1):
    e('shld\t$1, %s, %s'%(A[k-1],A[k]))
e('shl\t$1, %s'%A[0])
lbl('\t# m=sign-extend(V[9]=%s top); |M|[1..9]=a[1..9]^m (drop |M|[0]=a[0]); spill |M| to MR'%A[9])
e('mov\t%s, %%rax'%A[9]); e('sar\t$63, %rax')          # m
e('mov\t%rax, %rbx'); e('not\t%rbx'); e('mov\t%%rbx, %d(%%rsp)'%NEG)  # neg_mask=~m
for k in range(1,10): e('xor\t%%rax, %s'%A[k])         # |M|[1..9]=a[1..9]^m
for k in range(1,10): e('mov\t%s, %d(%%rsp)'%(A[k],Moff(k)))   # spill |M|[k] to MR
lbl('\t# ===== Delta HP = x_hi*|M|[1..9], keep [8,17] -> AH[0..9]; Delta=AH[1..9] =====')
AH=['%r8','%r9','%r10','%r11','%r12','%r13','%r14','%r15','%rcx','%rdi']  # a[0..9] positions [8,17]
# row 0 (i=0): |M|[8],|M|[9] positions 8,9,10
e('mov\t%d(%%rsp), %%rdx\t# x_hi[0]'%XHI)
e('mulx\t%d(%%rsp), %s, %s'%(Moff(8),AH[0],AH[1]))
e('mulx\t%d(%%rsp), %%rax, %s'%(Moff(9),AH[2]))
e('add\t%%rax, %s'%AH[1]); e('adc\t$0, %s'%AH[2])
# rows 1..7: row i introduces AH[i+2]
for i in range(1,8):
    e('mov\t%d(%%rsp), %%rdx\t# x_hi[%d]'%(XHI+8*i,i))
    nt=i+2
    e('xor\t%s, %s'%(r32[AH[nt]],r32[AH[nt]]))
    klo=max(1,8-i)
    for k in range(klo,10):
        p=i+k; ka=p-8
        e('mulx\t%d(%%rsp), %%rax, %%rbx'%Moff(k))
        e('adcx\t%%rax, %s'%AH[ka])
        e('adox\t%%rbx, %s'%AH[ka+1])
    e('adcx\t%%rbp, %s'%AH[nt])
lbl('\t# Delta = AH[1..9] (positions [9,17]); AH[0] discarded. out = x_hi*B^8 +/- Delta.')
e('mov\t%d(%%rsp), %%rsi'%OUT)        # out ptr
e('mov\t%d(%%rsp), %%rax'%NEG)        # neg_mask
e('test\t%rax, %rax')
e('jnz\t.Lsub816')
# ADD: out[0..7]=Delta[0..7]=AH[1..8]; out[8]=x_hi[0]+AH[9]; out[9..15]=x_hi[1..7]+carry
for t in range(8): e('mov\t%s, %d(%%rsi)'%(AH[1+t],8*t))
e('mov\t%d(%%rsp), %%rax'%XHI); e('add\t%s, %%rax'%AH[9]); e('mov\t%%rax, %d(%%rsi)'%(8*8))
for k in range(1,8):
    e('mov\t%d(%%rsp), %%rax'%(XHI+8*k)); e('adc\t$0, %rax'); e('mov\t%%rax, %d(%%rsi)'%(8*(8+k)))
e('jmp\t.Ldone816')
lbl('.Lsub816:')
e('xor\t%eax, %eax'); e('sub\t%s, %%rax'%AH[1]); e('mov\t%rax, 0(%rsi)')
for t in range(1,8):
    e('mov\t%rbp, %rax'); e('sbb\t%s, %%rax'%AH[1+t]); e('mov\t%%rax, %d(%%rsi)'%(8*t))
e('mov\t%d(%%rsp), %%rax'%XHI); e('sbb\t%s, %%rax'%AH[9]); e('mov\t%%rax, %d(%%rsi)'%(8*8))
for k in range(1,8):
    e('mov\t%d(%%rsp), %%rax'%(XHI+8*k)); e('sbb\t$0, %rax'); e('mov\t%%rax, %d(%%rsi)'%(8*(8+k)))
lbl('.Ldone816:')
e('add\t$%d, %%rsp'%FRAME)
for r in ('%r15','%r14','%r13','%r12','%rbp','%rbx'): e('pop\t'+r)
e('ret')
L.append('\t.size\trecip_step_8_16, .-recip_step_8_16')
L.append('\t.section\t.note.GNU-stack,"",@progbits')
open('/home/dev/bigint/GIMP/experimental/simd_bigint/recip_step_8_16r.s','w').write('\n'.join(L)+'\n')
print("wrote recip_step_8_16r.s (%d lines)"%len(L))
