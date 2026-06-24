#!/usr/bin/env python3
# Monolithic recip_u1024: recip_u512_mp body (top-half recip) + 8->16 body inlined,
# one prologue/frame, NO x8 round-trip (512-recip writes straight into XHI slot).
#   args: rdi=D(16), rsi=out(16)
# frame: [0,160)=recip_u512_mp scratch (unshifted) | XHI@160(8) | D@224(16) | OUT@352 | NEG@360 | MR@368(9)
h=8; n=16
XHI=160; DD=224; OUT=352; NEG=360; MR=368; FRAME=448
SRC='/home/dev/bigint/GIMP/experimental/simd_bigint/reciprocal_u512_mp.s'
L=[]
def e(s): L.append('\t'+s)
def raw(s): L.append(s)
def Doff(j): return DD+8*j
def Moff(k): return MR+8*(k-1)
r32={'%r8':'%r8d','%r9':'%r9d','%r10':'%r10d','%r11':'%r11d','%r12':'%r12d','%r13':'%r13d',
     '%r14':'%r14d','%r15':'%r15d','%rcx':'%ecx','%rdi':'%edi','%rsi':'%esi','%rax':'%eax'}

raw('\t.text'); raw('\t.globl\trecip_u1024'); raw('\t.type\trecip_u1024,@function')
raw('recip_u1024:')
for r in ('%rbx','%rbp','%r12','%r13','%r14','%r15'): e('push\t'+r)
e('sub\t$%d, %%rsp'%FRAME)
e('xor\t%ebp, %ebp')
e('mov\t%%rsi, %d(%%rsp)\t\t# save out'%OUT)
# copy D(16) -> DD ; then rdi = &D[8] (top half) for the 512-recip
for i in range(16):
    e('mov\t%d(%%rdi), %%rax'%(8*i)); e('mov\t%%rax, %d(%%rsp)'%(Doff(i)))
e('lea\t%d(%%rsp), %%rdi\t\t# rdi = &D[8] (top half)'%(Doff(8)))
# ===== splice recip_u512_mp body (top-half recip -> x8 in XHI), output redirected =====
raw('\t# ---- inlined recip_u512_mp (writes x8 to XHI@%d) ----'%XHI)
lines=open(SRC).read().splitlines()
i0=next(i for i,l in enumerate(lines) if l.strip()=='xor\t%ebp, %ebp' or l.strip().startswith('xor\t%ebp'))
i1=next(i for i,l in enumerate(lines) if l.strip().startswith('add\t$160'))
for l in lines[i0+1:i1]:
    if 'mov\t160(%rsp), %r8' in l:
        L.append('\tlea\t%d(%%rsp), %%r8\t\t# -> XHI'%XHI)   # redirect output ptr to XHI slot
    else:
        L.append(l)
raw('\t# ---- inlined 8->16 step (reads XHI@%d, D@%d) ----'%(XHI,DD))
# ===== 8->16 body (no copy-in: XHI & D already in frame) =====
A=['%r8','%r9','%r10','%r11','%r12','%r13','%r14','%r15','%rcx','%rdi','%rsi']
# parallelogram window [7,16]+guard
e('mov\t%d(%%rsp), %%rdx\t# x_hi[0]'%XHI)
e('mulx\t%d(%%rsp), %s, %s'%(Doff(7),A[0],A[1]))
for j in range(8,16):
    k=j-7
    e('mulx\t%d(%%rsp), %%rax, %s'%(Doff(j),A[k+1]))
    e(('add' if j==8 else 'adc')+'\t%%rax, %s'%A[k])
e('adc\t$0, %s'%A[9])
for i in range(1,8):
    e('mov\t%d(%%rsp), %%rdx\t# x_hi[%d]'%(XHI+8*i,i))
    if i==1: e('xor\t%s, %s'%(r32[A[10]],r32[A[10]]))
    else:    e('xor\t%eax, %eax')
    for p in range(7,17):
        j=p-i
        if j<0 or j>=16: continue
        k=p-7
        e('mulx\t%d(%%rsp), %%rax, %%rbx'%Doff(j))
        e('adcx\t%%rax, %s'%A[k]); e('adox\t%%rbx, %s'%A[k+1])
    e('adcx\t%%rbp, %s'%A[10])
# V=2W
for k in range(9,0,-1): e('shld\t$1, %s, %s'%(A[k-1],A[k]))
e('shl\t$1, %s'%A[0])
# m, neg, |M|=V^m, spill |M|[1..9]
e('mov\t%s, %%rax'%A[9]); e('sar\t$63, %rax')
e('mov\t%rax, %rbx'); e('not\t%rbx'); e('mov\t%%rbx, %d(%%rsp)'%NEG)
for k in range(1,10): e('xor\t%%rax, %s'%A[k])
for k in range(1,10): e('mov\t%s, %d(%%rsp)'%(A[k],Moff(k)))
# HP
AH=['%r8','%r9','%r10','%r11','%r12','%r13','%r14','%r15','%rcx','%rdi']
e('mov\t%d(%%rsp), %%rdx\t# x_hi[0]'%XHI)
e('mulx\t%d(%%rsp), %s, %s'%(Moff(8),AH[0],AH[1]))
e('mulx\t%d(%%rsp), %%rax, %s'%(Moff(9),AH[2]))
e('add\t%%rax, %s'%AH[1]); e('adc\t$0, %s'%AH[2])
for i in range(1,8):
    e('mov\t%d(%%rsp), %%rdx\t# x_hi[%d]'%(XHI+8*i,i))
    nt=i+2
    e('xor\t%s, %s'%(r32[AH[nt]],r32[AH[nt]]))
    for k in range(max(1,8-i),10):
        p=i+k; ka=p-8
        e('mulx\t%d(%%rsp), %%rax, %%rbx'%Moff(k))
        e('adcx\t%%rax, %s'%AH[ka]); e('adox\t%%rbx, %s'%AH[ka+1])
    e('adcx\t%%rbp, %s'%AH[nt])
# output
e('mov\t%d(%%rsp), %%rsi'%OUT)
e('mov\t%d(%%rsp), %%rax'%NEG)
e('test\t%rax, %rax'); e('jnz\t.Lsub1024')
for t in range(8): e('mov\t%s, %d(%%rsi)'%(AH[1+t],8*t))
e('mov\t%d(%%rsp), %%rax'%XHI); e('add\t%s, %%rax'%AH[9]); e('mov\t%%rax, %d(%%rsi)'%(8*8))
for k in range(1,8):
    e('mov\t%d(%%rsp), %%rax'%(XHI+8*k)); e('adc\t$0, %rax'); e('mov\t%%rax, %d(%%rsi)'%(8*(8+k)))
e('jmp\t.Ldone1024')
raw('.Lsub1024:')
e('xor\t%eax, %eax'); e('sub\t%s, %%rax'%AH[1]); e('mov\t%rax, 0(%rsi)')
for t in range(1,8):
    e('mov\t%rbp, %rax'); e('sbb\t%s, %%rax'%AH[1+t]); e('mov\t%%rax, %d(%%rsi)'%(8*t))
e('mov\t%d(%%rsp), %%rax'%XHI); e('sbb\t%s, %%rax'%AH[9]); e('mov\t%%rax, %d(%%rsi)'%(8*8))
for k in range(1,8):
    e('mov\t%d(%%rsp), %%rax'%(XHI+8*k)); e('sbb\t$0, %rax'); e('mov\t%%rax, %d(%%rsi)'%(8*(8+k)))
raw('.Ldone1024:')
e('add\t$%d, %%rsp'%FRAME)
for r in ('%r15','%r14','%r13','%r12','%rbp','%rbx'): e('pop\t'+r)
e('ret')
raw('\t.size\trecip_u1024, .-recip_u1024')
raw('\t.section\t.note.GNU-stack,"",@progbits')
open('/home/dev/bigint/GIMP/experimental/simd_bigint/recip_u1024_inl.s','w').write('\n'.join(L)+'\n')
print("wrote recip_u1024_inl.s (%d lines)"%len(L))
