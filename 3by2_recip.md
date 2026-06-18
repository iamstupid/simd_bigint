div3by2输入的reciprocal需要按照3 zmm/2 zmm的精度进行一次微调，需要+-1然后反乘，进行carry out，然后比较大小，进行inc/dec。我认为这个步骤最好直接和这个scalar reciprocal routine合并成一个函数。

其中用到的u416的一些算法：

```
// 乘法，取自general basecase multiplication；这里知道长度为8完全展开就行
// bx[ind] 是 zmm one to all broadcast
#define mul1_substep(a, ind) {                              \
    n1 = madd52hi(n1, (a), bx[ind]);                        \
    n0 = add(n0, alignr64(n1, m[ind], 7-(ind)));            \
    m[ind] = n1;                                            \
    n1 = madd52lo(zero(), (a), bx[ind]);                    \
}
#define mul1_substep0(a) {                                  \
    n1  = madd52hi(n1, (a), bx[0]);                         \
    n0  = add(n0, alignr64(n1, m[0], 7));                   \
    m[0] = n1;                                              \
    lo0 = madd52lo(zero(), (a), bx[0]);                     \
}

    for(an = an >> 3; an; --an){
        ax = load_vec(a++);
        n0 = zero();
        n1 = zero();
        switch(bn){
            case 8: mul1_substep (ax, 7);
            case 7: mul1_substep (ax, 6);
            case 6: mul1_substep (ax, 5);
            case 5: mul1_substep (ax, 4);
            case 4: mul1_substep (ax, 3);
            case 3: mul1_substep (ax, 2);
            case 2: mul1_substep (ax, 1);
            case 1: mul1_substep0(ax);
        }
        n0 = add(n0, lo0);
        store_vec(r++, n0);
    }
```


```
// 完全 canonize，用在乘法之后的进位
#define canonize(vec, carry, sigcarry) do {                         \
    static const _vec M = MASK52;\
    _vec hi  = srli(vec, 52);                /* full per-lane overflow */ \
    _vec his = alignr64(hi, carry, 7);       /* hi[i-1]; lane0 <- carry[7] */ \
    _vec _canon_t = add(and_v(vec, M), his); /* low + carry-in, < 2^53 */ \
    carry    = hi;                                                  \
    unsigned g = (unsigned)gtu(_canon_t, M); /* generate : t >  MASK */ \
    unsigned p = (unsigned)eq (_canon_t, M); /* propagate: t == MASK */ \
    unsigned chain = p + ((g << 1) | sigcarry);  /* SWAR carry ripple   */  \
    sigcarry = chain >> 8;                   /* carry out of lane 7  */     \
    __mmask8 cin = (__mmask8)(p ^ chain);    /* lanes that received a carry */ \
    vec = and_v(sub(_canon_t, M, cin, _canon_t), M);         \
} while(0)

```


```
// 已经canonical的两个zmm的带carry加法
#define carry_prop_t(add, sub, prp, r, a, b, carry, ...) { \
    _vec res = add(a, b); \
    __mmask16 _prop = eq(res, prp); \
    __mmask16 _carr = gtu(res, MASK52); \
    carry |= _carr << 1; \
    carry += _prop; \
    _prop ^= carry; \
    carry >>=8; \
    res = and_v(sub(res, MASK52, _prop, res), MASK52); \
    store_vec(r++, res, ##__VA_ARGS__); \
}

```


```
// 使用mask做比较
    for(bx = load_vec(bp, bm); ; --na, --nb){
        const uint8_t ne = (uint8_t)neq(ax, bx);
        if(ne){
            const uint8_t lt = (uint8_t)ltu(ax, bx);
            /* lt/gt lane-masks are disjoint, so lt>gt iff the top differing lane has a<b */
            if(lt > (uint8_t)(ne ^ lt)){
                SWAP(cpvec,a,b); SWAP(uint64_t,na,nb); SWAP(uint8_t,at,bt); sig ^= 1;
            }
            goto compared;
        }
        if(!na) return 0;                     /* every lane equal -> a == b */
        ax = load_vec(--ap); bx = load_vec(--bp); at = bt = 7;
    }

```

类似地推导一下dec和inc的写法。比canon更简单点。

```
// ---- decode -----------------------------------------------------------

// digit i gathers input bytes floor(6.5 i) .. +7 of its 52-byte block
alignas(64) static const uint8_t u52_dec_perm[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     6,  7,  8,  9, 10, 11, 12, 13,
    13, 14, 15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24, 25, 26,
    26, 27, 28, 29, 30, 31, 32, 33,
    32, 33, 34, 35, 36, 37, 38, 39,
    39, 40, 41, 42, 43, 44, 45, 46,
    45, 46, 47, 48, 49, 50, 51, 52,
};

// bit length of {ap, an}; tolerates zero high limbs
static inline uint64_t u64_bit_length(const uint64_t *ap, uint64_t an){
    while(an && ap[an - 1] == 0) --an;
    if(!an) return 0;
    return an * 64 - (uint64_t)__builtin_clzll(ap[an - 1]);
}

// unpack {ap, an} u64 limbs into n52 = ceil(bits/52) canonical u52 digits at
// r (vector-padded: full-vector stores, zero lanes beyond n52). Returns n52.
static inline uint64_t u52_from_u64(pvec r, const uint64_t *ap, uint64_t an){
    const uint64_t bits = u64_bit_length(ap, an);
    const uint64_t n52 = (bits + 51) / 52;
    const uint8_t *p = (const uint8_t *)ap;
    int64_t rem = (int64_t)(an * 8);
    const _vec perm = load_vec((cpvec)u52_dec_perm);
    const _vec sh = setr_64(0, 4, 0, 4, 0, 4, 0, 4);
    uint64_t blocks = (n52 + 7) >> 3;
    for(; rem >= 64 && blocks > 1; --blocks, p += 52, rem -= 52, ++r)
        store_vec(r, and_v(srlv(permb(perm, load_vec((cpvec)p)), sh), MASK52));
    for(; blocks; --blocks, p += 52, rem -= 52, ++r){    // at most two masked
        const _vec w = vec_fn(maskz_loadu_epi8)(
            rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p);
        store_vec(r, and_v(srlv(permb(perm, w), sh), MASK52));
    }
    return n52;
}
```

合成kernel的伪代码：

```
zmm 3by2_recip(u64[13] I){
  // B = 2^416
  // compute (B^3-1)/I - B
  u64[7] recip_buf;
  SNIPPET_COMPUTE_RECIP(recip_buf, I+6);
  // this computes (B^2-1)/hi(I)
  zmm recip = from_u64_for_reciprocal(recip_buf);
  // 这里的from_u64多抓了末尾的一位，少抓了最开头的1
  zmm [i0, i1] = from_u64(I);
  // 2^416 + recip ~ floor((2^832-1)/i0)
  // 之后我们需要比较的就是 [i0, i1] * recip 与 [all1, all1, neg(recip)] 之间的大小关系
}
```

---

## 验证与补完（2026-06-16，harness: `bench/recip_3by2_verify.c`）

### 1. 记号与精确定义
- 块基 **B = 2^416**（8 lane × 52 bit）。
- 除数 `I` 已归一化：**2^831 ≤ I < 2^832**（最高位置位），u64[13]，记两块 `[i0, i1]`：
  `i0 = I mod B`（低 416 bit），`i1 = ⌊I/B⌋`（高 416 bit，最高位置位）。
- **目标（3/2 倒数，单块）**：`V = ⌊(B³−1)/I⌋ − B = ⌊(2^1248−1)/I⌋ − 2^416 ∈ [0, B)`。
  完整商 `Q = V + B = ⌊(2^1248−1)/I⌋ ∈ [B, 2B)`。

### 2. `from_u64_for_reciprocal` 的精确语义
- `D = I` 的高 7 个 u64 = `⌊I/2^384⌋`（448-bit，已归一化）→ 即 `SNIPPET_COMPUTE_RECIP(recip_buf, I+6)`。
- `recip_mono` 给出 `R ≈ ⌊2^895/D⌋`；实测误差 `ε = R − ⌊2^895/D⌋ ∈ [−6, +3]`。
- **块种子** `seed = ⌊R/2^31⌋ − 2^416 ∈ [0, B)`，即提取 `R` 的 **bit[31..446]**（416 bit）重排为 8×52：
  - “丢掉最开头的 1” = 丢恒为 1 的 bit 447；
  - “多抓末尾一位” = 从 bit 31（而非 bit 32）起，补偿 `recip_mono` 携带的是 `2^895 = 2^896/2`（半位）。
  - 标度核对：`⌊R/2^31⌋ ≈ 2^895/(D·2^31) = 2^(1279)/I·... = 2^1248/I`，与 `Q` 同阶。✔

### 3. 核心结论：seed 与 V 至多差 1（且单边）
设 `q = seed + B = ⌊R/2^31⌋`（完整商估计）。利用 `D = ⌊I/2^384⌋ ⇒ I/2^384 − 1 < D ≤ I/2^384`：

- **上界 `q ≤ Q`（seed ≤ V）**：`q ≤ (⌊2^895/D⌋+3)/2^31 ≤ 2^864/D + 3·2^−31`；又 `D > (I−2^384)/2^384`
  ⇒ `2^864/D < 2^1248/I + 2^−30` ⇒ `q < 2^1248/I + 1`。实测严格 `q ≤ Q`（含 I=2^831 退化点也取等）。
- **下界 `q ≥ Q−1`（seed ≥ V−1）**：`q ≥ (⌊2^895/D⌋−6)/2^31 ≥ 2^864/D − 7·2^−31`；
  又 `D ≤ I/2^384 ⇒ 2^864/D ≥ 2^1248/I` ⇒ `q ≥ ⌊2^1248/I − 1⌋ ≥ Q − 1`。

> **∴ seed ∈ {V−1, V}，误差 ∈ {−1, 0}：至多差 1，且永不超过 V（单边低估）。**

**实证**（`recip_mono` 真实汇编 + GMP 精确大数）：
- random 5×10⁶、adversarial 7×10⁶（低 384 位全 1 / 最小 D / 2^831+k）：全部 `err = 0`；
- 针对整数边界构造 `I = round(2^1248/m)` 扫 5 个互不相交切片共 3×10⁸ 例：`err ∈ {−1, 0}`（≈60% 取 −1，40% 取 0），**无 −2，无 +1**。

### 4. 修正步骤（补完 kernel）——只乘 `seed·I`，与定值 `T` 比（DEC 免费 / INC 需 +I）
**不要**去形成 `(B+seed)·I`。只算 **`seed·I`**，与一个**预先算好的定值**比较：

```
T := B³ − 1 − B·I = 2^1248 − 1 − 2^416·I = [all1, ~i0, ~i1]
```
`T` 只需对除数两块取**按位补**（`~i0, ~i1`，最高块全 1）——无乘法、无进位、无 ≥2^832 的额外 lane。

记 `q = B+seed`，`Q = ⌊(2^1248−1)/I⌋`。用 `B·I = B³−1 − T` 把 `B·I` 折进 `T`，估计的余数为
```
ρ = (B³−1) − (B+seed)·I = (B³−1) − B·I − seed·I = T − seed·I
```

| 情形 | 条件 | (用 ρ=T−seed·I) | 代价 |
|---|---|---|---|
| **DEC** seed 太大 (`q≥Q+1`) | `seed·I > T` | `ρ < 0` | 直接比，**零额外算术**（“乘积较大就行”） |
| 精确 `q=Q` | `T−I < seed·I ≤ T` | `ρ ∈ [0,I)` | — |
| **INC** seed 太小 (`q≤Q−1`) | `seed·I + I ≤ T` | `ρ ≥ I` | **多加一次 I**（或预算 `T−I` 再比） |

> 顺序：先做免费的 DEC（`seed·I > T`），不中再做带 `+I` 的 INC；因 `|误差| ≤ 1`，至多触发其一。
> `seed·I < B³` 且 `T < B³` ⇒ 全程 **3 块**；而 `(B+seed)·I < 2B³` 需第 4 块（bit-1248 进位 lane）。
> 故相比之下省了：`+B·I` 那次整块加法、把 `i1` 加进 block-2(≥2^832) 及其往 bit-1248 的进位。

**⚠ 勘误**：原 note `[i0,i1]*recip 与 [all1,all1,neg(recip)]`——左边 `[i0,i1]*recip = seed·I` **对**，
右边应为 **`T = [all1, ~i0, ~i1]`**（`~i = 2^416−1−i` 按位补，最高块全 1），不是 `[all1,all1,neg(recip)]`。

**为什么 DEC 不能删（结构性原因，方向）**：seed 是「高位截断后的除数」的倒数。把除数**往小截** ⇒ 倒数**偏大**
⇒ seed 可能 `> V`，这就是 `+` 侧、DEC 必须存在的方向。具体：`I0` 可达 `B−1`，使真除数 `[I0,I1] → (I1+1)·B`，
真倒数 `→ (B²−1)/(I1+1) < (B²−1)/I1 = recip`。

**幅度（为何仍是 ±1）**：高估 `≈ B·I0/I1²`，但 recip 是按 **448-bit**（`recip_mono(I+6)`，吃进 `I0` 高 32 位）算的，
不是只按 416-bit 的 `I1`。吃 `I0` 高 `k` 位 ⇒ 高估 `< 4·2^−k`：
> `k=0`（只看 416-bit `I1`）→ 高估可达 +4，这只是反事实，**说明为何必须算到 448-bit**；
> `k≥2` 即 `<1`；doc 的 `k=32` ⇒ 高估 `≪2^−30`。
故 `|seed − V| ≤ 1`，单次 DEC / 单次 INC 各自足够，**不可能 +4**。（§3 的 `floor(R/2^31)` 模型实测落在 `{−1,0}`。）
DEC 侧零额外代价，保留不增成本。

**这是一个一次性的 verify pass**：算**全乘积** `seed·I`（3 块，`mul_u52_1` + `mpn_canon_pos`），精确比较、±1 纠正。
**不截断**——每次除法 setup 只做一次,没有截断的动机,截断只会引入正确性风险、省不了几个 PP。
verify pass 的职责是「实测乘积按结果纠正」,故 INC/DEC 两侧都查、都留（不硬假定种子偏哪边）；实测全乘积 + 448-bit
种子只有 INC 触发（`seed∈{V−1,V}`,`P` 永不 > `T`）,DEC 是零成本的另一侧兜底。参考实现 `bench/recip_3by2_kernel.c` 实测 0 失配。

### 5. INC / DEC 的 SWAR 写法（单块 8 lane，比 canonize 简单：无 generate 项）
进位/借位都是从 lane0 起、沿一段连续“全 1 / 全 0”lane 传播的单链：

```c
// INC: +1。 借 canonize 的 SWAR 链：carry 从 lane0 进，穿过低位连续 ==MASK52 的 lane
#define inc52(vec) do {                                                  \
    __mmask8 p     = eq(vec, MASK52);          /* 会传播进位的 lane */    \
    unsigned chain = (unsigned)p + 1u;         /* +1 注入 lane0 的 ripple */ \
    __mmask8 flip  = (__mmask8)((unsigned)p ^ chain); /* 被触及的 lane */  \
    /* flip 中的全 1 lane -> 0；唯一的非全 1 lane -> +1 */                \
    vec = add(vec, set1(1), (__mmask8)(flip & ~p), vec);  /* 吸收位 +1 */  \
    vec = and_v(vec, zero(), (__mmask8)(flip &  p), vec); /* 传播位清 0 */  \
    /* carry_out = chain>>8 : 仅当整块全 1（seed=B−1）才溢出进隐含位 */    \
} while(0)

// DEC: −1。 与 INC 对称，借位沿低位连续 ==0 的 lane 传播
#define dec52(vec) do {                                                  \
    __mmask8 z     = eq(vec, zero());          /* 会传播借位的 lane */    \
    unsigned chain = (unsigned)z + 1u;                                   \
    __mmask8 flip  = (__mmask8)((unsigned)z ^ chain);                    \
    vec = sub(vec, set1(1), (__mmask8)(flip & ~z), vec);  /* 吸收位 −1 */  \
    vec = or_v (vec, MASK52,(__mmask8)(flip &  z), vec);  /* 借位 lane 置全 1 */ \
} while(0)
```
（verify pass 两侧都查，故 `inc52`/`dec52` 都留着；实测只 INC 触发，`dec52` 为另一侧兜底。）

---

## 6. 证明：截断 load 的 `recip_mono` 输出已足够（|seed − V| ≤ 1），无需精确 floor 倒数 (2026-06-18)

**结论**：直接把 `recip_mono` 的输出按 bit[31..446] 截断装进 zmm 得到的 `seed`，与精确 3/2 倒数 `V` 至多差 1。因此把它直接喂给 `OurDiv3by2`，其自带的商修正即可吸收（最多多一次 ±1 修正），**不需要** reverse-multiply / canonize / 比较 / inc-dec 这条 verify tail。

**记号**：`B = 2^416`；除数归一化 `2^831 ≤ D < 2^832`；精确倒数
`V = ⌊(B³−1)/D⌋ − B`，全商 `Q := V+B = ⌊(2^1248−1)/D⌋`。种子来自 `recip_mono` 对 `D` 顶 448 位：
`d := ⌊D/2^384⌋ ∈ [2^447, 2^448)`，`R := recip_mono(d) ≈ ⌊2^895/d⌋`，
`seed = ⌊R/2^31⌋ − B`，`q := seed+B = ⌊R/2^31⌋`。因 `seed−V = q−Q`，只需证 **|q−Q| ≤ 1**。
设实数 `T := 2^1248/D`，则 `Q = ⌊T − 1/D⌋`，`0 < 1/D ≤ 2^-831`。

**引理 1（除数截断 —— 这正是为何取 448 位而非 416 位）**：令 `d' := D/2^384`（实数），`d = ⌊d'⌋`，`0 ≤ d'−d < 1`。
因 `2^1248/D = 2^864/d'`，故
`θ := 2^864/d − T = 2^864(d'−d)/(d·d')`。又 `d'−d < 1`，`D ≥ 2^831 ⇒ d' ≥ 2^447 ⇒ d > 2^446`，
所以 `d·d' > 2^893`，得 **`0 ≤ θ < 2^864/2^893 = 2^-29`**（单边，极小）。
> 对照：若只用高块 `i1=⌊D/2^416⌋`（416 位），同样计算给出高估可达 `B·i0/i1² < 2^832/2^830 = 4`。
> 多吃 `i0` 的高 32 位（取 `d=⌊D/2^384⌋`）把它缩小 `2^32`，从 ≤4 降到 <2^-29 —— 这就是把 ±4 种子变成 ±1 种子的那个因子。

**引理 2（`recip_mono` 的 ULP 误差经 `>>31` 后可忽略）**：`recip_mono` 满足 `|R − 2^895/d| ≤ E`，实测 `E ≤ 7`
（[−3,+6] ULP 加 floor 的小数部分）；下面只需 `E < 2^31`（即 R 只要顶 ~30 位正确，实际有 ~448 位，余量 ~28 位）。除以 `2^31`：
`R/2^31 = 2^864/d + ζ`，`|ζ| ≤ E/2^31 < 2^-28`。

**定理（|q−Q| ≤ 1）**：合并两引理，
`R/2^31 = (T+θ) + ζ = T + ρ`，`ρ = θ+ζ`，`|ρ| < 2^-29 + 2^-28 < 2^-27`。
于是 `q = ⌊T+ρ⌋`，`Q = ⌊T − 1/D⌋`，两个 floor 的参数之差
`|(T+ρ) − (T−1/D)| = |ρ + 1/D| < 2^-27 + 2^-831 < 2^-26 < 1`。
对任意实数 `|x−y| < 1` 有 `|⌊x⌋−⌊y⌋| ≤ 1`，故 **`|q−Q| ≤ 1`，即 `|seed−V| ≤ 1`**。∎

**为何 ±1 对 `OurDiv3by2` 足够**：3/2 一步里商估 `q̂ = q1+1`，其中 `⟨q1,q0⟩ = v·u2 + ⟨u2,u1⟩`。
把精确 `v=V` 换成 `v'=V−k`（`k∈{−1,0,1}`）只让乘积变 `k·u2 < k·B`，故 `⟨q1,q0⟩` 变 `<kB`，高半 `q1` 至多变 `|k|≤1`；
即 `q̂` 至多偏 1，余数检查 `r = ⟨u1,u0⟩ − q̂·d` 至多差一个 `d`，算法已有的修正（`r≥d ⇒ q̂++, r−=d`）刚好吸收：
±1 倒数至多多一次修正，绝不会更多。故精确 `⌊(B³−1)/D⌋` 非必需。

**收益**：单次 load 直接省掉整条 verify tail（reverse-multiply + 3 块 canonize + 比较 + inc/dec），
约占 `recip_3by2` 在 `recip_mono` 之后开销的大头（~192 cyc 中的 ~65 cyc），换成 `OurDiv3by2` 本就有的一次条件减 `d`。
精简核 `src/avx512/recip_3by2_seed.s`（= `recip_3by2.s` 去掉 verify tail），`bench/recip_seed_pm1.c` 实测 `seed−V ∈ {−1,0}`（单边低估，与 §3 一致）。
