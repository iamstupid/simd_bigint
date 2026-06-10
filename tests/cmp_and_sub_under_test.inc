static inline int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na > 0 && nb > 0);
    uint8_t sig = 0;
    if(na < nb){
        SWAP(cpvec, a, b);
        SWAP(uint64_t, na, nb);
        sig = 1;
    }
    uint8_t at = --na & 7, bt = --nb & 7;
    __mmask8 am = (__mmask8)((1u<<(at+1)) - 1), bm = (__mmask8)((1u<<(bt+1)) - 1);
    na >>= 3; nb >>= 3;
    cpvec ap = a + na, bp = b + nb;
    _vec ax, bx;

    /* skip a's vectors that sit above b's range (compared against zero) */
    for(ax = load_vec(ap, am); na > nb; --na, ax = load_vec(--ap)){
        if(neq(ax, zero())) goto compared;   /* a strictly longer -> a > b */
        at = 7;
    }
    /* same vector index now: walk down comparing a vs b */
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
compared:
    am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
    uint64_t i;
    uint16_t lbr = 0;
    for(i = 0; i < nb; ++i){
        ax = load_vec(a++), bx = load_vec(b++);
        borrow_prop(r, ax, bx, lbr);
    }
    if(na == nb){
        ax = load_vec(a, am), bx = load_vec(b, bm);
        borrow_prop(r, ax, bx, lbr, am);
    }else{
        ax = load_vec(a++), bx = load_vec(b, bm);
        borrow_prop(r, ax, bx, lbr);
        for(++i;i<na;++i){
            ax = load_vec(a++);
            borrow_prop(r, ax, zero(), lbr);
        }
        ax = load_vec(a, am);
        borrow_prop(r, ax, zero(), lbr, am);
    }
    return sig ? -(int64_t)(na*8+at+1) : (int64_t)(na*8+at+1);
}
