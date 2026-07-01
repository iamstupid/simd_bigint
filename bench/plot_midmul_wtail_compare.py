import csv,sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src=sys.argv[1] if len(sys.argv)>1 else "midmul_wtail_compare.csv"
out=src.rsplit(".",1)[0]+".svg"
n,s,w=[],[],[]
for r in csv.DictReader(open(src)):
    x=int(r["n"]); n.append(x); s.append(float(r["simple_ns"])/x); w.append(float(r["wtail_ns"])/x)
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5.5))
for ax in (a1,a2):
    ax.plot(n,s,lw=1.1,color="C0",label="midmul simple (masked horizontal tail)")
    ax.plot(n,w,lw=1.1,color="C1",label="midmul wtail (horiz-bottom + vertical-top tail)")
    ax.set_xlabel("n  (an=n, bn=2n)"); ax.set_ylabel("ns / n_limb"); ax.grid(alpha=.3); ax.legend(fontsize=8)
a1.set_title("full range 8..%d"%n[-1])
lo,hi=40,min(96,n[-1]); a2.set_xlim(lo,hi)
sel=[y for x,y in zip(n,s) if lo<=x<=hi]; a2.set_ylim(min(sel)*.96,max(sel)*1.04)
a2.set_title("zoom %d..%d: wtail flattens the staircase steps"%(lo,hi))
for x in n:
    if lo<=x<=hi and x%8==0: a2.axvline(x,color="gray",lw=.5,ls=":",alpha=.6)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".svg",".png"),dpi=115)
g=(sum(s)-sum(w))/sum(s)*100; print(f"wrote {out}; mean simple->wtail = {g:+.2f}%")
