#!/usr/bin/env python3
"""Compare decode arms: tok/s, ms/token and the MoE term, with every run shown."""
import json, sys, glob, os, statistics
root = sys.argv[1] if len(sys.argv) > 1 else 'results/dsv4-hugepage-ab'
def load(p):
    try: return json.load(open(p))
    except Exception: return None
arms = {}
for d in sorted(glob.glob(os.path.join(root, '*/'))):
    g = load(os.path.join(d, 'generation.json'))
    if not g or not g.get('generated_tokens'): continue
    arm = os.path.basename(d.rstrip('/')).rsplit('-', 1)[0]
    n, ds = g['generated_tokens'], g['decode_seconds']
    dec = g['phases']['decode']['graph']
    arms.setdefault(arm, []).append({
        'tok_s': n/ds, 'ms_tok': ds/n*1000,
        'moe_ms': dec.get('moe_seconds', 0)/n*1000,
        'layer_ms': dec.get('rank_local_layer_seconds', 0)/n*1000,
        'stage_s': g.get('resident_staging_seconds', 0),
        'prefill_s': g.get('prefill_seconds', 0),
        'ids': str(g.get('generated_token_ids', ''))[:44]})
hdr = f"{'arm':<22}{'tok/s':>9}{'ms/tok':>9}{'MoE ms':>9}{'layer ms':>10}{'stage s':>9}{'prefill s':>10}"
print(hdr); print('-'*len(hdr))
med = {}
for arm, rs in arms.items():
    for r in rs:
        print(f"{arm:<22}{r['tok_s']:>9.3f}{r['ms_tok']:>9.1f}{r['moe_ms']:>9.1f}"
              f"{r['layer_ms']:>10.1f}{r['stage_s']:>9.1f}{r['prefill_s']:>10.1f}")
    m = {k: statistics.median([r[k] for r in rs]) for k in ('tok_s','ms_tok','moe_ms','layer_ms')}
    med[arm] = m
    print(f"{arm+' MEDIAN':<22}{m['tok_s']:>9.3f}{m['ms_tok']:>9.1f}{m['moe_ms']:>9.1f}{m['layer_ms']:>10.1f}")
    print()
if 'no-arena-hugepages' in med and 'arena-hugepages' in med:
    b, c = med['no-arena-hugepages'], med['arena-hugepages']
    print(f"  decode   {b['tok_s']:.3f} -> {c['tok_s']:.3f} tok/s   {b['tok_s'] and c['tok_s']/b['tok_s']:.3f}x")
    print(f"  MoE term {b['moe_ms']:.1f} -> {c['moe_ms']:.1f} ms/tok  "
          f"{b['moe_ms'] and b['moe_ms']/c['moe_ms']:.3f}x faster")
ids = {arm: rs[0]['ids'] for arm, rs in arms.items()}
print(f"\n  token ids identical across arms: {len(set(ids.values()))==1}")
for a,v in ids.items(): print(f"    {a}: {v}")
