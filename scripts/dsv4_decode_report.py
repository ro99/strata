#!/usr/bin/env python3
"""Phase attribution for a DSV4 decode arm: which resource is argmax_r."""
import json, sys
g = json.load(open(sys.argv[1]))
n = g.get('generated_tokens', 0); ds = g.get('decode_seconds', 0.0)
print(f"prompt {g.get('prompt_tokens')} tok | prefill {g.get('prefill_seconds',0):.2f} s "
      f"| decode {ds:.3f} s / {n} tok")
if n and ds:
    print(f"\n  DECODE: {n/ds:.3f} tok/s   {ds/n*1000:.2f} ms/token\n")
per = ds/n if n else 1.0
rows=[]
for k, v in sorted(g.items()):
    if not isinstance(v,(int,float)) or not k.endswith('_seconds'): continue
    if k in ('decode_seconds','prefill_seconds','initialization_seconds'): continue
    if v <= 0.0005: continue
    rows.append((v,k))
for v,k in sorted(rows, reverse=True)[:18]:
    print(f"    {k:<48} {v:>9.3f} s  {v/n*1000 if n else 0:>8.2f} ms/tok  {v/ds*100 if ds else 0:>6.1f}%")
w = g.get('weight_cache') or {}
if w:
    h,m = w.get('hits',0), w.get('misses',0)
    print(f"\n  weight cache: hits {h} misses {m} "
          f"({h/(h+m)*100 if h+m else 0:.1f}%) evictions {w.get('evictions',0)}")
    print(f"  demand H2D {w.get('demand_h2d_bytes',0)/1e9:.2f} GB "
          f"wait {w.get('demand_wait_seconds',0):.2f} s")
    print(f"  cache cap/dev {[round(x/1e9,2) for x in w.get('capacity_bytes',[])]} GB "
          f"used {[round(x/1e9,2) for x in w.get('used_bytes',[])]} GB")
    print(f"  unpinned cap  {[round(x/1e9,2) for x in w.get('unpinned_capacity_bytes',[])]} GB")
print(f"\n  vram {[round(x/1e9,1) for x in g.get('device_vram_used_bytes',[])]} GB  "
      f"rss {g.get('rss_bytes',0)/1e9:.1f} GB")
print(f"  first ids: {str(g.get('generated_token_ids',g.get('token_ids','?')))[:90]}")
