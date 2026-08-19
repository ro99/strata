import json, sys
label = sys.argv[1]; rep = sys.argv[2]
d = json.load(sys.stdin); t = d.get('timings', {})
print("%s rep %s: decode %.3f tok/s (%.1f ms/tok)  prefill %.3f tok/s  n=%s"
      % (label, rep, t.get('predicted_per_second', 0), t.get('predicted_per_token_ms', 0),
         t.get('prompt_per_second', 0), t.get('predicted_n')))
