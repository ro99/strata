#!/usr/bin/env python3
"""Freeze GLM-5.3 k-pool sparse indexer selections from the vendored reference.

The adapter's <=2,048 exactness gate cannot see the sparse branch, because
selecting the top `index_topk` of at most `index_topk` candidates is the
identity there.  That blind spot is why the indexer went unimplemented through a
whole performance campaign (record 0237).  This oracle closes it: it runs
`Glm5NextTextIndexer` from `experiments/references/glm5-next/`, unmodified and
imported from the vendored source rather than reimplemented, and freezes the
selected positions for probe rows on both sides of the threshold.

Inputs are not stored.  Both this script and the C++ test derive every weight
and activation from the same documented 64-bit LCG, so the fixture holds only
the expected selections and stays small enough to live in the tree.

Usage:
    python3 scripts/glm53_indexer_oracle.py [--out tests/fixtures/glm53/indexer-oracle.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments" / "references" / "glm5-next" / "modeling_glm5_next.py"

# The checkpoint's own indexer geometry (models/glm53f/config.json).  Only
# `hidden_size` and `q_lora_rank` are reduced, and only to keep the fixture's
# projections cheap: they are plain matrix multiplies and are not what the
# threshold hides.  Everything the selection semantics depend on -- head count,
# head width, `index_topk`, `index_kpool`, the tail rule -- is the real value.
HIDDEN_SIZE = 256
Q_LORA_RANK = 192
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
INDEX_KPOOL = 4
SEQUENCE = 2600
SEED = 0xDEAD_BEEF

# Rows chosen to cover: the identity region, the exact threshold, every
# `history % index_kpool` residue just above it, and the deeply sparse regime.
PROBE_ROWS = [0, 1, 3, 100, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053,
              2054, 2055, 2200, 2400, 2599]


def load_reference_indexer() -> type:
    """Exec the reference's indexer class out of the vendored source.

    The file is a generated `transformers` module with package-relative imports
    that cannot be resolved standalone, and `transformers` does not ship
    `glm5_next` yet.  Slicing the class out keeps the semantics byte-identical to
    the vendored reference while avoiding a reimplementation, which is the whole
    point of the oracle.
    """
    source = REFERENCE.read_text().splitlines()
    try:
        begin = next(i for i, line in enumerate(source)
                     if line.startswith("class Glm5NextTextIndexer"))
        end = next(i for i, line in enumerate(source)
                   if i > begin and line.startswith("def repeat_kv"))
    except StopIteration as error:  # pragma: no cover - vendored file changed
        raise SystemExit(f"{REFERENCE}: Glm5NextTextIndexer not found") from error
    module = types.ModuleType("glm5_next_indexer_reference")
    module.__dict__.update(torch=torch, nn=nn, F=F, Cache=object)
    exec(compile("\n".join(source[begin:end]), str(REFERENCE), "exec"),
         module.__dict__)
    return module.Glm5NextTextIndexer


class Lcg:
    """The 64-bit LCG shared with `tests/test_glm53_indexer_oracle.cpp`.

    Knuth's MMIX multiplier and increment; the float is built from bits 40..63
    so it is exactly representable in binary32 and both languages agree bit for
    bit.
    """

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_float(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & self.MASK
        return (self.state >> 40) / 8388608.0 - 1.0

    def tensor(self, *shape: int) -> torch.Tensor:
        count = 1
        for extent in shape:
            count *= extent
        values = [self.next_float() for _ in range(count)]
        return torch.tensor(values, dtype=torch.float32).view(*shape)


def build() -> dict:
    indexer_type = load_reference_indexer()
    config = types.SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=0,
        index_topk=INDEX_TOPK,
        q_lora_rank=Q_LORA_RANK,
        index_kpool=INDEX_KPOOL,
        index_kpool_always_select_tail=True,
    )
    indexer = indexer_type(config, layer_idx=0).eval()

    # Fill order is part of the contract with the C++ test.  Do not reorder.
    rng = Lcg(SEED)
    with torch.no_grad():
        indexer.wk.weight.copy_(rng.tensor(INDEX_HEAD_DIM, HIDDEN_SIZE))
        indexer.k_norm.weight.copy_(rng.tensor(INDEX_HEAD_DIM))
        indexer.k_norm.bias.copy_(rng.tensor(INDEX_HEAD_DIM))
        indexer.index_kpool_compress_gate.copy_(
            rng.tensor(INDEX_HEAD_DIM, HIDDEN_SIZE))
        indexer.index_kpool_compress_ape.copy_(
            rng.tensor(INDEX_KPOOL, INDEX_HEAD_DIM))
        indexer.wq_b.weight.copy_(
            rng.tensor(INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK))
        indexer.weights_proj.weight.copy_(rng.tensor(INDEX_N_HEADS, HIDDEN_SIZE))
        hidden = rng.tensor(1, SEQUENCE, HIDDEN_SIZE)
        q_resid = rng.tensor(1, SEQUENCE, Q_LORA_RANK)
        mask = torch.ones(1, SEQUENCE, dtype=torch.bool)
        topk = indexer(hidden_states=hidden, q_resid=q_resid,
                       attention_mask=mask, past_key_values=None)

    margin = selection_margin(indexer, hidden, q_resid, mask)
    probes = []
    for row in PROBE_ROWS:
        chosen = sorted(int(value) for value in topk[0, row].tolist() if value >= 0)
        assert len(set(chosen)) == len(chosen), f"row {row} selected a duplicate"
        probes.append({"row": row, "count": len(chosen), "selected": chosen})
    return {
        "reference": "experiments/references/glm5-next/modeling_glm5_next.py"
                     " :: Glm5NextTextIndexer",
        "seed": SEED,
        "hidden_size": HIDDEN_SIZE,
        "q_lora_rank": Q_LORA_RANK,
        "index_n_heads": INDEX_N_HEADS,
        "index_head_dim": INDEX_HEAD_DIM,
        "index_topk": INDEX_TOPK,
        "index_kpool": INDEX_KPOOL,
        "sequence": SEQUENCE,
        "selection_margin": margin,
        "probes": probes,
    }


def selection_margin(indexer, hidden, q_resid, mask) -> float:
    """Smallest score gap across the top-k boundary, over the probe rows.

    The fixture asserts an exact match against a scalar C++ implementation, so it
    is only meaningful if no probe row sits on a numerical knife edge.  A margin
    far above float32 matmul error means reassociation cannot reorder the cut.
    """
    with torch.no_grad():
        q = indexer.wq_b(q_resid).view(1, SEQUENCE, -1, INDEX_HEAD_DIM)
        k = indexer.k_norm(indexer.wk(hidden)).view(1, SEQUENCE, -1,
                                                    INDEX_HEAD_DIM).squeeze(2)
        gate = F.linear(hidden, indexer.index_kpool_compress_gate)
        packed = torch.cat([k, gate, mask.to(k.dtype)[..., None]], dim=-1)
        pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed)
        scores = F.relu(torch.matmul(q.float(),
                                     pool_keys.transpose(-1, -2).float().unsqueeze(1))
                        * indexer.softmax_scale)
        weights = indexer.weights_proj(hidden).float() * (INDEX_N_HEADS ** -0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        visible = indexer.get_visible_tokens(packed[..., -1].bool(), SEQUENCE,
                                             SEQUENCE)
        pool_end = pool_indices[..., -1].clamp(0, SEQUENCE - 1)
        candidate = visible.gather(-1, pool_end[:, None, :].expand(1, SEQUENCE, -1))
        candidate = candidate & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~candidate, float("-inf"))
        keep = INDEX_TOPK // INDEX_KPOOL
        smallest = float("inf")
        for row in PROBE_ROWS:
            values = index_scores[0, row]
            live = values[torch.isfinite(values)]
            if live.numel() <= keep:
                continue  # every candidate is kept; there is no boundary
            ordered = live.sort(descending=True).values
            smallest = min(smallest, float(ordered[keep - 1] - ordered[keep]))
        return smallest if smallest != float("inf") else -1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "tests" / "fixtures" / "glm53" /
                        "indexer-oracle.json")
    arguments = parser.parse_args()
    fixture = build()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"{arguments.out}: {len(fixture['probes'])} probe rows, "
          f"selection margin {fixture['selection_margin']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
