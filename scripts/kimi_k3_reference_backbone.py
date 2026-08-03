#!/usr/bin/env python3
"""Emit the Kimi-K3 backbone fixture: every layer of a prompt page and a decode step.

Gates 5 and 6. Running the whole 1.45 TiB model in torch is not possible on this
machine — 64 GiB of VRAM across three cards, 245 GiB of RAM, and disk offload is
forbidden because it would write model bytes to the NVMe. So the reference runs
**one layer at a time**: each `KimiDecoderLayer` is built with real weights,
called, and freed, with the hidden state and block residual threaded between them
exactly as `KimiLinearModel.forward` does.

That is a stronger gate than a single end-to-end comparison, not a weaker one.
Comparing at every layer says *where* divergence enters and how it accumulates
across depth, which is the number that decides whether 93 layers of F32-versus-
BF16 rounding stay bounded.

Two passes, because prefill and decode are different code on both sides:

  prompt   the whole page at once — chunkwise KDA, causal MLA over the page
  decode   one token against the cache the prompt left — recurrent KDA, MLA
           against committed rows. The token is the reference's own greedy
           choice, so this is a generation oracle and not a second teacher
           forcing pass.

The only part of the reference not executed verbatim is the loop that threads
`block_residual` between layers, reproduced below from `KimiLinearModel.forward`:

    block_residual = hidden_states.new_zeros(B * T, 0, H)
    for decoder_layer in self.layers:
        layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
        hidden_states, block_residual = decoder_layer(
            hidden_states, attention_mask=layer_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            block_residual=block_residual)
    hidden_states = self._apply_output_attn_res(hidden_states, block_residual)
    hidden_states = self.norm(hidden_states)

Note that `block_residual` is created inside `forward`, so it does not survive
from the prompt page into the decode step; only the KV and recurrent state do.
Every layer, both masks, the output attention-residual mix, the final norm, and
the head are the reference's own code, called rather than transcribed.

Nothing here writes to an NVMe. See `kimi_k3_reference_fixture.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from kimi_k3_reference_fixture import (
    FixtureWriter,
    ShardSet,
    assign_parameter,
    dequantize_mxfp4,
    load_reference,
    narrow_padded,
)

# Queue depth for checkpoint reads. The arena benchmark in `docs/experiments/0048`
# measured this disk saturating at QD >= 4 (406 MB/s) against 167 MB/s at QD1.
READ_QUEUE_DEPTH = 8


def load_into(module: torch.nn.Module, shards: ShardSet, prefix: str,
              device: torch.device, dtype: torch.dtype,
              skip: tuple[str, ...] = ()) -> None:
    """Fill a meta-device module from the checkpoint, failing loudly on a miss.

    Read at queue depth for the same reason `materialize_experts` is: a decoder
    layer's dense half is about 1.1 GiB spread over some fifty tensors, and this
    disk needs more than one outstanding request to reach its rated throughput.
    """
    wanted = [name for name, _ in module.named_parameters()
              if not any(name.startswith(prefix_) for prefix_ in skip)]
    with ThreadPoolExecutor(max_workers=READ_QUEUE_DEPTH) as pool:
        staged = dict(pool.map(
            lambda name: (name, shards.get(f"{prefix}.{name}")), wanted))

    for name, parameter in list(module.named_parameters()):
        tensor = staged.get(name)
        if tensor is None:
            continue
        if tuple(tensor.shape) != tuple(parameter.shape):
            tensor = narrow_padded(f"{prefix}.{name}", tensor, parameter)
        # `A_log`, `dt_bias`, and the convolution kernels ship as float32 and the
        # reference keeps them there; casting a decay parameter the recurrence
        # is exponentially sensitive to would be a silent precision change.
        target = torch.float32 if tensor.dtype == torch.float32 else dtype
        assign_parameter(module, name, tensor.to(device=device, dtype=target))


def materialize_experts(block: torch.nn.Module, shards: ShardSet, prefix: str,
                        experts: list[int], config, device: torch.device,
                        dtype: torch.dtype) -> None:
    """Stage the routed experts of one layer, reading at queue depth.

    Measured first, then fixed: at one read per tensor this ran at 88 MB/s on a
    disk the arena benchmark clocks at 178-400 MB/s, with the GPUs idle and a
    fifth of one core busy. An order-of-magnitude gap against the rated figure
    is a serialization defect, not a bandwidth limit — six reads per expert at
    queue depth one, and the smaller of each pair is only 344 KiB, so per-read
    latency dominated. An expert's six tensors are one unbroken 17,547,264-byte
    run, so they coalesce into a single read, and `pread` drops the GIL, so a
    small pool gives the disk the queue depth it wants. Dequantization stays on
    the calling thread: it is CUDA work and was never the constraint.
    """
    latent = config.routed_expert_hidden_size
    inner = config.moe_intermediate_size
    shapes = {"w1": (inner, latent), "w3": (inner, latent), "w2": (latent, inner)}

    def read(expert: int) -> tuple[int, dict[str, torch.Tensor]]:
        names = [f"{prefix}.experts.{expert}.{module}.{part}"
                 for module in ("w1", "w2", "w3")
                 for part in ("weight_packed", "weight_scale")]
        return expert, shards.coalesced(names)

    with ThreadPoolExecutor(max_workers=READ_QUEUE_DEPTH) as pool:
        for expert, tensors in pool.map(read, experts):
            for module, (rows, columns) in shapes.items():
                base = f"{prefix}.experts.{expert}.{module}"
                assign_parameter(
                    block, f"experts.{expert}.{module}.weight",
                    dequantize_mxfp4(tensors[f"{base}.weight_packed"],
                                     tensors[f"{base}.weight_scale"],
                                     rows, columns, device, dtype))


def attach_expert_loader(block: torch.nn.Module, shards: ShardSet, prefix: str,
                         config, device: torch.device, dtype: torch.dtype,
                         selected: list[int]):
    """Materialize only the experts this page routes to, from the real gate input.

    896 experts per layer is 15 GiB dequantized, so they cannot be built eagerly
    on a 24 GiB card. The routed set can only be known once the layer has run its
    attention: the gate reads the post-attention normed hidden state, not the
    layer input. A pre-hook is the one place that state exists, which is why the
    selection happens here rather than before the layer is called — picking
    experts from the layer input would silently route the fixture differently
    from the model. `moe_infer` skips any expert with zero assigned tokens, so
    the rest stay on the meta device, unread and untouched.
    """

    def hook(module: torch.nn.Module, args: tuple) -> None:
        with torch.no_grad():
            topk_idx, _ = module.gate(args[0])
        chosen = sorted({int(index) for index in topk_idx.reshape(-1).tolist()})
        materialize_experts(module, shards, prefix, chosen, config, device, dtype)
        selected.extend(chosen)

    return block.register_forward_pre_hook(hook)


def build_layer(reference, shards: ShardSet, config, layer: int,
                device: torch.device, dtype: torch.dtype):
    """One real decoder layer with real weights and a lazy expert loader."""
    with torch.device("meta"):
        module = reference.KimiDecoderLayer(config, layer)
    module.eval()
    prefix = f"language_model.model.layers.{layer}"
    load_into(module, shards, prefix, device, dtype,
              skip=("block_sparse_moe.experts.",))
    selected: list[int] = []
    if hasattr(module, "block_sparse_moe"):
        attach_expert_loader(module.block_sparse_moe, shards,
                             f"{prefix}.block_sparse_moe", config, device, dtype,
                             selected)
    return module, selected


class Tail:
    """The output attention-residual site, the final norm, and the head.

    `KimiLinearModel._apply_output_attn_res` is four lines of reshaping around
    `_apply_attn_res`; the mix itself, both norms and the projection are the
    reference's own. Held across passes so the 2.19 GiB head is read once.
    """

    def __init__(self, reference, shards: ShardSet, config,
                 device: torch.device, dtype: torch.dtype) -> None:
        self.reference = reference
        self.hidden_size = config.hidden_size
        with torch.device("meta"):
            self.res_norm = reference.KimiRMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)
            self.res_proj = torch.nn.Linear(config.hidden_size, 1, bias=False)
            self.final_norm = reference.KimiRMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)
        for module, name in ((self.res_norm, "output_attn_res_norm.weight"),
                             (self.res_proj, "output_attn_res_proj.weight"),
                             (self.final_norm, "norm.weight")):
            tensor = shards.get(f"language_model.model.{name}")
            assign_parameter(module, "weight",
                             tensor.to(device=device, dtype=dtype))
        self.head = shards.get("language_model.lm_head.weight").to(
            device=device, dtype=dtype)

    def __call__(self, hidden: torch.Tensor, block_residual: torch.Tensor):
        tokens = hidden.shape[1]
        with torch.no_grad():
            mixed = self.reference._apply_attn_res(
                hidden.view(-1, self.hidden_size), block_residual, self.res_proj,
                self.res_norm).view(1, tokens, self.hidden_size)
            final = self.final_norm(mixed)
            logits = torch.nn.functional.linear(final[0, -1], self.head)
        return final, logits


def sweep(reference, shards: ShardSet, config, tail: Tail, cache, hidden,
          position: int, depth: int, device: torch.device, dtype: torch.dtype,
          writer: FixtureWriter, tag: str):
    """One `KimiLinearModel.forward` over a page, recorded at every layer."""
    tokens = hidden.shape[1]
    cache_position = torch.arange(position, position + tokens, device=device)
    causal_mask = reference.create_causal_mask(
        config=config, input_embeds=hidden, attention_mask=None,
        cache_position=cache_position, past_key_values=cache,
        position_ids=cache_position.unsqueeze(0))
    # Unbound on purpose: the method ignores `self`, so this is the reference's
    # own mask rule rather than a copy of it. With no padding it returns None.
    linear_mask = reference.KimiLinearModel._update_linear_attn_mask(
        None, None, cache_position)
    block_residual = hidden.new_zeros(hidden.shape[0] * tokens, 0,
                                      hidden.shape[2])

    started = time.time()
    for layer in range(depth):
        module, selected = build_layer(reference, shards, config, layer, device,
                                       dtype)
        mask = linear_mask if module.is_linear_attn else causal_mask
        with torch.no_grad():
            hidden, block_residual = module(hidden, attention_mask=mask,
                                            past_key_values=cache,
                                            cache_position=cache_position,
                                            block_residual=block_residual)
        writer.add(f"{tag}.layer.{layer}", hidden[0])
        del module
        torch.cuda.empty_cache()
        print(f"{tag} layer {layer + 1:3d}/{config.num_hidden_layers}  "
              f"{'kda' if mask is linear_mask else 'mla'}  "
              f"{len(selected):4d} experts  "
              f"{(time.time() - started) / 60.0:6.1f} min", flush=True)

    if depth != config.num_hidden_layers:
        return None, None
    final, logits = tail(hidden, block_residual)
    writer.add(f"{tag}.final_hidden", final[0])
    writer.add(f"{tag}.last_logits", logits)
    top = torch.topk(logits.float(), 8)
    print(f"{tag} top-8 next tokens:",
          [(int(index), round(float(value), 4))
           for index, value in zip(top.indices, top.values)], flush=True)
    return final, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/kimi-k3", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("/data/strata-results/kimi-k3-fixtures"))
    # Four tokens: enough to exercise causal attention across positions, the
    # chunkwise KDA path, and tokens that share and do not share experts, while
    # keeping the routed set inside one card. Every extra token widens the set
    # the reference must read and dequantize, which is the whole cost of the run.
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    # Smoke-test knobs. A truncated backbone is not a gate fixture, so a
    # truncated run refuses to write one rather than leaving a plausible file
    # that would pass a shallow check and fail the claim it is meant to make.
    parser.add_argument("--layers", type=int, default=0,
                        help="stop each pass after N layers, write nothing (0 = all)")
    parser.add_argument("--decode-token", type=int, default=0,
                        help="decode this id instead of the greedy choice; "
                             "only for truncated smoke runs")
    parser.add_argument("--no-decode", action="store_true",
                        help="prompt page only, gate 5 without gate 6")
    arguments = parser.parse_args()

    reference = load_reference(arguments.model)
    text_config = json.loads((arguments.model / "config.json").read_text())["text_config"]
    config = reference.KimiLinearConfig(**text_config)
    config._attn_implementation = "eager"

    shards = ShardSet(arguments.model)
    device = torch.device(arguments.device)
    dtype = torch.bfloat16
    writer = FixtureWriter()
    depth = arguments.layers or config.num_hidden_layers
    truncated = depth != config.num_hidden_layers

    tokens = arguments.tokens
    # Fixed token ids rather than random embeddings: the runtime looks the same
    # ids up in the same embedding table, so both sides start from identical
    # bits and the comparison measures the graph, not the input.
    ids = [1234, 5678, 91011, 121314]
    while len(ids) < tokens:
        ids.append(ids[len(ids) % 4] + len(ids))
    ids = ids[:tokens]
    writer.add("prompt.token_ids", torch.tensor(ids, dtype=torch.float32))

    embedding = shards.get("language_model.model.embed_tokens.weight")

    def embed(sequence: list[int]) -> torch.Tensor:
        rows = torch.stack([embedding[index] for index in sequence])
        return rows.unsqueeze(0).to(device=device, dtype=dtype)

    hidden = embed(ids)
    writer.add("prompt.embeddings", hidden[0])

    tail = Tail(reference, shards, config, device, dtype)
    cache = reference.KimiDynamicCache(config)

    started = time.time()
    _, logits = sweep(reference, shards, config, tail, cache, hidden, 0, depth,
                      device, dtype, writer, "prompt")

    if not arguments.no_decode:
        if logits is not None:
            # Greedy: the reference consumes its own choice, which is what makes
            # the second pass a generation oracle rather than teacher forcing.
            next_token = int(torch.argmax(logits.float()))
        elif arguments.decode_token:
            next_token = arguments.decode_token
        else:
            print("truncated run and no --decode-token: skipping the decode pass")
            next_token = None
        if next_token is not None:
            writer.add("decode.token_id",
                       torch.tensor([next_token], dtype=torch.float32))
            sweep(reference, shards, config, tail, cache, embed([next_token]),
                  tokens, depth, device, dtype, writer, "decode")

    if truncated:
        print(f"smoke run: stopped after {depth} layers, no fixture written")
        return 0

    destination = arguments.out / "kimi-k3-backbone.fixture"
    writer.write(destination)
    print(f"wrote {destination} ({destination.stat().st_size} bytes) in "
          f"{(time.time() - started) / 60.0:.1f} min")

    # Attributed to this process, because the disk-level counter is not.
    # `/sys/block/nvme0n1/stat` moves for anything on the machine: the first
    # full run reported 94 KiB/s against a 29 KiB/s idle control and failed its
    # gate, and the excess was a concurrent `cmake --build` writing object files
    # into the working tree, which is on that disk. This number is the oracle's
    # own, and the only file it opens for writing is the fixture, whose
    # directory `refuse_forbidden_disk` has already resolved off the NVMe.
    written = 0
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("write_bytes:"):
            written = int(line.split()[1])
    print(f"process write_bytes: {written} ({written / (1 << 20):.1f} MiB), "
          f"fixture {destination.stat().st_size / (1 << 20):.1f} MiB on "
          f"{destination.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
