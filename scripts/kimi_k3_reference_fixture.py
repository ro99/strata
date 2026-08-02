#!/usr/bin/env python3
"""Emit layer-level Kimi-K3 fixtures by running the checkpoint's own reference.

This is a test oracle, not runtime code. It imports `modeling_kimi_linear.py`
straight out of `/data/kimi-k3` and executes the real `KimiDeltaAttention` and
`KimiMLAAttention` modules with real weights, so the fixture records what the
reference does rather than what a transcription of it would do. A single layer
is under a gigabyte, which is why this fits on one GPU while the model does not.

Nothing here writes to an NVMe: weights are read from `/data` (SATA), the
fixture is written back to the same disk, and TMPDIR / the Triton and CUDA
caches are pointed at tmpfs by `run_kimi_k3_reference_fixture.sh`. The fixture
holds reference activations and is therefore derived from model weights, so
`results/` in the working tree is not an admissible destination: it sits on
`/dev/nvme0n1p2`.

Usage:
    kimi_k3_reference_fixture.py --model /data/kimi-k3 \
        --out /data/strata-results/kimi-k3-fixtures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"KMFX"
VERSION = 1


FORBIDDEN_DISKS = ("nvme0n1", "sdb")


def refuse_forbidden_disk(directory: Path) -> None:
    """Refuse to write model-derived bytes onto a disk the operator protects.

    Resolved through `/sys/dev/block/MAJ:MIN` the same way the runtime's guard
    does, so the two agree on what "on the NVMe" means. A major of zero is a
    memory-backed filesystem: resolved, and safe.
    """
    device = os.stat(directory).st_dev
    major, minor = os.major(device), os.minor(device)
    if major == 0:
        return
    link = Path(f"/sys/dev/block/{major}:{minor}")
    if not link.exists():
        raise RuntimeError(
            f"{directory}: cannot resolve backing device {major}:{minor}; "
            "refusing to write rather than guessing it is safe")
    # `.../nvme0n1/nvme0n1p2` -> the whole disk is the parent when a partition
    # directory carries a `partition` file.
    resolved = link.resolve()
    disk = resolved.parent.name if (resolved / "partition").exists() else resolved.name
    if disk in FORBIDDEN_DISKS:
        raise RuntimeError(
            f"{directory} is backed by /dev/{disk}, which may not receive bytes "
            "derived from model weights; pass --out on /data or a tmpfs")


class FixtureWriter:
    """A flat sequence of named float32 arrays with their shapes.

    Deliberately trivial: the C++ side reads it in thirty lines and there is no
    schema to drift. Every array is float32 even when the reference computed in
    bfloat16, because the comparison happens in float32 either way.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, list[int], torch.Tensor]] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        flat = tensor.detach().to(torch.float32).cpu().contiguous()
        self._entries.append((name, list(tensor.shape), flat.reshape(-1)))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        refuse_forbidden_disk(path.parent)
        with path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<II", VERSION, len(self._entries)))
            for name, shape, flat in self._entries:
                encoded = name.encode("utf-8")
                handle.write(struct.pack("<I", len(encoded)))
                handle.write(encoded)
                handle.write(struct.pack("<I", len(shape)))
                for extent in shape:
                    handle.write(struct.pack("<Q", extent))
                handle.write(struct.pack("<Q", flat.numel()))
                handle.write(flat.numpy().tobytes())


def load_reference(model_directory: Path):
    """Import the checkpoint's own modeling code as a package.

    `modeling_kimi_linear.py` uses a relative import for its config, so the two
    files have to be reachable as one package. Linking them into a scratch
    package keeps the checkpoint directory untouched.
    """
    package = Path(os.environ.get("KIMI_REFERENCE_PACKAGE", "/dev/shm/kimi_reference"))
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    for name in ("modeling_kimi_linear.py", "configuration_kimi_k3.py"):
        link = package / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(model_directory / name)
    sys.path.insert(0, str(package.parent))
    module_name = f"{package.name}.modeling_kimi_linear"
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"cannot import {module_name}")
    return importlib.import_module(module_name)


class ShardSet:
    """Positional reads out of the sharded checkpoint, one tensor at a time."""

    def __init__(self, model_directory: Path) -> None:
        self.directory = model_directory
        index = json.loads((model_directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers: dict[str, dict] = {}

    def _header(self, shard: str) -> dict:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        with (self.directory / shard).open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        header["__data_offset__"] = 8 + length
        self._headers[shard] = header
        return header

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        header = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        base = header["__data_offset__"]
        with (self.directory / shard).open("rb") as handle:
            handle.seek(base + begin)
            raw = handle.read(end - begin)
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32,
                 "F16": torch.float16, "U8": torch.uint8,
                 "I32": torch.int32, "I64": torch.int64}[entry["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])


def narrow_padded(name: str, tensor: torch.Tensor,
                  parameter: torch.Tensor) -> torch.Tensor:
    """Trim a zero-padded checkpoint tensor down to what the module declares.

    `A_log` ships as `[head_dim] = [128]` while it is a per-head scalar and only
    the first `num_heads = 96` entries are live. The reference's own loader
    narrows it the same way. The zero tail is asserted rather than assumed: if a
    padded slot ever held a value, trimming it would change the decay of a
    quarter of the heads and nothing downstream would notice.
    """
    if tensor.dim() != parameter.dim():
        raise RuntimeError(
            f"{name}: checkpoint {tuple(tensor.shape)} against module "
            f"{tuple(parameter.shape)}")
    for axis, (have, want) in enumerate(zip(tensor.shape, parameter.shape)):
        if have < want:
            raise RuntimeError(
                f"{name}: checkpoint {tuple(tensor.shape)} is smaller than "
                f"module {tuple(parameter.shape)} on axis {axis}")
        if have == want:
            continue
        tail = tensor.narrow(axis, want, have - want)
        if bool(tail.to(torch.float32).abs().max() > 0.0):
            raise RuntimeError(
                f"{name}: axis {axis} padding [{want}, {have}) is not zero; "
                "narrowing it would drop live values")
        tensor = tensor.narrow(axis, 0, want).contiguous()
    return tensor


def load_module_weights(module: torch.nn.Module, shards: ShardSet, prefix: str,
                        device: torch.device, dtype: torch.dtype) -> None:
    """Fill a reference module from the checkpoint and fail loudly on a miss.

    A silently unloaded parameter would leave random initialization in place and
    the fixture would then encode noise, which is exactly the kind of result the
    charter says to treat as a defect rather than a datapoint.
    """
    module.to(device=device, dtype=dtype)
    sources = {}
    for name, parameter in module.named_parameters():
        full = f"{prefix}.{name}"
        tensor = shards.get(full)
        if tuple(tensor.shape) != tuple(parameter.shape):
            tensor = narrow_padded(full, tensor, parameter)
        sources[name] = tensor
    missing = [name for name in dict(module.named_parameters()) if name not in sources]
    if missing:
        raise RuntimeError(f"{prefix}: no checkpoint tensor for {missing}")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            source = sources[name]
            # `A_log`, `dt_bias`, and the convolution kernels ship as float32 and
            # the reference keeps them there. Casting them into bfloat16 with the
            # rest of the module would quantize a decay parameter that the whole
            # recurrence is exponentially sensitive to.
            target = torch.float32 if source.dtype == torch.float32 else dtype
            parameter.data = source.to(device=device, dtype=target)


def deterministic_input(tokens: int, hidden: int, seed: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(1, tokens, hidden, generator=generator, dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def emit_kda(reference, shards: ShardSet, config, layer: int, tokens: int,
             device: torch.device, dtype: torch.dtype, writer: FixtureWriter) -> None:
    prefix = f"language_model.model.layers.{layer}.self_attn"
    module = reference.KimiDeltaAttention(config=config, layer_idx=layer)
    load_module_weights(module, shards, prefix, device, dtype)
    module.eval()

    hidden = deterministic_input(tokens, config.hidden_size, 4801, device, dtype)
    writer.add(f"kda.{layer}.input", hidden[0])

    # Prefill: no cache, so the reference takes its chunkwise path.
    with torch.no_grad():
        prefill = module(hidden_states=hidden, cache_params=None)
    writer.add(f"kda.{layer}.prefill_output", prefill[0])

    # Prefill again with a cache, then one more token through the recurrent
    # path. Decode and prefill must agree on the state they hand each other, so
    # the fixture records both halves of that handoff.
    cache = reference.KimiDynamicCache(config)
    with torch.no_grad():
        _ = module(hidden_states=hidden, cache_params=cache)
    state = cache.recurrent_states[layer]
    writer.add(f"kda.{layer}.prefill_state", state[0])
    conv_q, conv_k, conv_v = cache.conv_states[layer]
    writer.add(f"kda.{layer}.prefill_conv_q", conv_q[0])
    writer.add(f"kda.{layer}.prefill_conv_k", conv_k[0])
    writer.add(f"kda.{layer}.prefill_conv_v", conv_v[0])

    step = deterministic_input(1, config.hidden_size, 4802, device, dtype)
    writer.add(f"kda.{layer}.decode_input", step[0])
    with torch.no_grad():
        decode = module(hidden_states=step, cache_params=cache)
    writer.add(f"kda.{layer}.decode_output", decode[0])
    writer.add(f"kda.{layer}.decode_state", cache.recurrent_states[layer][0])
    del module
    torch.cuda.empty_cache()


def emit_mla(reference, shards: ShardSet, config, layer: int, tokens: int,
             device: torch.device, dtype: torch.dtype, writer: FixtureWriter) -> None:
    prefix = f"language_model.model.layers.{layer}.self_attn"
    module = reference.KimiMLAAttention(config=config, layer_idx=layer)
    load_module_weights(module, shards, prefix, device, dtype)
    module.eval()

    hidden = deterministic_input(tokens, config.hidden_size, 4803, device, dtype)
    writer.add(f"mla.{layer}.input", hidden[0])

    # `eager_attention_forward` needs an additive mask; the causal one is what
    # the model builds for a full-attention layer.
    minimum = torch.finfo(torch.float32).min
    mask = torch.full((tokens, tokens), minimum, device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1).view(1, 1, tokens, tokens).to(dtype)

    cache = reference.KimiDynamicCache(config)
    with torch.no_grad():
        prefill = module(hidden_states=hidden, attention_mask=mask,
                         past_key_values=cache)
    writer.add(f"mla.{layer}.prefill_output", prefill[0])

    step = deterministic_input(1, config.hidden_size, 4804, device, dtype)
    writer.add(f"mla.{layer}.decode_input", step[0])
    decode_mask = torch.zeros((1, 1, 1, tokens + 1), device=device, dtype=dtype)
    with torch.no_grad():
        decode = module(hidden_states=step, attention_mask=decode_mask,
                        past_key_values=cache)
    writer.add(f"mla.{layer}.decode_output", decode[0])
    del module
    torch.cuda.empty_cache()


def assign_parameter(root: torch.nn.Module, dotted: str, tensor: torch.Tensor,
                     declared: torch.Tensor | None = None) -> None:
    """Replace a parameter on a module built under the meta device.

    A meta parameter cannot take `.data =` from a real tensor, so the Parameter
    object itself is replaced. The declared shape is checked when it is known:
    silently loading a mis-shaped tensor would leave the fixture encoding
    something other than the model.
    """
    if declared is not None and tuple(declared.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"{dotted}: checkpoint {tuple(tensor.shape)} against module "
            f"{tuple(declared.shape)}")
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1],
            torch.nn.Parameter(tensor, requires_grad=False))


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor, rows: int,
                     columns: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """Decode one MXFP4 module using compressed-tensors' own routines.

    The nibble order, sign bit, magnitude table, and E8M0 bias all come from the
    library rather than from a transcription of the format, for the same reason
    the layer fixtures run the real modules.
    """
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

    values = unpack_fp4_from_uint8(packed.to(device), rows, columns,
                                   dtype=torch.float32)
    scale = decompress_mx_scale(scales.to(device)).to(torch.float32)
    # One scale per group of 32 along the input axis.
    grouped = values.reshape(rows, columns // 32, 32)
    return (grouped * scale.reshape(rows, columns // 32, 1)).reshape(
        rows, columns).to(dtype)


def emit_moe(reference, shards: ShardSet, config, layer: int, tokens: int,
             device: torch.device, dtype: torch.dtype,
             writer: FixtureWriter) -> None:
    """Run the real Stable LatentMoE block with real MXFP4 experts.

    All 896 experts are built on the meta device so nothing is allocated for
    them, the reference's own gate picks the top-k, and only the selected
    experts are materialized. `moe_infer` never touches an expert with zero
    assigned tokens, so the unselected meta modules are never read.
    """
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
    with torch.device("meta"):
        block = reference.KimiSparseMoeBlock(config)
    # The reference's gate asserts inference mode outright.
    block.eval()

    dense = {
        "gate.weight": f"{prefix}.gate.weight",
        "gate.e_score_correction_bias": f"{prefix}.gate.e_score_correction_bias",
        "routed_expert_down_proj.weight": f"{prefix}.routed_expert_down_proj.weight",
        "routed_expert_up_proj.weight": f"{prefix}.routed_expert_up_proj.weight",
        "routed_expert_norm.weight": f"{prefix}.routed_expert_norm.weight",
        "shared_experts.gate_proj.weight": f"{prefix}.shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight": f"{prefix}.shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight": f"{prefix}.shared_experts.down_proj.weight",
    }
    parameters = dict(block.named_parameters())
    for local, full in dense.items():
        source = shards.get(full)
        target = torch.float32 if source.dtype == torch.float32 else dtype
        assign_parameter(block, local, source.to(device=device, dtype=target),
                         parameters[local])

    hidden = deterministic_input(tokens, config.hidden_size, 4805, device, dtype)
    writer.add(f"moe.{layer}.input", hidden[0])

    with torch.no_grad():
        topk_idx, _ = block.gate(hidden)
    selected = sorted({int(index) for index in topk_idx.reshape(-1).tolist()})
    print(f"moe layer {layer}: {len(selected)} distinct experts over {tokens} tokens")

    latent = config.routed_expert_hidden_size
    inner = config.moe_intermediate_size
    shapes = {"w1": (inner, latent), "w3": (inner, latent), "w2": (latent, inner)}
    with torch.no_grad():
        for expert in selected:
            for module, (rows, columns) in shapes.items():
                packed = shards.get(f"{prefix}.experts.{expert}.{module}.weight_packed")
                scales = shards.get(f"{prefix}.experts.{expert}.{module}.weight_scale")
                dense_weight = dequantize_mxfp4(packed, scales, rows, columns,
                                                device, dtype)
                assign_parameter(block, f"experts.{expert}.{module}.weight",
                                 dense_weight)

    with torch.no_grad():
        output = block(hidden)
    writer.add(f"moe.{layer}.output", output[0])
    writer.add(f"moe.{layer}.experts",
               torch.tensor(selected, dtype=torch.float32))
    del block
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/kimi-k3", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("/data/strata-results/kimi-k3-fixtures"))
    parser.add_argument("--tokens", type=int, default=64,
                        help="prefill length; the reference chunks at 64")
    parser.add_argument("--kda-layer", type=int, default=0)
    parser.add_argument("--mla-layer", type=int, default=3)
    parser.add_argument("--moe-layer", type=int, default=1)
    parser.add_argument("--moe-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    arguments = parser.parse_args()

    reference = load_reference(arguments.model)
    text_config = json.loads((arguments.model / "config.json").read_text())["text_config"]
    config = reference.KimiLinearConfig(**text_config)
    # The fixture compares numerics, not kernel dispatch; eager attention is the
    # reference's own definition and needs no flash-attention build.
    config._attn_implementation = "eager"

    shards = ShardSet(arguments.model)
    device = torch.device(arguments.device)
    dtype = torch.bfloat16

    writer = FixtureWriter()
    emit_kda(reference, shards, config, arguments.kda_layer, arguments.tokens,
             device, dtype, writer)
    emit_mla(reference, shards, config, arguments.mla_layer, arguments.tokens,
             device, dtype, writer)
    # The MoE fixture uses fewer tokens on purpose: at 64 tokens the top-16
    # router would select close to all 896 experts of the layer, and
    # materializing them is 15 GiB. Four tokens keep it inside one card while
    # still exercising several tokens sharing and not sharing experts.
    emit_moe(reference, shards, config, arguments.moe_layer, arguments.moe_tokens,
             device, dtype, writer)

    destination = arguments.out / "kimi-k3-layers.fixture"
    writer.write(destination)
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
