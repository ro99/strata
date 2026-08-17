"""Recorded exceptions to Phase 1's layering enforcement (docs/experiments/0121).

Every entry here is a *deferred* violation, not an accepted one: `check_layers.py`
still finds the underlying `#include`, still classifies it as a real layering
violation, and only then treats it as forgiven because it appears in one of the
`matches` lists below. Two properties this file's own consumer enforces, not
just documents:

- If a listed `(file, included_header)` pair stops being a real violation --
  the include is removed, the header moves, whatever -- `check_layers.py`
  treats that as a hard error, not a silent pass. A stale exception is how a
  boundary quietly reopens; the check must notice the boundary was already
  crossed by something else, not assume the exception is still needed.
- `check_layers.py` always prints every exception it applied, by name, with
  the violations it covered. Nothing here is invisible to a normal run.

Three entries, all with the same expiry: phase 4, when each model's own
fan-out work is already touching the file in question, and the underlying fix
in every case needs a registry or a generalized contract designed once with
all six models in view -- not designed now against one or two and stretched
to fit the rest later. See docs/experiments/0121 for the full reasoning.
"""

from __future__ import annotations

EXCEPTIONS = [
    {
        "name": "quantization-contract-in-model-hpp",
        "reason": (
            "compressed_tensors.hpp needs QuantizedWeightSpec and "
            "QuantizationGranularity, which are model-neutral quantization "
            "vocabulary bundled inside model.hpp alongside the genuinely "
            "model-specific Spec structs. The intended fix is the same "
            "model-neutral contract extraction that would let "
            "GlmManifestTensor (checkpoint.hpp's former coupling, resolved "
            "in this unit by reassigning checkpoint.* to strata_models "
            "outright) extend a generic ManifestTensor/IndexManifest base "
            "instead of being the base. Not designed against one caller "
            "today; wants all six models' actual shapes in view."
        ),
        "expiry_phase": 4,
        "matches": [
            ("include/strata/compressed_tensors.hpp", "model.hpp"),
        ],
    },
    {
        "name": "placement-model-inventory-registry",
        "reason": (
            "placement_model.cpp's six build_<model>_inventory functions "
            "(~72% of the file) belong with each model, not inlined in an "
            "engine-tier dispatcher. The solver itself (plan_model_placement) "
            "is already close to a pure function of inventory plus hardware "
            "-- PlacementInventory is the right shape -- but extracting the "
            "builders cleanly needs build_inventory's dispatch to become a "
            "registry each model populates, which is a design decision, not "
            "a mechanical split. Phase 4 is when each model's own code is "
            "already open for the oracle fan-out; the registry gets designed "
            "once, with all six builders in view, instead of once now for "
            "two models and stretched to fit four more later."
        ),
        "expiry_phase": 4,
        "matches": [
            ("src/placement_model.cpp", "checkpoint.hpp"),
            ("src/placement_model.cpp", "deepseek_admission.hpp"),
            ("src/placement_model.cpp", "deepseek_checkpoint.hpp"),
            ("src/placement_model.cpp", "gemma4_checkpoint.hpp"),
            ("src/placement_model.cpp", "kimi_k3_checkpoint.hpp"),
            ("src/placement_model.cpp", "inkling_checkpoint.hpp"),
            ("src/placement_model.cpp", "laguna_checkpoint.hpp"),
            ("src/placement_model.cpp", "model_adapter.hpp"),
        ],
    },
    {
        "name": "tokenizer-pretokenizer-split",
        "reason": (
            "tokenizer.cpp inlines two models' generated Unicode category "
            "tables (inkling_unicode.hpp, laguna_unicode.hpp) and their "
            "specific pretokenize functions directly into the shared "
            "ModelTokenizer dispatch -- the same shape as "
            "placement-model-inventory-registry, one level down. "
            "Pretokenization is exactly the kind of per-model logic that "
            "belongs with each model's own fan-out work at phase 4, not "
            "split out ad hoc for two of six models now."
        ),
        "expiry_phase": 4,
        "matches": [
            ("src/tokenizer.cpp", "inkling_unicode.hpp"),
            ("src/tokenizer.cpp", "laguna_unicode.hpp"),
        ],
    },
]
