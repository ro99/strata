# Model runbooks

These are operator-facing, copy-paste guides for running each model on its
fastest measured Strata path. Every speed is tied to an operating point and an
explicit measurement description; it is not a promise that the same number
transfers to a different context length or machine.

| Model | Runbook | Fastest accepted single-stream decode |
|---|---|---:|
| Gemma 4 31B-IT | [Build, chat, serve, and benchmark](gemma4.md) | 29.747 tok/s; 128-token prefill 668.99 tok/s |
| Laguna S 2.1 | [Build, chat, serve, and benchmark](laguna.md) | 18.626 tok/s steady-state median |
| Inkling Small | [Build, chat, serve, and benchmark](inkling.md) | 9.072 tok/s fresh / 28.010 tok/s same-route warm |
| DeepSeek V4 | [Build, serve, and benchmark](deepseek.md) | 9.171 tok/s with the routed-expert tier, 8.571 without |
| GLM-5.2 | [Build, preflight, and run](glm52.md) | I/O-dependent on the reference workstation |
| GLM-5.3-Flash | [Build, preflight, chat, and serve](glm53.md) | I/O-dependent; exact text context currently capped at 2,048 tokens |
| Kimi-K3 | [Build, preflight, and run](kimi-k3.md) | ~0.02 tok/s; SATA-bound on the reference workstation |
