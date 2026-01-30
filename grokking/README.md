# Grokking (minimal, self-contained)

This folder adds a **minimal modular-addition grokking** experiment, inspired by the standard setup used in the grokking literature and the reference implementation in [`danielmamay/grokking`](https://github.com/danielmamay/grokking).

## What this runs

- **Task**: modular addition \((a + b) \bmod p\), with \(a,b \in \{0,\dots,p-1\}\).
- **Model**: small Transformer encoder over 2 tokens (the pair \((a,b)\)).
- **Train/Test split**: train on a random subset of pairs; test on the complement.
- **Typical grokking behavior**: training accuracy becomes high early; test accuracy improves *much later* (often requiring weight decay + long training).

## Run

From repo root:

```bash
python /egr/research-slim/ghoshavr/complexity_measures_study/grokking/run_modular_addition.py
```

Outputs are saved under:

- `complexity_measures_study/RESULTS/grokking/.../metrics.pt`
- `complexity_measures_study/RESULTS/grokking/.../metrics.png`

## Tips

- To more reliably see grokking, try:
  - smaller `--train_frac` (e.g. `0.2` or `0.1`) instead of the default `0.8`
  - nonzero `--weight_decay` (e.g. `1e-2` to `1e-1`)
  - large `--steps` (e.g. `100000`+)


