# complexity_measures_study

Minimal code to study optimization dynamics and **complexity / sharpness measures** for neural networks, plus a small **grokking** toy experiment.

## Complexity measures (what we log / plot)

Let $L(\theta)$ be the (empirical) training loss, $g=\nabla_\theta L(\theta)$, and $H=\nabla_\theta^2 L(\theta)$.

- **Sharpness** (spectral norm / top Hessian eigenvalue):

$$
\text{sharpness} \;\equiv\; \lambda_{\max}(H).
$$
  - Implemented via Hessian–vector products + Lanczos (`src/utilities.py:get_hessian_eigenvalues`).
  - In grokking we estimate \(\lambda_{\max}(H)\) by power iteration on HVPs.

- **Trace of Hessian**:

$$
\operatorname{tr}(H)\;=\;\sum_i \lambda_i(H).
$$

Estimated with Hutchinson’s estimator:

$$
\operatorname{tr}(H)\;\approx\;\frac{1}{K}\sum_{k=1}^K v_k^\top H v_k,\quad v_k\sim \mathcal{N}(0,I).
$$
  - Implemented in `src/utilities.py:get_trace`.

- **Second-order loss expansion (one step, with cubic correction)**:

$$
-\rho\, g^\top\!\left(I-\frac{\rho}{2}H\right)g \;+\; \frac{\rho^2\sigma^2}{2N_s}\operatorname{tr}(H^3),
\qquad \sigma^2=\frac{1}{h_0\,\mathrm{ess}}.
$$

This is implemented as:

$$
-\rho\|g\|^2+\frac{\rho^2}{2}g^\top Hg \;+\; \frac{\rho^2\sigma^2}{2N_s}\operatorname{tr}(H^3)
$$
  in `src/utilities.py:second_order_loss_expansion`.

## Repository layout

- **`grokking/`**: modular addition grokking toy task
  - `grokking/modular_addition.py`: dataset construction + train/test split
  - `grokking/model.py`: tiny Transformer encoder over the 2-token input \([a,b]\)
  - `grokking/run_modular_addition.py`: training loop + metrics plot (optionally sharpness)
- **`src/`**: shared utilities (datasets, architectures, Hessian/HVP tools, training loops)
  - `src/utilities.py`: HVP, sharpness (\(\lambda_{\max}\)), trace estimator, second-order expansion
  - `src/gd.py`: main training loop for CIFAR/MNIST/etc; logs loss/acc + complexity measures
- **`RESULTS/`**: experiment outputs (ignored by git)
- **`DATASETS/`**: dataset cache (ignored by git)

## Setup (recommended env vars)

Some scripts expect these environment variables:

```bash
export RESULTS=/egr/research-slim/ghoshavr/complexity_measures_study/RESULTS
export DATASETS=/egr/research-slim/ghoshavr/complexity_measures_study/DATASETS
```

## Run: grokking (modular addition)

Quick run (CPU):

```bash
python /egr/research-slim/ghoshavr/complexity_measures_study/run_grokking.py --device cpu
```

GPU run + sharpness logging (prints sharpness every `--sharpness_every` steps):

```bash
python /egr/research-slim/ghoshavr/complexity_measures_study/run_grokking.py \
  --device 7 --sharpness --sharpness_every 200 --steps 20000
```

Outputs go to:

- `RESULTS/grokking/.../metrics.pt`
- `RESULTS/grokking/.../metrics.png`

## Run: ResNet on CIFAR (GD loop + complexity measures)

Example: ResNet-32 on CIFAR-10 (10k training subset), compute Hessian measures every 200 steps and save periodically:

```bash
python /egr/research-slim/ghoshavr/complexity_measures_study/src/gd.py \
  --dataset cifar10-10k \
  --arch_id resnet32 \
  --loss ce \
  --opt gd \
  --lr 0.05 \
  --max_steps 2000 \
  --device_id 7 \
  --eig_freq 200 \
  --neigs 1 \
  --save_freq 200
```

Saved tensors include:
- `eigs` (top Hessian eigenvalue = sharpness when `--neigs 1`)
- `trace_hessian`
- `second_order_loss`
- `train_loss`, `test_loss`, `train_acc`, `test_acc`


