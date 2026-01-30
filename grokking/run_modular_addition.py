from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from modular_addition import make_split
from model import ModularAddTransformer

# Allow importing `src/utilities.py` when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utilities import compute_hvp  # noqa: E402


def _infinite_loader(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def _eval(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total)}


def _plot(metrics: Dict[str, List[float]], *, out_png: str, xlog: bool) -> None:
    steps = metrics["step"]
    has_sharpness = ("sharpness_step" in metrics) and (len(metrics["sharpness_step"]) > 0)
    ncols = 3 if has_sharpness else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    ax = axes[0]
    ax.plot(steps, metrics["train_loss"], color="0.35", linewidth=2.0, label="train")
    ax.plot(steps, metrics["test_loss"], color="0.0", linewidth=2.0, label="test")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-entropy")
    if xlog:
        ax.set_xscale("log")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(steps, metrics["train_acc"], color="0.35", linewidth=2.0, label="train")
    ax.plot(steps, metrics["test_acc"], color="0.0", linewidth=2.0, label="test")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    if xlog:
        ax.set_xscale("log")
    ax.legend(frameon=False)

    if has_sharpness:
        ax = axes[2]
        ax.plot(metrics["sharpness_step"], metrics["sharpness"], color="0.15", linewidth=2.0)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Sharpness (top Hessian eig.)")
        if xlog:
            ax.set_xscale("log")

    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


@torch.no_grad()
def _make_subset_dataset(ds: TensorDataset, *, n: int, seed: int) -> TensorDataset:
    x, y = ds.tensors
    n_total = int(x.shape[0])
    n_use = min(int(n), n_total)
    if x.is_cuda:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(n_total, generator=g, device=x.device)[:n_use]
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(n_total, generator=g)[:n_use]
    return TensorDataset(x[idx], y[idx])


def _estimate_top_hessian_eig(
    model: torch.nn.Module,
    *,
    dataset: TensorDataset,
    device: torch.device,
    physical_batch_size: int,
    power_iters: int,
    seed: int,
) -> float:
    """
    Estimate the top Hessian eigenvalue (sharpness) via power iteration on HVPs.
    Uses `compute_hvp` from `src/utilities.py`.
    """
    model.eval()

    # `compute_hvp` assumes "sum" reduction and internally divides by n.
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    n_params = int(parameters_to_vector(model.parameters()).numel())
    # Generator must match the tensor device (CUDA generator for CUDA tensors).
    if device.type == "cuda":
        g = torch.Generator(device=device)
    else:
        g = torch.Generator()
    g.manual_seed(int(seed))

    v = torch.randn(n_params, generator=g, device=device)
    v = v / (v.norm() + 1e-12)

    eig = 0.0
    dev_str = str(device)
    for _ in range(int(power_iters)):
        Hv = compute_hvp(model, loss_fn, dataset, v, dev_str, physical_batch_size=physical_batch_size).detach()
        eig = float(torch.dot(v, Hv).item())
        v = Hv / (Hv.norm() + 1e-12)
    return eig


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal grokking: modular addition (a+b mod p).")
    parser.add_argument("--p", type=int, default=97, help="modulus p (dataset size = p^2)")
    parser.add_argument("--train_frac", type=float, default=0.8, help="fraction of pairs used for training")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=20000, help="number of SGD/AdamW steps")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=10, help="evaluate train/test every N steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--xlog", action="store_true", help="log-scale x-axis in plots")
    parser.add_argument("--sharpness", action="store_true", help="log/plot sharpness (top Hessian eigenvalue) during training")
    parser.add_argument("--sharpness_every", type=int, default=200, help="compute sharpness every N steps (expensive)")
    parser.add_argument("--sharpness_n_examples", type=int, default=512, help="examples to use for sharpness estimate (subset of train)")
    parser.add_argument("--sharpness_power_iters", type=int, default=100, help="power-iteration steps for sharpness estimate")
    parser.add_argument("--sharpness_phys_bs", type=int, default=256, help="physical batch size for sharpness HVPs")

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RESULTS", "grokking"))
    parser.add_argument("--save_root", type=str, default=default_root)

    args = parser.parse_args()

    # Convenience: allow `--device 7` to mean `cuda:7` (also supports `cpu`, `cuda`, `cuda:0`, etc.)
    dev_str = args.device.strip()
    if dev_str.isdigit():
        dev_str = f"cuda:{dev_str}"

    device = torch.device(dev_str)
    torch.manual_seed(args.seed)

    split = make_split(p=args.p, train_frac=args.train_frac, seed=args.seed, device=device)
    train_dl = DataLoader(split.train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(split.test, batch_size=args.batch_size, shuffle=False)
    full_train_dl = DataLoader(split.train, batch_size=args.batch_size, shuffle=False)

    model = ModularAddTransformer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    run_name = f"modadd_p{args.p}_train{args.train_frac:g}_seed{args.seed}_wd{args.weight_decay:g}"
    out_dir = os.path.join(args.save_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics: Dict[str, List[float]] = {
        "step": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    if args.sharpness:
        metrics["sharpness_step"] = []
        metrics["sharpness"] = []

    train_iter = _infinite_loader(train_dl)

    # Step 0 eval
    m_train = _eval(model, full_train_dl, device)
    m_test = _eval(model, test_dl, device)
    metrics["step"].append(1.0)
    metrics["train_loss"].append(m_train["loss"])
    metrics["train_acc"].append(m_train["acc"])
    metrics["test_loss"].append(m_test["loss"])
    metrics["test_acc"].append(m_test["acc"])

    for step in range(1, args.steps + 1):
        model.train()
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Sharpness (optional): compute on its own schedule (independent of eval printing).
        sharp_val = None
        do_sharp = args.sharpness and (step % args.sharpness_every == 0 or step == args.steps)
        if do_sharp:
            print(f"step {step:>7d} | computing sharpness...", flush=True)
            subset = _make_subset_dataset(split.train, n=args.sharpness_n_examples, seed=args.seed)
            sharp_val = _estimate_top_hessian_eig(
                model,
                dataset=subset,
                device=device,
                physical_batch_size=args.sharpness_phys_bs,
                power_iters=args.sharpness_power_iters,
                seed=args.seed + 1337 + step,
            )
            metrics["sharpness_step"].append(float(step))
            metrics["sharpness"].append(float(sharp_val))

            # If this step isn't an eval step, still emit a line so you see it immediately.
            if not (step % args.eval_every == 0 or step == args.steps):
                print(f"step {step:>7d} | sharp {sharp_val:.3g}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            m_train = _eval(model, full_train_dl, device)
            m_test = _eval(model, test_dl, device)
            metrics["step"].append(float(step))
            metrics["train_loss"].append(m_train["loss"])
            metrics["train_acc"].append(m_train["acc"])
            metrics["test_loss"].append(m_test["loss"])
            metrics["test_acc"].append(m_test["acc"])

            sharp_str = f" | sharp {sharp_val:.3g}" if sharp_val is not None else ""
            print(
                f"step {step:>7d} | "
                f"train acc {m_train['acc']:.3f} loss {m_train['loss']:.3f} | "
                f"test acc {m_test['acc']:.3f} loss {m_test['loss']:.3f}"
                f"{sharp_str}"
            , flush=True)

    bundle = {
        "args": vars(args),
        "split": asdict(split),
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }
    # TensorDataset isn't serializable; remove it from the split dict (we keep metadata only).
    bundle["split"].pop("train", None)
    bundle["split"].pop("test", None)

    out_pt = os.path.join(out_dir, "metrics.pt")
    torch.save(bundle, out_pt)

    out_png = os.path.join(out_dir, "metrics.png")
    _plot(metrics, out_png=out_png, xlog=args.xlog)

    print("saved:", out_pt)
    print("saved:", out_png)


if __name__ == "__main__":
    main()


