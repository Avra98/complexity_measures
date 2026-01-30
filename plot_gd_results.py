import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch


def _load_first_existing(directory: str, stem: str) -> Optional[torch.Tensor]:
    """
    gd.py saves tensors either as {stem} (periodic saves) or {stem}_final (final save).
    Prefer *_final if present.
    """
    final_path = os.path.join(directory, f"{stem}_final")
    if os.path.exists(final_path):
        return torch.load(final_path, map_location="cpu")
    path = os.path.join(directory, stem)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot saved gd.py results (no training).")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Results directory produced by src/gd.py (contains train_loss, test_loss, eigs, trace_hessian, ...).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. Defaults to <dir>/summary.png",
    )
    args = parser.parse_args()

    directory = args.dir
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Not a directory: {directory}")

    out_path = args.out or os.path.join(directory, "summary.png")

    train_loss = _load_first_existing(directory, "train_loss")
    test_loss = _load_first_existing(directory, "test_loss")
    train_acc = _load_first_existing(directory, "train_acc")
    test_acc = _load_first_existing(directory, "test_acc")
    eigs = _load_first_existing(directory, "eigs")
    trace_hessian = _load_first_existing(directory, "trace_hessian")

    # Convert to 1D floats for plotting
    def to_np(x: Optional[torch.Tensor]):
        if x is None:
            return None
        x = x.detach().cpu()
        if x.ndim == 0:
            x = x.unsqueeze(0)
        return x.numpy()

    train_loss_np = to_np(train_loss)
    test_loss_np = to_np(test_loss)
    train_acc_np = to_np(train_acc)
    test_acc_np = to_np(test_acc)
    trace_np = to_np(trace_hessian)
    # eigs is [T, neigs] typically; use top eigenvalue column 0 if available.
    top_eig_np = None
    if eigs is not None:
        e = eigs.detach().cpu()
        if e.ndim == 1:
            top_eig_np = e.numpy()
        elif e.ndim == 2 and e.shape[1] >= 1:
            top_eig_np = e[:, 0].numpy()

    # Figure: loss + acc + sharpness + trace (if present)
    nrows = 2
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10.5, 7.5))
    axs = axs.reshape(nrows, ncols)

    # Loss
    ax = axs[0, 0]
    if train_loss_np is not None:
        ax.plot(train_loss_np, lw=2.0, label="train")
    if test_loss_np is not None:
        ax.plot(test_loss_np, lw=2.0, label="test")
    ax.set_title("loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)

    # Accuracy
    ax = axs[0, 1]
    if train_acc_np is not None:
        ax.plot(train_acc_np, lw=2.0, label="train")
    if test_acc_np is not None:
        ax.plot(test_acc_np, lw=2.0, label="test")
    ax.set_title("accuracy")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)

    # Sharpness (top Hessian eigenvalue)
    ax = axs[1, 0]
    if top_eig_np is not None:
        ax.plot(top_eig_np, lw=2.0)
        ax.set_title("sharpness (top Hessian eigenvalue)")
        ax.set_xlabel("eig checkpoint index")
        ax.set_ylabel(r"$\lambda_{\max}(H)$")
        ax.grid(True, alpha=0.35)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No eigs found", ha="center", va="center")

    # Trace(H)
    ax = axs[1, 1]
    if trace_np is not None:
        ax.plot(trace_np, lw=2.0)
        ax.set_title("trace(H)")
        ax.set_xlabel("eig checkpoint index")
        ax.set_ylabel(r"$\mathrm{tr}(H)$")
        ax.grid(True, alpha=0.35)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No trace_hessian found", ha="center", va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"✓ Saved plot to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()


