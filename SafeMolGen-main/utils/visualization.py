"""Plotting utilities."""

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    for name, values in history.items():
        plt.plot(values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
