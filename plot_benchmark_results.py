import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
from pathlib import Path

if __name__ == "__main__":
    nargs = len(sys.argv)
    if (nargs < 2):
        raise RuntimeError("Please specify the directory with benchmark results as a command-line argument!")

    store_dir = str(sys.argv[1])
    store_path = Path.cwd() / store_dir
    if (not store_path.exists()):
        raise RuntimeError(f"{store_path} doesn't exist!")

    plt.rc("font", size=18)
    fig, ax = plt.subplots(figsize=(14, 10))

    for path in store_path.glob("*.txt"):
        matsize, max_gflops = np.loadtxt(path).T
        legend = path.stem
        ax.plot(matsize, max_gflops, "-", linewidth=2, label=legend)

    ax.set_xlabel("m=n=k", fontsize=24)
    ax.set_ylabel("GFLOP/S", fontsize=24)
    ax.set_title("sgemm.cu", fontsize=22)
    ax.legend(fontsize=12, loc='lower right', prop={'size': 19})
    ax.grid(axis='y')
    fig.savefig("benchmark_data.png")