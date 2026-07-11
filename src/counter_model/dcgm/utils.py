import matplotlib.pyplot as plt
import numpy as np


def print_reference_results(metrics: dict[str, list[float]], flops: float, mem_bw: float, gpu: str):
    """Print reference hardware results, convert runtime(ms) to second"""
    """No need to have total time"""
    print(f"\n{'=' * 60}")
    print(f"Reference Hardware: {gpu}\n")
    print(f"Estimated TFLOPS: {flops:.2f}")
    print(f"Estimated GPU Memory Bandwidth: {mem_bw:.2f} GB/s")

    print(f"\nEstimated Kernel Time: {sum(metrics['t_kernel']) / 1000:.2f} s")
    print(f"\nEstimated PCIe Time: {sum(metrics['t_pcie']) / 1000:.2f} s")
    print(f"Estimated Host Time: {sum(metrics['t_host']) / 1000:.2f} s")
    print(f"{'=' * 60}\n")


def print_target_results(
    metrics: dict[str, list[float]], flops: dict[str, float], mem_bw: dict[str, float], gpu: str
):
    """Print target hardware results, convert runtime(ms) to second"""
    print(f"\n{'=' * 60}")
    print(f"Target Hardware: {gpu}")

    print(f"Estimated TFLOPS [Lower SMOCC]: {flops.get('flops_lower'):.2f} GB/s")
    print(f"Estimated TFLOPS [Mid SMOCC]: {flops.get('flops_mid'):.2f} GB/s")
    print(f"Estimated TFLOPS [Upper SMOCC]: {flops.get('flops_upper'):.2f} GB/s")
    print(f"Estimated TFLOPS [Mock SMOCC]: {flops.get('flops_mock'):.2f} GB/s")

    print(f"Estimated Memory Bandwidth [Lower SMOCC]: {mem_bw.get('dram_lower'):.2f} GB/s")
    print(f"Estimated Memory Bandwidth [Mid SMOCC]: {mem_bw.get('dram_mid'):.2f} GB/s")
    print(f"Estimated Memory Bandwidth [Upper SMOCC]: {mem_bw.get('dram_upper'):.2f} GB/s")
    print(f"Estimated Memory Bandwidth [Mock SMOCC]: {mem_bw.get('dram_mock'):.2f} GB/s")

    print(f"\nEstimated Kernel Time [Lower SMOCC]: {sum(metrics['t_kernel_lower']) / 1000:.2f} s")
    print(f"Estimated Kernel Time [Mid SMOCC]:   {sum(metrics['t_kernel_mid']) / 1000:.2f} s")
    print(f"Estimated Kernel Time [Upper SMOCC]: {sum(metrics['t_kernel_upper']) / 1000:.2f} s")
    print(f"Estimated Kernel Time [Mock SMOCC]: {sum(metrics['t_kernel_mock']) / 1000:.2f} s")

    print(f"\nEstimated PCIe Time: {sum(metrics['t_pcie']) / 1000:.2f} s")
    print(f"\nEstimated Host Time: {sum(metrics['t_host']) / 1000:.2f} s")

    print(f"\nEstimated Total Runtime [Lower SMOCC]: {sum(metrics['t_total_lower']) / 1000:.2f} s")
    print(f"Estimated Total Runtime [Mid SMOCC]:   {sum(metrics['t_total_mid']) / 1000:.2f} s")
    print(f"Estimated Total Runtime [Upper SMOCC]: {sum(metrics['t_total_upper']) / 1000:.2f} s")
    print(f"Estimated Total Runtime [Mock SMOCC]: {sum(metrics['t_total_mock']) / 1000:.2f} s")
    print(f"{'=' * 60}\n")


def plot_speedup_distribution(result_df, gpu_name, outpath, bins=40):
    """Node-hours-weighted speedup PDF + cumulative % for one target GPU."""
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    s = result_df["speedup"].to_numpy()
    weights = result_df["node_hours"].to_numpy()

    ax.hist(s, bins=bins, weights=weights, alpha=0.6, label=gpu_name)

    # Cumulative percentage curve.
    order = np.argsort(s)
    cum = 100.0 * np.cumsum(weights[order]) / np.sum(weights)
    ax2.plot(s[order], cum, linestyle="--", marker=".", label=f"{gpu_name} Cumulative %")

    ax.set_xlabel(f"Speedup relative to A100 ({gpu_name})")
    ax.set_ylabel("PDF (weighted by node-hours)")
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax.legend(loc="center left")
    fig.savefig(outpath, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)
