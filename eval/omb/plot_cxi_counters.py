import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import re
import argparse
import os


def parse_counter_file(filename):
    """Parse a counter file and return a list of samples, each containing counters"""
    samples = []
    current_sample = None
    current_timestamp = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for sample start
            if line.startswith("=== SAMPLE_START"):
                parts = line.split()
                if len(parts) >= 3:
                    current_timestamp = float(parts[2])
                    current_sample = {}

            # Check for sample end
            elif line.startswith("=== SAMPLE_END"):
                if current_sample is not None and current_timestamp is not None:
                    samples.append(
                        {"timestamp": current_timestamp, "counters": current_sample}
                    )
                current_sample = None
                current_timestamp = None

            # Parse counter line
            elif current_sample is not None:
                parts = line.split()
                if len(parts) >= 2:
                    counter_name = parts[0]
                    value_timestamp = parts[1]

                    # Handle both formats: value@timestamp or just value
                    if "@" in value_timestamp:
                        value_str = value_timestamp.split("@")[0]
                    else:
                        value_str = value_timestamp

                    try:
                        value = float(value_str)
                        current_sample[counter_name] = value
                    except ValueError:
                        continue

    return samples


def get_job_files(directory, job_id):
    """Get all files for a given job ID in the specified directory"""
    pattern = os.path.join(directory, f"*{job_id}*.txt")
    return sorted(glob.glob(pattern))


def get_all_files(directory):
    """Get all .txt files in the specified directory"""
    pattern = os.path.join(directory, "*.txt")
    return sorted(glob.glob(pattern))


def extract_node_id(filename):
    """Extract node ID (hostname) from filename"""
    basename = os.path.basename(filename)
    # Pattern: cxi_snapshot.before.${SLURM_JOB_ID}-$(hostname)
    # or cxi_snapshot.after.${SLURM_JOB_ID}-$(hostname)
    # or cxi_telemetry.t$(INTERVAL).${SLURM_JOB_ID}-$(hostname)
    match = re.search(r"-([^.]+)(?:\.txt)?$", basename)
    return match.group(1) if match else None


def determine_phase(filename):
    """Determine the phase (before/during/after) from filename"""
    basename = os.path.basename(filename)
    if basename.startswith("cxi_snapshot.before."):
        return "before"
    elif basename.startswith("cxi_snapshot.after."):
        return "after"
    elif basename.startswith("cxi_monitor."):
        return "during"
    else:
        return "unknown"


def plot_counter(counter_name, directory=".", job_id=None, output_file=None):
    """Plot a specific counter across all files and samples"""

    # Get all relevant files
    if job_id:
        files = get_job_files(directory, job_id)
    else:
        files = get_all_files(directory)

    if not files:
        print(f"No files found in directory: {directory}")
        return

    print(f"Found {len(files)} files to process")

    # Organize data by node and phase
    nodes_data = {}

    for filename in files:
        node_id = extract_node_id(filename)
        if node_id is None:
            continue

        phase = determine_phase(filename)
        samples = parse_counter_file(filename)

        if node_id not in nodes_data:
            nodes_data[node_id] = []

        # Extract counter values from each sample
        for sample in samples:
            if counter_name in sample["counters"]:
                nodes_data[node_id].append(
                    {
                        "timestamp": sample["timestamp"],
                        "value": sample["counters"][counter_name],
                        "phase": phase,
                        "filename": os.path.basename(filename),
                    }
                )

    if not nodes_data:
        print(f"Counter '{counter_name}' not found in any files!")
        return

    total_samples = sum(len(v) for v in nodes_data.values())
    print(f"Found counter '{counter_name}' in {total_samples} samples across {len(nodes_data)} nodes")

    # Define colors and markers for phases
    phase_colors = {"before": "#1f77b4", "during": "#2ca02c", "after": "#d62728"}
    phase_markers = {"before": "o", "during": "s", "after": "^"}
    phase_order = ["before", "during", "after"]

    # Find global min timestamp for relative time calculation
    all_timestamps = []
    for node_data in nodes_data.values():
        all_timestamps.extend([d["timestamp"] for d in node_data])

    if not all_timestamps:
        print("No data to plot!")
        return

    start_time = min(all_timestamps)

    # Create subplots - one for each node
    num_nodes = len(nodes_data)
    fig, axes = plt.subplots(num_nodes, 1, figsize=(14, 5 * num_nodes), squeeze=False)
    axes = axes.flatten()

    # Plot each node in its own subplot
    for idx, node_id in enumerate(sorted(nodes_data.keys())):
        ax = axes[idx]
        data = sorted(nodes_data[node_id], key=lambda x: x["timestamp"])

        timestamps = [d["timestamp"] for d in data]
        values = [d["value"] for d in data]
        phases = [d["phase"] for d in data]

        # Convert to relative time
        relative_times = [(t - start_time) for t in timestamps]

        # Plot by phase
        for phase in phase_order:
            phase_indices = [i for i, p in enumerate(phases) if p == phase]
            if not phase_indices:
                continue

            phase_times = [relative_times[i] for i in phase_indices]
            phase_values = [values[i] for i in phase_indices]

            ax.scatter(phase_times, phase_values, 
                       label=f"{phase}", marker=phase_markers[phase], color=phase_colors[phase],
                       s=100, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)

        # Connect all points with a light line
        ax.plot(relative_times, values,
                alpha=0.3, linewidth=1.5, color="gray", linestyle="-", zorder=1)

        # Calculate and display differences for 'during' phase
        during_indices = [i for i, p in enumerate(phases) if p == "during"]
        before_indices = [i for i, p in enumerate(phases) if p == "before"]
        after_indices = [i for i, p in enumerate(phases) if p == "after"]
        during_diffs = []

        if len(during_indices) > 1:
            for i in range(len(during_indices) - 1):
                idx1 = during_indices[i]
                idx2 = during_indices[i + 1]

                diff = values[idx2] - values[idx1]
                during_diffs.append(diff)
                mid_time = (relative_times[idx1] + relative_times[idx2]) / 2
                mid_value = (values[idx1] + values[idx2]) / 2

                # Shift annotation to the left in data coordinates
                time_range = max(relative_times) - min(relative_times)
                shift_amount = -time_range * 0.02  # Shift left by 2% of time range

                # Annotate the difference - shifted left to avoid covering markers
                ax.annotate(
                    f"Δ={diff:.0f}",
                    xy=(mid_time + shift_amount, mid_value),
                    xytext=(0, 15 if diff >= 0 else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="#2ca02c",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="#2ca02c",
                        alpha=0.8,
                    ),
                    zorder=4,
                )

        # Calculate difference between last value of "before" and first value of "after"
        last_before_to_first_after_diff = None
        if before_indices and after_indices:
            last_before_idx = before_indices[-1]
            first_after_idx = after_indices[0]
            last_before_to_first_after_diff = (values[first_after_idx] - values[last_before_idx])

            # Print the difference
            print(f"Node {node_id}: Difference between last 'before' and first 'after' = {last_before_to_first_after_diff:.2f}")

            # Draw a line connecting last before to first after
            ax.plot(
                [relative_times[last_before_idx], relative_times[first_after_idx]],
                [values[last_before_idx], values[first_after_idx]],
                color="purple",
                linewidth=2,
                linestyle="--",
                alpha=0.6,
                zorder=2,
            )

            # Annotate this transition - shifted RIGHT to avoid overlap
            mid_time = (relative_times[last_before_idx] + relative_times[first_after_idx]) / 2
            mid_value = (values[last_before_idx] + values[first_after_idx]) / 2

            # Shift annotation to the right in data coordinates
            time_range = max(relative_times) - min(relative_times)
            shift_amount_right = time_range * 0.15  # Shift right by 15% of time range

            ax.annotate(
                f"Δ={last_before_to_first_after_diff:.0f}",
                xy=(mid_time + shift_amount_right, mid_value),
                xytext=(0, 20 if last_before_to_first_after_diff >= 0 else -20),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="purple",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="yellow",
                    edgecolor="purple",
                    alpha=0.8,
                ),
                zorder=5,
            )

        # Add phase regions as background
        phase_times = {}
        for entry in data:
            phase = entry["phase"]
            rel_time = entry["timestamp"] - start_time
            if phase not in phase_times:
                phase_times[phase] = []
            phase_times[phase].append(rel_time)

        for phase in phase_order:
            if phase in phase_times and phase_times[phase]:
                min_t = min(phase_times[phase])
                max_t = max(phase_times[phase])
                ax.axvspan(min_t, max_t, alpha=0.05, color=phase_colors[phase], zorder=0)

        # Add margins at top and bottom to prevent overlap with title/labels
        if values:
            value_range = max(values) - min(values)
            # Add 15% margin at top and bottom
            margin = value_range * 0.15 if value_range > 0 else max(values) * 0.15
            ax.set_ylim(min(values) - margin, max(values) + margin)

        # Formatting
        ax.set_xlabel("Time (seconds from first measurement)", fontsize=11)
        ax.set_ylabel("Counter Value", fontsize=11)

        # Add average difference and last-before-to-first-after difference to title
        title = f"Node {node_id} - Counter: {counter_name}"
        if during_diffs:
            avg_diff = np.mean(during_diffs)
            title += f" (Avg Δ during: {avg_diff:.2f}"
            if last_before_to_first_after_diff is not None:
                title += f", Δ last_before→first_after: {last_before_to_first_after_diff:.2f}"
            title += ")"
        elif last_before_to_first_after_diff is not None:
            title += (f" (Δ last_before→first_after: {last_before_to_first_after_diff:.2f})")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
            

def main():
    parser = argparse.ArgumentParser(
        description="Plot counter values from CXI snapshot and telemetry files with multiple samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s -c hni_pkts_sent_by_tc_0 -d /path/to/data
            %(prog)s -c hni_pkts_sent_by_tc_0 -d ./results -j 47426802
            %(prog)s -c hni_rx_ok_0 -d ./data -j 47426802 -o output.png
        """,
    )

    parser.add_argument("-c", "--counter", required=True, 
                        help="Counter name to plot (e.g., hni_pkts_sent_by_tc_0)")

    parser.add_argument("-d", "--directory", default=".",
                        help="Directory containing the counter files (default: current directory)")

    parser.add_argument("-j", "--job-id", help="Slurm job ID to filter files (optional)")

    parser.add_argument("-o", "--output", help="Output file path to save the plot (optional, default: show plot)")

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist!")
        return 1

    plot_counter(args.counter, args.directory, args.job_id, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
