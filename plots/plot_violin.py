import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Generate violin plot of relative errors from CSV files"
)
parser.add_argument(
    "input_path", type=str, help="Path to a CSV data file or folder containing CSV files"
)
parser.add_argument(
    "--format",
    type=str,
    default="png",
    choices=["png", "pdf", "svg", "jpg"],
    help="Output image format (default: png)",
)
parser.add_argument(
    "--output-name",
    type=str,
    default="relative_error_violin_plot",
    help="Output filename (default: relative_error_violin_plot)",
)
args = parser.parse_args()

# Check if input is a file or directory
input_path = Path(args.input_path)

csv_files = []
if input_path.is_file():
    # Process single file
    if input_path.suffix.lower() == ".csv":
        csv_files = [input_path]
    else:
        print(f"Error: {input_path} is not a CSV file")
        exit(1)
elif input_path.is_dir():
    # Process all CSV files in directory
    csv_files = sorted(list(input_path.glob("*.csv")))

    if not csv_files:
        print(f"No CSV files found in directory: {input_path}")
        exit(1)
else:
    print(f"Error: {input_path} is neither a file nor a directory")
    exit(1)

print(f"Found {len(csv_files)} CSV file(s)")

# Collect relative errors for each SMOCC variant
smocc_variants = ["smocc_lower", "smocc_mid", "smocc_upper", "mock_smocc"]
relative_errors = {variant: [] for variant in smocc_variants}

# Number of independent runs per evaluation result
RUNS_PER_EVAL = 3

# Process each CSV file
for csv_file in csv_files:
    print(f"Processing: {csv_file.name}")
    try:
        df = pd.read_csv(csv_file, comment="#")

        if len(df) % RUNS_PER_EVAL != 0:
            print(
                f"Warning: {csv_file.name} has {len(df)} rows, "
                f"which is not a multiple of {RUNS_PER_EVAL}. "
                "Rows may not group cleanly into runs."
            )

        # Group rows into chunks of RUNS_PER_EVAL consecutive rows
        group_ids = np.arange(len(df)) // RUNS_PER_EVAL
        averaged = df.groupby(group_ids).mean(numeric_only=True)

        for _, row in averaged.iterrows():
            measured = row["measured"]

            # Calculate relative error for each variant: (predicted - measured) / measured * 100
            for variant in smocc_variants:
                if variant in row:
                    rel_error = (row[variant] - measured) / measured * 100
                    relative_errors[variant].append(rel_error)
                else:
                    print(f"Warning: Column '{variant}' not found in {csv_file.name}")
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")

# Check if we have any data
if all(len(errors) == 0 for errors in relative_errors.values()):
    print("Error: No valid data found in the CSV files")
    exit(1)

# Prepare data for violin plot
data = [relative_errors[variant] for variant in smocc_variants]
labels = ["SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]

# Create violin plot
fig, ax = plt.subplots(figsize=(12, 6))

parts = ax.violinplot(
    data, positions=range(len(labels)), showmeans=True, showmedians=True, widths=0.7
)

# Color the violins with different colors
colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Customize mean and median lines
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)
parts["cmeans"].set_color("red")
parts["cmeans"].set_linewidth(2)

# Add a horizontal line at y=0 (perfect prediction)
ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Perfect Prediction")

# Customize the plot
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
ax.set_ylabel("Relative Error (%)", fontsize=20, fontweight="bold")
# ax.set_title("Relative Error Distribution Across SMOCC Variants", fontsize=18, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")

ax.tick_params(axis="y", direction="in", labelsize=20)
ax.tick_params(axis="x", length=0, labelsize=20)

# Set y-axis range
ax.set_ylim(-60, 70)

legend_elements = [
    Line2D([0], [0], color="black", linewidth=2, label="Median"),
    Line2D([0], [0], color="red", linewidth=2, label="Mean"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Perfect Estimation"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=18)

# Add statistics text
stats_text = []
for i, variant in enumerate(smocc_variants):
    if len(relative_errors[variant]) > 0:
        mean_err = np.mean(relative_errors[variant])
        median_err = np.median(relative_errors[variant])
        stats_text.append(f"{labels[i]}: Mean={mean_err:.1f}%, Median={median_err:.1f}%")
    else:
        stats_text.append(f"{labels[i]}: No data")

# Add text box with statistics
textstr = "\n".join(stats_text)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(
    0.02, 0.98, textstr, transform=ax.transAxes, fontsize=17, verticalalignment="top", bbox=props
)

# Set frame (spines) linewidth
frame_linewidth = 2
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

plt.tight_layout()

# Generate output filename
output_file = f"{args.output_name}.{args.format}"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\nViolin plot saved to: {output_file}")
