import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(data_file, output_format="png"):
    """Generate scatter plot for a single CSV file"""
    # Read data from CSV file, skipping lines that start with #
    data = pd.read_csv(data_file, comment="#")

    # Extract columns
    measured = data["measured"].values
    lower = data["smocc_lower"].values
    middle = data["smocc_mid"].values
    upper = data["smocc_upper"].values
    mock = data["mock_smocc"].values

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the four lines with markers
    ax.plot(measured, lower, color="teal", linewidth=2, markersize=7, marker="X", label="lower")
    ax.plot(
        measured, middle, color="skyblue", linewidth=2, markersize=7, marker="*", label="middle"
    )
    ax.plot(measured, upper, color="orange", linewidth=2, markersize=7, marker="s", label="upper")
    ax.plot(measured, mock, color="purple", linewidth=2, markersize=7, marker="o", label="mock")

    # Plot ideal diagonal line
    max_val = max(measured.max(), lower.max(), middle.max(), upper.max(), mock.max())
    ideal_limit = max_val * 1.2  # 20% beyond max value
    ideal_x = [0, ideal_limit]
    ideal_y = [0, ideal_limit]
    ax.plot(ideal_x, ideal_y, "--", color="magenta", linewidth=2, alpha=0.7, label="ideal")

    # Grid
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    # Labels
    ax.set_xlabel("Measured runtime (sec)", fontsize=14)
    ax.set_ylabel("Predicted runtime (sec)", fontsize=14)

    # Axis limits (auto-adjust based on data)
    ax.set_xlim(0, ideal_limit)
    ax.set_ylim(0, ideal_limit)

    # Legend
    ax.legend(fontsize=12, loc="lower right")

    # Tick parameters
    ax.tick_params(labelsize=12)

    # Make the plot square
    ax.set_aspect("equal")

    plt.tight_layout()

    # Generate output filename from input data filename
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    output_file = f"{base_name}_scatter.{output_format}"

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved to: {output_file}")

    plt.close(fig)  # Close the figure to free memory


def plot_combined_scatter(csv_files, output_format="png", output_name="combined"):
    """Generate a single scatter plot combining data from multiple CSV files"""
    import numpy as np

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))

    all_measured = []
    all_predicted = []

    # Collect all data points for each prediction type across all CSV files
    all_lower_x, all_lower_y = [], []
    all_middle_x, all_middle_y = [], []
    all_upper_x, all_upper_y = [], []
    all_mock_x, all_mock_y = [], []

    # Process each CSV file and collect points
    for _, csv_file in enumerate(csv_files):
        # Read data from CSV file, skipping lines that start with #
        data = pd.read_csv(csv_file, comment="#")

        # Extract columns
        measured = data["measured"].values
        lower = data["smocc_lower"].values
        middle = data["smocc_mid"].values
        upper = data["smocc_upper"].values
        mock = data["mock_smocc"].values

        # Collect points for each type
        all_lower_x.extend(measured)
        all_lower_y.extend(lower)

        all_middle_x.extend(measured)
        all_middle_y.extend(middle)

        all_upper_x.extend(measured)
        all_upper_y.extend(upper)

        all_mock_x.extend(measured)
        all_mock_y.extend(mock)

        # Collect all values for determining plot limits
        all_measured.extend(measured)
        all_predicted.extend(lower)
        all_predicted.extend(middle)
        all_predicted.extend(upper)
        all_predicted.extend(mock)

    def sort_points_by_angle(x, y):
        """Sort points by angle around their centroid to form a smooth cycle"""
        x = np.array(x)
        y = np.array(y)

        # Calculate centroid
        cx = np.mean(x)
        cy = np.mean(y)

        # Calculate angle of each point relative to centroid
        angles = np.arctan2(y - cy, x - cx)

        # Sort by angle
        sorted_indices = np.argsort(angles)

        return x[sorted_indices], y[sorted_indices]

    # Sort and plot each prediction type as a closed cycle
    lower_x, lower_y = sort_points_by_angle(all_lower_x, all_lower_y)
    lower_x = list(lower_x) + [lower_x[0]]
    lower_y = list(lower_y) + [lower_y[0]]
    ax.plot(
        lower_x,
        lower_y,
        color="red",
        linewidth=1.5,
        markersize=5,
        marker="X",
        alpha=0.6,
        label="lower",
    )

    middle_x, middle_y = sort_points_by_angle(all_middle_x, all_middle_y)
    middle_x = list(middle_x) + [middle_x[0]]
    middle_y = list(middle_y) + [middle_y[0]]
    ax.plot(
        middle_x,
        middle_y,
        color="teal",
        linewidth=1.5,
        markersize=5,
        marker="*",
        alpha=0.6,
        label="middle",
    )

    upper_x, upper_y = sort_points_by_angle(all_upper_x, all_upper_y)
    upper_x = list(upper_x) + [upper_x[0]]
    upper_y = list(upper_y) + [upper_y[0]]
    ax.plot(
        upper_x,
        upper_y,
        color="orange",
        linewidth=1.5,
        markersize=5,
        marker="s",
        alpha=0.6,
        label="upper",
    )

    mock_x, mock_y = sort_points_by_angle(all_mock_x, all_mock_y)
    mock_x = list(mock_x) + [mock_x[0]]
    mock_y = list(mock_y) + [mock_y[0]]
    ax.plot(
        mock_x,
        mock_y,
        color="purple",
        linewidth=1.5,
        markersize=5,
        marker="o",
        alpha=0.6,
        label="mock",
    )

    # Plot ideal diagonal line
    max_val = max(max(all_measured), max(all_predicted))
    ideal_limit = max_val * 1.2  # 20% beyond max value
    ideal_x = [0, ideal_limit]
    ideal_y = [0, ideal_limit]
    ax.plot(ideal_x, ideal_y, "--", color="magenta", linewidth=2, alpha=0.7, label="ideal")

    # Grid
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    # Labels
    ax.set_xlabel("Measured runtime (sec)", fontsize=14)
    ax.set_ylabel("Predicted runtime (sec)", fontsize=14)

    # Axis limits (auto-adjust based on data)
    ax.set_xlim(0, ideal_limit)
    ax.set_ylim(0, ideal_limit)

    # Legend
    ax.legend(fontsize=12, loc="lower right")

    # Tick parameters
    ax.tick_params(labelsize=12)

    # Make the plot square
    ax.set_aspect("equal")

    plt.tight_layout()

    # Generate output filename
    output_file = f"{output_name}_scatter.{output_format}"

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined scatter plot saved to: {output_file}")

    plt.close(fig)  # Close the figure to free memory


# Set up argument parser
parser = argparse.ArgumentParser(description="Plot runtime prediction scatter plot")
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
    "--combine", action="store_true", help="Combine all CSV files from folder into a single plot"
)
parser.add_argument(
    "--output-name",
    type=str,
    default="combined",
    help="Output filename for combined plot (default: combined)",
)
args = parser.parse_args()

# Check if input is a file or directory
input_path = Path(args.input_path)

if input_path.is_file():
    # Process single file
    if input_path.suffix.lower() == ".csv":
        plot_scatter(str(input_path), args.format)
    else:
        print(f"Error: {input_path} is not a CSV file")
elif input_path.is_dir():
    # Process all CSV files in directory
    csv_files = sorted(list(input_path.glob("*.csv")))

    if not csv_files:
        print(f"No CSV files found in directory: {input_path}")
    else:
        print(f"Found {len(csv_files)} CSV file(s) in {input_path}")

        if args.combine:
            # Combine all files into a single plot
            print(f"\nCreating combined plot from {len(csv_files)} files...")
            try:
                plot_combined_scatter(csv_files, args.format, args.output_name)
            except Exception as e:
                print(f"Error creating combined plot: {e}")
        else:
            # Process each file individually
            for csv_file in csv_files:
                print(f"\nProcessing: {csv_file.name}")
                try:
                    plot_scatter(str(csv_file), args.format)
                except Exception as e:
                    print(f"Error processing {csv_file.name}: {e}")
else:
    print(f"Error: {input_path} is neither a file nor a directory")
