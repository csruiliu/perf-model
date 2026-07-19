# import libraries
import matplotlib.pyplot as plt
import pandas as pd

# set the style of the axes and the text color
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["text.color"] = "black"

# read the data from the CSV file
raw = pd.read_csv("dgemm-a100-ref.csv")

# average the numeric columns (three rows per column)
avg = raw.mean(numeric_only=True)

# map the averaged CSV columns to the labels used in the plot
percentages = pd.Series(
    [avg["mock_smocc"], avg["smocc_upper"], avg["smocc_mid"], avg["smocc_lower"], avg["measured"]],
    index=[
        "Estimate (SMOCC Mock)",
        "Estimate (SMOCC Upper)",
        "Estimate (SMOCC Mid)",
        "Estimate (SMOCC Lower)",
        "H100 Measurement",
    ],
)
df = pd.DataFrame({"percentage": percentages})
# df = df.sort_values(by='percentage')

# reference value for computing error (the H100 measurement)
ref_value = avg["measured"]

# (1) add spacing between bars by scaling the y positions with a spacing factor
spacing = 1.6  # increase for more space between bars
my_range = [i * spacing for i in range(1, len(df.index) + 1)]

fig, ax = plt.subplots(figsize=(6, 2.4))

# create for each expense type an horizontal line that starts at x = 0 with the length
# represented by the specific expense percentage value.
plt.hlines(y=my_range, xmin=0, xmax=df["percentage"], color="teal", alpha=0.2, linewidth=14)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df["percentage"], my_range, "o", markersize=13, color="teal", alpha=0.6)

# (2) add error percentage annotation on the right of the four "Estimate" bars
# (2) add error percentage annotation on the right of the four "Estimate" bars
estimate_indices = [0, 1, 2, 3]  # positions of the Estimate rows
annotation_offset = 4  # increase for more space between the point and the text
for i in estimate_indices:
    value = df["percentage"].iloc[i]
    error_pct = (value - ref_value) / ref_value * 100
    ax.text(
        value + annotation_offset,
        my_range[i],
        f"{error_pct:+.1f}%",  # signed error relative to H100 measurement
        va="center",
        ha="left",
        fontsize=15,
        fontstyle="italic",
        color="#333F4B",
    )

# set labels
ax.set_xlabel("Compute Throughput (TFLOPS)", fontsize=18, fontweight="black", color="#333F4B")
ax.set_ylabel("")

# set axis
ax.tick_params(axis="both", which="major", labelsize=18)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(my_range, df.index)

# change the style of the axis spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_bounds((my_range[0], my_range[-1]))
ax.set_xlim(0, max(df["percentage"]) * 1.3)

ax.spines["left"].set_position(("outward", 8))
ax.spines["bottom"].set_position(("outward", 5))

plt.savefig("dgemm_ref_a100.png", format="png", bbox_inches="tight", pad_inches=0.05)
