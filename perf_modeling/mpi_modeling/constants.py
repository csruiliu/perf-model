"""
constants.py

Single source of truth for all constants used across the MPI model.

All other modules import from here — never define constants elsewhere.

Sections:
    1. Counter name lists      — four base lists, edit here only
    2. Derived counter layout  — aggregated lists, dimensions, slices
    3. Network parameters      — MTU, header size, payload limits
    4. Histogram bucket bounds — parsed from counter names
    5. Traffic class config    — TC index assignments
    6. Message size bins       — predefined MPI message sizes
"""

import numpy as np
from typing import List, Tuple

# =============================================================
# 1. Counter name lists
#
# These four lists are the ONLY things to edit when adding or
# removing hardware counters. Everything else is derived.
#
# Histogram counters : packet size distribution buckets
# TC counters        : traffic class packet counts
# =============================================================

TX_HIST_COUNTER_NAMES: List[str] = [
    "hni_tx_ok_64",
    "hni_tx_ok_65_to_127",
    "hni_tx_ok_128_to_255",
    "hni_tx_ok_256_to_511",
    "hni_tx_ok_512_to_1023",
    "hni_tx_ok_1024_to_2047",
    "hni_tx_ok_2048_to_4095",
]

TX_TC_COUNTER_NAMES: List[str] = [
    "hni_pkts_sent_by_tc_0",
    "hni_pkts_sent_by_tc_1"
]

RX_HIST_COUNTER_NAMES: List[str] = [
    "hni_rx_ok_64",
    "hni_rx_ok_65_to_127",
    "hni_rx_ok_128_to_255",
    "hni_rx_ok_256_to_511",
    "hni_rx_ok_512_to_1023",
    "hni_rx_ok_1024_to_2047",
    "hni_rx_ok_2048_to_4095",
]

RX_TC_COUNTER_NAMES: List[str] = [
    "hni_pkts_recv_by_tc_0",
    "hni_pkts_recv_by_tc_1"
]

# =============================================================
# 2. Derived counter layout
#
# All derived from the four lists above — do not edit directly.
# =============================================================

# Validate symmetry before deriving anything
assert len(TX_HIST_COUNTER_NAMES) == len(RX_HIST_COUNTER_NAMES), (
    "TX and RX histogram counter lists must have the same length"
)
assert len(TX_TC_COUNTER_NAMES) == len(RX_TC_COUNTER_NAMES), (
    "TX and RX TC counter lists must have the same length"
)

# Aggregated per-direction lists
TX_COUNTER_NAMES: List[str] = TX_HIST_COUNTER_NAMES + TX_TC_COUNTER_NAMES
RX_COUNTER_NAMES: List[str] = RX_HIST_COUNTER_NAMES + RX_TC_COUNTER_NAMES

# Full ordered counter vector:
#   y = [TX_hist | TX_TC | RX_hist | RX_TC]
COUNTER_ORDER: List[str] = TX_COUNTER_NAMES + RX_COUNTER_NAMES

# Dimensions — derived, never hardcoded
N_HIST: int = len(TX_HIST_COUNTER_NAMES)
N_TC: int = len(TX_TC_COUNTER_NAMES)
M: int = N_HIST + N_TC

# Named index slices into a counter vector of length 2 * M
TX_HIST_SLICE: slice = slice(0, N_HIST)
TX_TC_SLICE:   slice = slice(N_HIST, M)
RX_HIST_SLICE: slice = slice(M, M + N_HIST)
RX_TC_SLICE:   slice = slice(M + N_HIST, 2 * M)

# =============================================================
# 3. Network parameters
# =============================================================

MTU: int = 2048   # max packet size including header (bytes)
HEADER_SIZE: int = 64     # Cassini packet header size (bytes)
PAYLOAD_MAX: int = MTU - HEADER_SIZE

# =============================================================
# 4. Histogram bucket bounds
#
# Parsed automatically from TX_HIST_COUNTER_NAMES so they stay
# consistent if the counter list changes.
#
# Parsing rules:
#   "hni_tx_ok_64"          --> exact:  lower=upper=64
#   "hni_tx_ok_65_to_127"   --> range:  lower=65,   upper=127
#   "hni_tx_ok_2048_to_4095"--> range:  lower=2048, upper=4095
# =============================================================

def _parse_bucket_bounds(hist_names: List[str], prefix: str) -> Tuple[List[int], List[int]]:
    """
    Parse histogram bucket lower/upper bounds from counter names.

    Parameters
    ----------
    hist_names : list of str
        e.g. TX_HIST_COUNTER_NAMES
    prefix : str
        e.g. "hni_tx_ok_" or "hni_rx_ok_"

    Returns
    -------
    lowers : list of int
    uppers : list of int
    """
    lowers: List[int] = []
    uppers: List[int] = []

    for name in hist_names:
        suffix = name[len(prefix):]
        if "_to_" in suffix:
            lo, hi = suffix.split("_to_")
            lowers.append(int(lo))
            uppers.append(int(hi))
        else:
            val = int(suffix)
            lowers.append(val)
            uppers.append(val)

    return lowers, uppers


BUCKET_LOWER: List[int]
BUCKET_UPPER: List[int]
BUCKET_LOWER, BUCKET_UPPER = _parse_bucket_bounds(
    TX_HIST_COUNTER_NAMES, prefix="hni_tx_ok_"
)

# Sanity check — ACK size (HEADER_SIZE) must land in bucket 0
assert BUCKET_LOWER[0] <= HEADER_SIZE <= BUCKET_UPPER[0], (
    f"HEADER_SIZE={HEADER_SIZE} does not fall in bucket 0 "
    f"[{BUCKET_LOWER[0]}, {BUCKET_UPPER[0]}]"
)


# =============================================================
# 5. Traffic class config
# =============================================================
TC_DATA: int = 0   # data packets routed to TC0
TC_ACK:  int = 1   # ACK  packets routed to TC1

# Validate TC indices are within range
assert TC_DATA < N_TC, f"TC_DATA={TC_DATA} >= N_TC={N_TC}"
assert TC_ACK  < N_TC, f"TC_ACK={TC_ACK}   >= N_TC={N_TC}"


# =============================================================
# 6. Message size bins
#
# Bucket assignments per empirically confirmed Cassini NIC mapping:
#   single-packet ranges:
#     bucket 0 [64B]        :   1 –  11 B
#     bucket 1 [65–127B]    :  12 –  74 B
#     bucket 2 [128–255B]   :  75 – 192 B
#   two-packet ranges (64B control + data, TC/ACK count UNVERIFIED):
#     bucket 0 + bucket 2   : 193 – 202 B
#     bucket 0 + bucket 3   : 203 – 458 B
#     bucket 0 + bucket 4   : 459 – 970 B
#     bucket 0 + bucket 5   : 971 – 1994 B
#     bucket 0 + bucket 6   : 1995 – 2048 B
#   super-MTU (fragmentation formula UNVERIFIED):
#     > 2048 B
# =============================================================
MSG_SIZES: np.ndarray = np.array([
    11, 74, 192, 202, 458, 970, 1994,
    2  * 1024,  4  * 1024,  8  * 1024,  16 * 1024,
    32 * 1024,  64 * 1024,  128 * 1024, 256 * 1024,
    512 * 1024, 1024 * 1024, 2048 * 1024, 4096 * 1024,
], dtype=np.float64)

MSG_SIZES_FINE: np.ndarray = np.array([
    4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56,
    64, 80, 96, 112, 128, 160,
    192, 224, 256, 320, 384, 448,
    512, 640, 768, 896,
    1024, 1280, 1536, 1792,
    2048,
    4  * 1024,  8  * 1024,  16 * 1024,  32 * 1024,
    64 * 1024,  128 * 1024, 256 * 1024, 512 * 1024,
    1024 * 1024, 2048 * 1024, 4096 * 1024,
], dtype=np.float64)

MSG_SIZES_COARSE: np.ndarray = np.array([
    64, 128, 256, 512, 1024, 2048,
    4  * 1024,  8  * 1024,  16 * 1024,  32 * 1024,
    64 * 1024,  128 * 1024, 256 * 1024, 512 * 1024,
    1024 * 1024, 2048 * 1024, 4096 * 1024,
], dtype=np.float64)

# Lookup dict — used by main.py --msg_set argument
MSG_SIZE_SETS: dict = {
    "default" : MSG_SIZES,
    "fine"    : MSG_SIZES_FINE,
    "coarse"  : MSG_SIZES_COARSE,
}


# =============================================================
# Summary printout — useful for debugging
# =============================================================
def print_constants() -> None:
    """Print a summary of all derived constants for verification."""
    print("=== MPI Model Constants ===")
    print()
    print("Counter layout:")
    print(f"  N_HIST  = {N_HIST}  (histogram buckets per direction)")
    print(f"  N_TC    = {N_TC}  (traffic class counters per direction)")
    print(f"  M       = {M}  (total counters per direction)")
    print(f"  TWO_M   = {2 * M}  (total counters TX + RX)")
    print()
    print("Index slices:")
    print(f"  TX_HIST_SLICE = {TX_HIST_SLICE}")
    print(f"  TX_TC_SLICE   = {TX_TC_SLICE}")
    print(f"  RX_HIST_SLICE = {RX_HIST_SLICE}")
    print(f"  RX_TC_SLICE   = {RX_TC_SLICE}")
    print()
    print("Network parameters:")
    print(f"  MTU         = {MTU} bytes")
    print(f"  HEADER_SIZE = {HEADER_SIZE} bytes")
    print(f"  PAYLOAD_MAX = {PAYLOAD_MAX} bytes")
    print()
    print("Histogram bucket bounds:")
    for b, (lo, hi) in enumerate(zip(BUCKET_LOWER, BUCKET_UPPER)):
        name = TX_HIST_COUNTER_NAMES[b]
        print(f"  bucket {b}: [{lo:>5}, {hi:>5}]  ({name})")
    print()
    print("Traffic class assignments:")
    print(f"  TC_DATA = {TC_DATA}  (data packets)")
    print(f"  TC_ACK  = {TC_ACK}  (ACK  packets)")
    print()
    print("Message size bins:")
    for j, m in enumerate(MSG_SIZES):
        size_str = f"{int(m / 1024)}KB" if m >= 1024 else f"{int(m)}B"
        print(f"  bin {j:>2}: {size_str:>8}")