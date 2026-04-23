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

# =============================================================
# Counter name lists
#
# Histogram counters : packet size distribution buckets
# TC counters        : traffic class packet counts
# =============================================================

# Ruff won't format this code
# fmt: off


TX_HIST_CNTRS: list[str] = [
    "hni_tx_ok_64",
    "hni_tx_ok_65_to_127",
    "hni_tx_ok_128_to_255",
    "hni_tx_ok_256_to_511",
    "hni_tx_ok_512_to_1023",
    "hni_tx_ok_1024_to_2047",
    "hni_tx_ok_2048_to_4095"
]

TX_TC_CNTRS: list[str] = ["hni_pkts_sent_by_tc_0", "hni_pkts_sent_by_tc_1"]

RX_HIST_CNTRS: list[str] = [
    "hni_rx_ok_64",
    "hni_rx_ok_65_to_127",
    "hni_rx_ok_128_to_255",
    "hni_rx_ok_256_to_511",
    "hni_rx_ok_512_to_1023",
    "hni_rx_ok_1024_to_2047",
    "hni_rx_ok_2048_to_4095",
]

RX_TC_CNTRS: list[str] = ["hni_pkts_recv_by_tc_0", "hni_pkts_recv_by_tc_1"]

MISC_CNTRS = ["lpe_net_match_priority_0", "lpe_net_match_overflow_0"]

# Validate symmetry before deriving anything
assert len(TX_HIST_CNTRS) == len(RX_HIST_CNTRS), "TX and RX histogram counter lists must have the same length"
assert len(TX_TC_CNTRS) == len(RX_TC_CNTRS), "TX and RX TC counter lists must have the same length"

# =============================================================
# Derived counter layout
#
# All derived from the four lists above — do not edit directly.
# =============================================================

# Aggregated per-direction lists
TX_CNTRS: list[str] = TX_HIST_CNTRS + TX_TC_CNTRS
RX_CNTRS: list[str] = RX_HIST_CNTRS + RX_TC_CNTRS

# Full ordered counter vector:
#   y = [TX_HIST | TX_TC | RX_HIST | RX_TC]
ALL_CNTRS: list[str] = TX_CNTRS + RX_CNTRS

# Dimensions
NUM_HIST_CNTRS: int = len(TX_HIST_CNTRS)
NUM_TC_CNTRS: int = len(TX_TC_CNTRS)
NUM_ALL_CNTRS: int = NUM_HIST_CNTRS + NUM_TC_CNTRS

# Named index slices into a counter vector of length 2 * N_ALL_CNTRS
TX_HIST_SLICE: slice = slice(0, NUM_HIST_CNTRS)
TX_TC_SLICE: slice = slice(NUM_HIST_CNTRS, NUM_ALL_CNTRS)
RX_HIST_SLICE: slice = slice(NUM_ALL_CNTRS, NUM_ALL_CNTRS + NUM_HIST_CNTRS)
RX_TC_SLICE: slice = slice(NUM_ALL_CNTRS + NUM_HIST_CNTRS, 2 * NUM_ALL_CNTRS)

# =============================================================
# Message size bins
#
# Bucket assignments per empirically confirmed Cassini NIC mapping:
#   single-packet ranges:
#     bucket 0 [64B]        :   1 –  11 B
#     bucket 1 [65–127B]    :  12 –  74 B
#     bucket 2 [128–255B]   :  75 – 192 B
#   two-packet ranges (64B control + data, TC/ACK count):
#     bucket 0 + bucket 2   : 193 – 202 B
#     bucket 0 + bucket 3   : 203 – 458 B
#     bucket 0 + bucket 4   : 459 – 970 B
#     bucket 0 + bucket 5   : 971 – 1994 B
#     bucket 0 + bucket 6   : 1995 – 2048 B
#   super-MTU (fragmentation):
#     > 2048 B
# =============================================================
MSG_SIZES_PM: np.ndarray = np.array([
    11, 74, 192, 202, 458, 970, 1994,
    2  * 1024,  4  * 1024,  8  * 1024,  16 * 1024,
    32 * 1024,  64 * 1024,  128 * 1024, 256 * 1024,
    512 * 1024, 1024 * 1024, 2048 * 1024, 4096 * 1024,
    8192 * 1024,
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
    1024 * 1024, 2048 * 1024, 4096 * 1024, 8192 * 1024,
], dtype=np.float64)

MSG_SIZES_COARSE: np.ndarray = np.array([
    64, 128, 256, 512, 1024, 2048,
    4  * 1024,  8  * 1024,  16 * 1024,  32 * 1024,
    64 * 1024,  128 * 1024, 256 * 1024, 512 * 1024,
    1024 * 1024, 2048 * 1024, 4096 * 1024, 8192 * 1024,
], dtype=np.float64)

# dict of msg size sets
MSG_SIZE_SETS: dict = {
    "permultter" : MSG_SIZES_PM,
    "fine" : MSG_SIZES_FINE,
    "coarse" : MSG_SIZES_COARSE,
}

# Groups all four counter slices with their human-readable names and counter name lists.
# Used by validate_solution to report per-group breakdowns consistently.
COUNTER_GROUPS = [
    ("TX Histogram",     TX_HIST_SLICE, TX_HIST_CNTRS),
    ("TX Traffic Class", TX_TC_SLICE,   TX_TC_CNTRS),
    ("RX Histogram",     RX_HIST_SLICE, RX_HIST_CNTRS),
    ("RX Traffic Class", RX_TC_SLICE,   RX_TC_CNTRS),
]

# =============================================================
# Network parameters
# =============================================================
MTU: int = 2048  # max packet size including header (bytes)
# HEADER_SIZE: int = 64     # Cassini packet header size (bytes)
# PAYLOAD_MAX: int = MTU - HEADER_SIZE

# =============================================================
# Protocol threshold
#
# Messages >  RENDEZVOUS_THRESHOLD bytes --> rendezvous protocol
# Messages <= RENDEZVOUS_THRESHOLD bytes --> eager protocol
#
# Observed sender-side TC flip in rendezvous:
#   eager      : tc_DATA is the request class (TX), tc_ACK is the response class (RX)
#   rendezvous : tc0 = response (RX),  tc1 = request (TX) <-- swapped
# =============================================================
RDZV_THRESHOLD: int = 16384  # bytes

# =============================================================
# Traffic class config
# =============================================================
TC_DATA: int = 0  # request packets routed to TC0
TC_ACK: int = 1  # response packets routed to TC1

# =============================================================
# Empirically confirmed sub-MTU fragmentation map
# Each tuple is (upper msg_size, [histogram data bucket indices])
# =============================================================
SUB_MTU_MAP: list[tuple[int, list[int]]] = [
    (11, [0]),
    (74, [1]),
    (192, [2]),
    (202, [0, 2]),
    (458, [0, 3]),
    (970, [0, 4]),
    (1994, [0, 5]),
    (2048, [0, 6]),
]
