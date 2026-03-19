"""
build_matrix.py

Builds the system signature matrix A:

    y = A * x

All constants imported from constants.py.

Dimensions:
    A : shape (TWO_M, 2*N_msg)
    x : shape (2*N_msg,) — [sends; recvs]
    y : shape (TWO_M,) — [TX_hist; TX_TC; RX_hist; RX_TC]
"""

import numpy as np

from constants import (
    N_HIST, N_TC, M, TWO_M,
    TX_HIST_SLICE, TX_TC_SLICE,
    RX_HIST_SLICE, RX_TC_SLICE,
    MTU, HEADER_SIZE, PAYLOAD_MAX,
    BUCKET_LOWER, BUCKET_UPPER,
    TC_DATA, TC_ACK,
    MSG_SIZES, N_MSG,
)

# =============================================================
# Helpers
# =============================================================
def _packet_to_bucket(pkt_size: int) -> int:
    """Map packet size (bytes, including header) to bucket index."""
    for b in range(N_HIST):
        if BUCKET_LOWER[b] <= pkt_size <= BUCKET_UPPER[b]:
            return b
    raise ValueError(
        f"Packet size {pkt_size}B outside histogram range "
        f"[{BUCKET_LOWER[0]}, {BUCKET_UPPER[-1]}]"
    )


def _message_to_histogram(msg_size: int) -> np.ndarray:
    """
    Compute packet histogram for one MPI message.

    Parameters
    ----------
    msg_size : int   MPI message payload in bytes

    Returns
    -------
    hist : np.ndarray, shape (N_HIST,)
    """
    hist = np.zeros(N_HIST, dtype=np.float64)
    msg_size = int(msg_size)

    if msg_size == 0:
        hist[_packet_to_bucket(HEADER_SIZE)] += 1
        return hist

    if msg_size <= PAYLOAD_MAX:
        hist[_packet_to_bucket(msg_size + HEADER_SIZE)] += 1
    else:
        n_full = msg_size // PAYLOAD_MAX
        remainder = msg_size %  PAYLOAD_MAX
        hist[_packet_to_bucket(MTU)] += n_full
        if remainder > 0:
            hist[_packet_to_bucket(remainder + HEADER_SIZE)] += 1

    return hist


def _ack_histogram(n_data_pkts: int) -> np.ndarray:
    """
    ACK histogram — all ACKs are HEADER_SIZE bytes → bucket 0.

    Parameters
    ----------
    n_data_pkts : int

    Returns
    -------
    hist : np.ndarray, shape (N_HIST,)
    """
    hist = np.zeros(N_HIST, dtype=np.float64)
    hist[_packet_to_bucket(HEADER_SIZE)] += n_data_pkts
    return hist


def _tc_vector(n_packets: int, tc_index: int) -> np.ndarray:
    """
    Build a traffic class counter vector.

    Parameters
    ----------
    n_packets : int
    tc_index  : int   (0 to N_TC-1)

    Returns
    -------
    tc : np.ndarray, shape (N_TC,)
    """
    if tc_index >= N_TC:
        raise ValueError(f"tc_index={tc_index} >= N_TC={N_TC}")
    tc = np.zeros(N_TC, dtype=np.float64)
    tc[tc_index] = n_packets
    return tc


# =============================================================
# build_A — main function
# =============================================================
def build_matrixA(msg_sizes: np.ndarray = MSG_SIZES) -> np.ndarray:
    """
    Build the system signature matrix A.

    Row layout (TWO_M rows):
        TX_HIST_SLICE → TX histogram
        TX_TC_SLICE   → TX traffic class
        RX_HIST_SLICE → RX histogram
        RX_TC_SLICE   → RX traffic class

    Column layout (2*N_msg cols):
        cols 0:N_msg       → Send, one per message size bin
        cols N_msg:2*N_msg → Recv, one per message size bin

    Parameters
    ----------
    msg_sizes : np.ndarray, shape (N_msg,)

    Returns
    -------
    A : np.ndarray, shape (TWO_M, 2*N_msg) = (30, 38)
    """
    N = len(msg_sizes)
    A = np.zeros((TWO_M, 2 * N), dtype=np.float64)

    for j, msg_size in enumerate(msg_sizes):
        data_hist   = _message_to_histogram(int(msg_size))
        n_data_pkts = int(np.sum(data_hist))
        ack_hist    = _ack_histogram(n_data_pkts)
        n_ack_pkts  = n_data_pkts

        # Send column j — sender NIC
        A[TX_HIST_SLICE, j] = data_hist
        A[TX_TC_SLICE,   j] = _tc_vector(n_data_pkts, TC_DATA)
        A[RX_HIST_SLICE, j] = ack_hist
        A[RX_TC_SLICE,   j] = _tc_vector(n_ack_pkts,  TC_ACK)

        # Recv column N+j — receiver NIC
        A[TX_HIST_SLICE, N+j] = ack_hist
        A[TX_TC_SLICE,   N+j] = _tc_vector(n_ack_pkts,  TC_ACK)
        A[RX_HIST_SLICE, N+j] = data_hist
        A[RX_TC_SLICE,   N+j] = _tc_vector(n_data_pkts, TC_DATA)

    return A


# =============================================================
# Diagnostics
# =============================================================
def print_matrix_summary(A: np.ndarray, msg_sizes: np.ndarray = MSG_SIZES) -> None:
    """Print a human-readable summary of A for verification."""
    print(f"A shape          : {A.shape}  "
          f"(TWO_M={A.shape[0]}, 2*N_msg={A.shape[1]})")
    print(f"N_HIST           : {N_HIST}")
    print(f"N_TC             : {N_TC}")
    print(f"Condition number : {np.linalg.cond(A):.3e}")
    print(f"Non-zero entries : {np.count_nonzero(A)} / {A.size}")
    print(f"Bucket bounds    : {list(zip(BUCKET_LOWER, BUCKET_UPPER))}")
    print()
    print(f"  {'Bin':>3}  {'MsgSize':>10}  {'n_pkts':>7}  "
          f"{'TX_hist':>8}  {'RX_hist':>8}")
    print("  " + "-" * 46)
    for j, m in enumerate(msg_sizes):
        hist = _message_to_histogram(int(m))
        n_pkts = int(np.sum(hist))
        tx_hist_sum = int(np.sum(A[TX_HIST_SLICE, j]))
        rx_hist_sum = int(np.sum(A[RX_HIST_SLICE, j]))
        size_str = f"{int(m/1024)}KB" if m >= 1024 else f"{int(m)}B"
        print(f"  {j:>3}  {size_str:>10}  {n_pkts:>7}  "
              f"{tx_hist_sum:>8}  {rx_hist_sum:>8}")