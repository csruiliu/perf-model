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
from typing import List, Tuple

from constants import (
    N_HIST, N_TC, M,
    TX_HIST_SLICE, TX_TC_SLICE,
    RX_HIST_SLICE, RX_TC_SLICE,
    MTU, HEADER_SIZE,
    BUCKET_LOWER, BUCKET_UPPER,
    TC_DATA, TC_ACK,
    MSG_SIZES,
    RDZV_THRESHOLD
)

# =============================================================
# Empirically confirmed sub-MTU fragmentation map
# =============================================================
_SUB_MTU_RANGES: List[Tuple[int, List[int]]] = [
    (11,   [0]),
    (74,   [1]),
    (192,  [2]),
    (202,  [0, 2]),
    (458,  [0, 3]),
    (970,  [0, 4]),
    (1994, [0, 5]),
    (2048, [0, 6]),
]


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
    """
    hist = np.zeros(N_HIST, dtype=np.uint32)
    msg_size = int(msg_size)

    if msg_size == 0:
        hist[0] += 1
        return hist

    if msg_size <= MTU:
        for upper, buckets in _SUB_MTU_RANGES:
            if msg_size <= upper:
                for b in buckets:
                    hist[b] += 1
                return hist

    else:
        n_full = msg_size // MTU
        remainder = msg_size %  MTU

        hist[_packet_to_bucket(MTU + HEADER_SIZE)] += n_full
        
        if remainder > 0:
            for upper, buckets in _SUB_MTU_RANGES:
                if remainder <= upper:
                    for b in buckets:
                        hist[b] += 1
                    break
        
        if msg_size > RDZV_THRESHOLD:
            hist[0] += 1

    return hist


def _ack_histogram(msg_size: int) -> np.ndarray:
    """Generate the hardware histogram for ACK/CTS packets sent by the receiver."""
    hist = np.zeros(N_HIST, dtype=np.uint32)
    
    if msg_size <= MTU:
        hist[0] += 1
    else:
        n_full = msg_size // MTU
        remainder = msg_size % MTU
        
        hist[0] += n_full
        if remainder > 0:
            hist[0] += 1
            
        if msg_size > RDZV_THRESHOLD:
            hist[0] += 1
            
    return hist


def _tc_vector(msg_size: int, n_packets: int, tc_index: int) -> np.ndarray:
    """
    Build a traffic class counter vector.
    """
    if tc_index >= N_TC:
        raise ValueError(f"tc_index={tc_index} >= N_TC={N_TC}")
    tc = np.zeros(N_TC, dtype=np.float64)

    if msg_size <= RDZV_THRESHOLD:
        tc[tc_index] = n_packets
    else:
        other_tc = 1 - tc_index if N_TC == 2 else 0
        tc[other_tc] = 1
        tc[tc_index] = max(0, n_packets - 1)

    return tc


def _transmit_tc(tc_eager: int, msg_size: int) -> int:
    """
    Resolve the effective TC index for a sender-NIC counter slot.

    Under the eager protocol (msg_size <= RENDEZVOUS_THRESHOLD) TC
    assignments are straightforward: TC_DATA carries data packets,
    TC_ACK carries acknowledgements.

    Under the rendezvous protocol (msg_size > RENDEZVOUS_THRESHOLD) the
    sender-NIC hardware flips tc0 and tc1:
        tc0 --> response  (ACK / CTS arriving at the sender)
        tc1 --> request   (data departing from the sender)
    """
    if msg_size <= RDZV_THRESHOLD:
        return tc_eager

    # Rendezvous sender flip
    if tc_eager == TC_DATA:
        return TC_ACK
    if tc_eager == TC_ACK:
        return TC_DATA
    return tc_eager


def _receive_tc(tc_eager: int, msg_size: int) -> int:
    """
    Resolve the effective TC index for a receiver-NIC counter slot.

    Under the eager protocol (msg_size <= RENDEZVOUS_THRESHOLD) TC
    assignments are straightforward: TC_DATA carries data packets,
    TC_ACK carries acknowledgements.

    Under the rendezvous protocol (msg_size > RENDEZVOUS_THRESHOLD) the
    receiver-NIC hardware is assumed to apply the same tc0/tc1 flip as
    the sender NIC:
        tc0 → response  (data arriving at the receiver)
        tc1 → request   (ACK / CTS departing from the receiver)

    NOTE: This flip is assumed symmetric with the sender-NIC flip but has
    not yet been independently confirmed against hardware. Update the body
    of this function once empirical data is available without touching
    _sender_tc or build_matrixA.
    """
    if msg_size <= RDZV_THRESHOLD:
        return tc_eager

    # Rendezvous receiver flip: TC_DATA ↔ TC_ACK
    # *** assumed symmetric with sender — update once hardware-confirmed ***
    if tc_eager == TC_DATA:
        return TC_ACK
    if tc_eager == TC_ACK:
        return TC_DATA
    return tc_eager   # any other TC classes are unaffected

# =============================================================
# Sub-MTU mapping validation
# =============================================================
def _validate_sub_mtu_mapping() -> None:
    """Confirm _message_to_histogram matches the empirically confirmed Cassini NIC mapping."""
    boundary_cases: List[Tuple[int, List[int]]] = [
        (1,    [0]),
        (11,   [0]),
        (12,   [1]),
        (74,   [1]),
        (75,   [2]),
        (192,  [2]),
        (193,  [0, 2]),
        (202,  [0, 2]),
        (203,  [0, 3]),
        (458,  [0, 3]),
        (459,  [0, 4]),
        (970,  [0, 4]),
        (971,  [0, 5]),
        (1994, [0, 5]),
        (1995, [0, 6]),
        (2048, [0, 6]),
    ]

    errors: List[str] = []
    for msg_size, expected in boundary_cases:
        hist = _message_to_histogram(msg_size)
        actual = sorted(int(b) for b in np.where(hist > 0)[0])
        if actual != expected:
            errors.append(
                f"  msg_size={msg_size:>5}B : "
                f"expected buckets {expected}, got {actual}"
            )

    if errors:
        raise AssertionError(
            "Sub-MTU empirical mapping validation failed:\n"
            + "\n".join(errors)
        )

    print(f"  [OK] Sub-MTU mapping: all {len(boundary_cases)} boundary cases pass")


# =============================================================
# Duplicate column checker
# =============================================================
def check_duplicate_columns(A: np.ndarray, msg_sizes: np.ndarray) -> None:
    """Detect message size bins whose NIC counter fingerprints are identical."""
    N = len(msg_sizes)
    seen: dict = {}
    duplicates: list = []

    for j in range(N):
        key = tuple(np.round(A[:, j], 8))
        if key in seen:
            duplicates.append((seen[key], j))
        else:
            seen[key] = j

    if not duplicates:
        print("  [OK] All message size bins produce distinct NIC fingerprints")
        return

    print(f"  [WARN] {len(duplicates)} duplicate column pair(s) detected — "
          f"hardware cannot distinguish these bins:")
    for k, j in duplicates:
        m_k, m_j = int(msg_sizes[k]), int(msg_sizes[j])
        s_k = f"{m_k // 1024}KB" if m_k >= 1024 else f"{m_k}B"
        s_j = f"{m_j // 1024}KB" if m_j >= 1024 else f"{m_j}B"
        hist_k = _message_to_histogram(m_k)
        n_pkts = int(np.sum(hist_k))
        print(f"    bin {k:>3} ({s_k:>8})  ==  bin {j:>3} ({s_j:>8})"
              f"  [n_pkts={n_pkts}, hist={hist_k.astype(int).tolist()}]")


# =============================================================
# build_A — main function
# =============================================================
def build_matrixA(msg_sizes: np.ndarray = MSG_SIZES) -> np.ndarray:
    """
    Build the system signature matrix A.

    Row layout (TWO_M rows):
        TX_HIST_SLICE --> TX histogram
        TX_TC_SLICE   --> TX traffic class
        RX_HIST_SLICE --> RX histogram
        RX_TC_SLICE   --> RX traffic class

    Column layout (2*N_msg cols):
        cols 0:N_msg       --> sender-NIC view, one per message size bin
        cols N_msg:2*N_msg --> receiver-NIC view, one per message size bin

    TC assignment per protocol
    --------------------------
    eager  (msg_size <= RENDEZVOUS_THRESHOLD):
        sender TX --> TC_DATA,  sender RX --> TC_ACK
        recv   TX --> TC_ACK,   recv   RX --> TC_DATA

    rendezvous (msg_size > RENDEZVOUS_THRESHOLD):
        sender TX --> TC_ACK,   sender RX --> TC_DATA   (flip via _sender_tc)
        recv   TX --> TC_ACK,   recv   RX --> TC_DATA   (unchanged)
    """
    print("  Validating sub-MTU empirical mapping...")
    _validate_sub_mtu_mapping()

    N = len(msg_sizes)
    A = np.zeros((2 * M, 2 * N), dtype=np.float64)

    for j, msg_size in enumerate(msg_sizes):
        msg_sz = int(msg_size)
        data_hist = _message_to_histogram(msg_sz)
        n_data_pkts = int(np.sum(data_hist))
        ack_hist = _ack_histogram(n_data_pkts)
        n_ack_pkts = int(np.sum(ack_hist))

        # ----------------------------------------------------------
        # Send column j — sender-NIC view
        # ----------------------------------------------------------
        A[TX_HIST_SLICE, j] = data_hist
        A[TX_TC_SLICE, j] = _tc_vector(msg_size, n_data_pkts, _transmit_tc(TC_DATA, msg_sz))
        A[RX_HIST_SLICE, j] = ack_hist
        A[RX_TC_SLICE, j] = _tc_vector(msg_size, n_ack_pkts,  _transmit_tc(TC_ACK,  msg_sz))

        # ----------------------------------------------------------
        # Recv column N+j — receiver-NIC view
        # ----------------------------------------------------------
        A[TX_HIST_SLICE, N+j] = ack_hist
        A[TX_TC_SLICE, N+j] = _tc_vector(msg_size, n_ack_pkts,  _receive_tc(TC_ACK,  msg_sz))
        A[RX_HIST_SLICE, N+j] = data_hist
        A[RX_TC_SLICE, N+j] = _tc_vector(msg_size, n_data_pkts, _receive_tc(TC_DATA, msg_sz))

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
    check_duplicate_columns(A, msg_sizes)
    print()
    print(f"  {'Bin':>3}  {'MsgSize':>10}  {'n_pkts':>7}  "
          f"{'TX_hist':>8}  {'RX_hist':>8}")
    print("  " + "-" * 46)
    for j, m in enumerate(msg_sizes):
        hist = _message_to_histogram(int(m))
        n_pkts = int(np.sum(hist))
        tx_hist_sum = int(np.sum(A[TX_HIST_SLICE, j]))
        rx_hist_sum = int(np.sum(A[RX_HIST_SLICE, j]))
        size_str = f"{int(m / 1024)}KB" if m >= 1024 else f"{int(m)}B"
        print(f"  {j:>3}  {size_str:>10}  {n_pkts:>7}  "
              f"{tx_hist_sum:>8}  {rx_hist_sum:>8}")