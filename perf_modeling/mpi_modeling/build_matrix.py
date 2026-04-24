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
    ALL_CNTRS,
    MTU,
    NUM_ALL_CNTRS,
    NUM_HIST_CNTRS,
    NUM_TC_CNTRS,
    RDZV_THRESHOLD,
    RX_HIST_SLICE,
    RX_TC_SLICE,
    SUB_MTU_MAP,
    TC_ACK,
    TC_DATA,
    TX_HIST_CNTRS,
    TX_HIST_SLICE,
    TX_TC_SLICE,
)


# =============================================================
# build_A — main function
# =============================================================
def build_matrix_a(msg_size_sets: np.ndarray) -> np.ndarray:
    """
    Build the system signature matrix A.

    Row layout:
        TX_HIST_SLICE --> TX histogram
        TX_TC_SLICE   --> TX traffic class
        RX_HIST_SLICE --> RX histogram
        RX_TC_SLICE   --> RX traffic class

    Column layout (2 * num_msg cols):
        cols 0:num_msg       --> sender-NIC view, one per message size bin
        cols num_msg:2*n_msg --> receiver-NIC view, one per message size bin

    TC assignment per protocol
    --------------------------
    eager  (msg_size <= RENDEZVOUS_THRESHOLD):
        sender TX --> TC_DATA,  sender RX --> TC_ACK
        recv   TX --> TC_ACK,   recv   RX --> TC_DATA

    rendezvous:
        sender TX --> TC_ACK,   sender RX --> TC_DATA   (flipped)
        recv   TX --> TC_DATA,  recv   RX --> TC_ACK    (also flipped)
    """
    num_msg_sizes = len(msg_size_sets)
    matrix_a = np.zeros((2 * NUM_ALL_CNTRS, 2 * num_msg_sizes), dtype=np.float64)

    for msg_idx, msg_size in enumerate(msg_size_sets):
        data_hist = _message_to_histogram(int(msg_size))
        num_data_pkts = int(np.sum(data_hist))
        ack_hist = _ack_histogram(int(msg_size))
        num_ack_pkts = int(np.sum(ack_hist))

        # ----------------------------------------------------------
        # Send column msg_idx — sender-NIC view
        # ----------------------------------------------------------
        matrix_a[TX_HIST_SLICE, msg_idx] = data_hist
        matrix_a[TX_TC_SLICE, msg_idx] = _tc_vector(
            msg_size, num_data_pkts, _tc_assignment(TC_DATA, int(msg_size))
        )
        matrix_a[RX_HIST_SLICE, msg_idx] = ack_hist
        matrix_a[RX_TC_SLICE, msg_idx] = _tc_vector(
            msg_size, num_ack_pkts, _tc_assignment(TC_ACK, int(msg_size))
        )

        # ----------------------------------------------------------
        # Recv column num_msg_sizes + msg_idx — receiver-NIC view
        # ----------------------------------------------------------
        matrix_a[TX_HIST_SLICE, num_msg_sizes + msg_idx] = ack_hist
        matrix_a[TX_TC_SLICE, num_msg_sizes + msg_idx] = _tc_vector(
            msg_size, num_ack_pkts, _tc_assignment(TC_ACK, int(msg_size))
        )
        matrix_a[RX_HIST_SLICE, num_msg_sizes + msg_idx] = data_hist
        matrix_a[RX_TC_SLICE, num_msg_sizes + msg_idx] = _tc_vector(
            msg_size, num_data_pkts, _tc_assignment(TC_DATA, int(msg_size))
        )

    return matrix_a


def _message_to_histogram(msg_size: int) -> np.ndarray:
    """
    Compute packet histogram for one MPI message.
    """
    hist_counters = np.zeros(NUM_HIST_CNTRS, dtype=np.uint32)
    msg_size = int(msg_size)

    # zero-size messages produce one packet in the smallest histogram bin, with no TC counts.
    if msg_size == 0:
        hist_counters[0] += 1
        return hist_counters

    # For sub-MTU messages, apply the empirically confirmed Cassini NIC mapping:
    if msg_size <= MTU:
        for upper, buckets in SUB_MTU_MAP:
            if msg_size <= upper:
                for b in buckets:
                    hist_counters[b] += 1
                return hist_counters

    # For messages larger than MTU, count one full MTU packet per MTU chunk, plus a final partial packet if needed.
    else:
        num_full_mtu = msg_size // MTU
        remainder = msg_size % MTU

        hist_counters[_packet_to_bucket(MTU)] += num_full_mtu

        if remainder > 0:
            for upper, buckets in SUB_MTU_MAP:
                if remainder <= upper:
                    for b in buckets:
                        hist_counters[b] += 1
                    break
        else:
            # If remainder is 0, an eager control packet is still sent!
            if msg_size <= RDZV_THRESHOLD:
                hist_counters[0] += 1

        # Rendezvous messages > MTU produce one additional control packet in the smallest histogram bin, i.e, 64B.
        if msg_size > RDZV_THRESHOLD:
            hist_counters[0] += 1

    return hist_counters


def _packet_to_bucket(pkt_size: int) -> int:
    """Map packet size (bytes, including header) to bucket index."""
    bucket_lower: list[int]
    bucket_upper: list[int]
    bucket_lower, bucket_upper = _parse_bucket_bounds(TX_HIST_CNTRS, prefix="hni_tx_ok_")

    for cntr_idx in range(NUM_HIST_CNTRS):
        if bucket_lower[cntr_idx] <= pkt_size <= bucket_upper[cntr_idx]:
            return cntr_idx
    raise ValueError(
        f"Packet size {pkt_size}B outside histogram range [{bucket_lower[0]}, {bucket_upper[-1]}]"
    )


def _parse_bucket_bounds(hist_names: list[str], prefix: str) -> tuple[list[int], list[int]]:
    """
    Parse histogram bucket lower/upper bounds from counter names.

    Parameters
    ----------
    hist_names : list of str
        e.g. TX_HIST_COUNTER_NAMES
    prefix : str
        e.g. "hni_tx_ok_" or "hni_rx_ok_"
    """
    lowers: list[int] = []
    uppers: list[int] = []

    for name in hist_names:
        suffix = name[len(prefix) :]
        if "_to_" in suffix:
            lo, hi = suffix.split("_to_")
            lowers.append(int(lo))
            uppers.append(int(hi))
        else:
            val = int(suffix)
            lowers.append(val)
            uppers.append(val)

    return lowers, uppers


def _ack_histogram(msg_size: int) -> np.ndarray:
    """Generate the hardware histogram for ACK/CTS packets sent by the receiver."""
    hist_counters = np.zeros(NUM_HIST_CNTRS, dtype=np.uint32)

    # ACK/CTS packets are always small and should fall into the first histogram bucket (64B) regardless of message size,
    if msg_size <= MTU:
        hist_counters[0] += 1
    # For messages larger than MTU, the rendezvous protocol produces one additional control packet in the smallest histogram bin, i.e, 64B.
    else:
        n_full = msg_size // MTU
        remainder = msg_size % MTU

        hist_counters[0] += n_full
        if remainder > 0:
            hist_counters[0] += 1

        # Rendezvous messages > MTU produce one additional control packet in the smallest histogram bin, i.e, 64B.
        if msg_size > RDZV_THRESHOLD:
            hist_counters[0] += 1

    return hist_counters


def _tc_vector(msg_size: int, num_pkts: int, tc_index: int) -> np.ndarray:
    """
    Build a traffic class counter vector.
    """
    if tc_index >= NUM_TC_CNTRS:
        raise ValueError(f"tc_index={tc_index} >= NUM_TC_CNTRS={NUM_TC_CNTRS}")
    tc = np.zeros(NUM_TC_CNTRS, dtype=np.float64)

    # For eager messages, all packets should be in the expected TC.
    if msg_size <= RDZV_THRESHOLD:
        tc[tc_index] = num_pkts

    # For rendezvous messages, TC flipping.
    else:
        # If there are exactly 2 traffic class counters (NUM_TC_CNTRS == 2), then other_tc is the opposite index.
        # Since indices are 0 and 1, 1 - tc_index flips between them.
        # If there's only 1 traffic class counter (or any other count), other_tc defaults to 0.
        other_tc = 1 - tc_index if NUM_TC_CNTRS == 2 else 0

        # Sets the other traffic class's counter to 1.
        # Sets the current traffic class's counter to num_pkts - 1, but never below 0:
        tc[other_tc] = 1
        tc[tc_index] = max(0, num_pkts - 1)

    return tc


def _tc_assignment(tc_index: int, msg_size: int) -> int:
    """
    Resolve the effective TC index for a sender-NIC counter slot.
    """
    # For eager messages, TC assignment is as-is.
    if msg_size <= RDZV_THRESHOLD:
        return tc_index

    # Rendezvous sender flip TC assignment: TC_DATA ↔ TC_ACK
    if tc_index == TC_DATA:
        return TC_ACK
    if tc_index == TC_ACK:
        return TC_DATA

    # This is a safety fallback when tc_index is neither TC_DATA nor TC_ACK.
    return tc_index


def validate_matrix_a(
    matrix_a: np.ndarray, msg_sizes: np.ndarray, target_size: int, count: int, case_name: str
):
    num_msgs = len(msg_sizes)

    # Find the index of the target message size
    try:
        idx = np.where(msg_sizes == target_size)[0][0]
    except IndexError:
        print(f"Error: Size {target_size}B not found in the message size set.")
        return

    # ---------------------------------------------------------
    # Sender Node Perspective
    # ---------------------------------------------------------
    x_sender = np.zeros(2 * num_msgs, dtype=np.float64)
    x_sender[idx] = count  # Index in the first half (Sender-NIC view)
    y_sender = matrix_a @ x_sender

    # ---------------------------------------------------------
    # Receiver Node Perspective
    # ---------------------------------------------------------
    x_receiver = np.zeros(2 * num_msgs, dtype=np.float64)
    x_receiver[num_msgs + idx] = count  # Index in the second half (Receiver-NIC view)
    y_receiver = matrix_a @ x_receiver

    # ---------------------------------------------------------
    # Print Results
    # ---------------------------------------------------------
    print("============================================================")
    print(f"{case_name}: {count:,} messages of {target_size:,} Bytes")
    print("============================================================")

    print("\n--- SENDER NODE COUNTERS ---")
    for name, val in zip(ALL_CNTRS, y_sender, strict=True):
        if val > 0:
            print(f"{name:<30} | {int(val):,}")

    print("\n--- RECEIVER NODE COUNTERS ---")
    for name, val in zip(ALL_CNTRS, y_receiver, strict=True):
        if val > 0:
            print(f"{name:<30} | {int(val):,}")
    print("\n")
