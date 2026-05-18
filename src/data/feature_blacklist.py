"""Features to drop from CIC-IDS2017.

Critical for cross-attack generalization (S3): port numbers and IPs are
attack-specific signatures, not generalizable patterns.

Reasoning:
- Source IP, Destination IP, Flow ID: identifiers, not features
- Timestamp: temporal leak, will not exist at deployment
- Source Port, Destination Port: DDoS targets port 80, PortScan touches many
  ports -- including ports lets the model memorize the attack type instead of
  learning behavior. Removing ports is essential for S3 to be a fair test.
- Label, binary_label: targets, never features

Add more here if you find features that leak.
"""
from __future__ import annotations

# Identifiers and metadata - drop always
IDENTIFIER_COLS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
]

# Port-based features - drop to prevent attack signature leakage
PORT_COLS = [
    "Source Port",
    "Src Port",
    "Destination Port",
    "Dst Port",
]

# Label columns - always present in data, never used as features
LABEL_COLS = [
    "Label",
    " Label",
    "label",
    "binary_label",
]

# Protocol number leaks attack type: specific attacks use specific protocols
# (e.g. ICMP for some DoS variants, TCP for DDoS LOIT). Including Protocol
# lets the model use it as a shortcut for attack identity, which inflates S1
# results and undermines the S3 cross-attack test.
# Document this decision in the methodology: identifiers, ports, timestamps,
# and protocol are removed to reduce shortcut learning.
SUSPICIOUS_COLS = [
    "Protocol",
]


def get_blacklist() -> list[str]:
    """All columns to drop before training."""
    return IDENTIFIER_COLS + PORT_COLS + LABEL_COLS + SUSPICIOUS_COLS
