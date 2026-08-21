"""Decipherment Bench: from-scratch visual decipherment research stack.

Hard constraints (see CONVENTIONS.md §Hard constraints — never violate):
- no pretrained model weights anywhere outside contrib/openbook/;
- every run is preregistered in the experiment ledger before results are read;
- test splits are frozen and every read of them is audited.
"""

__version__ = "0.0.1"
