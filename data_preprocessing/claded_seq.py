from dataclasses import dataclass


@dataclass
class CladedSequence:
    sequence: str
    clade: str
