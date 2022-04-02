# Standard

# Local
from typing import List

# External
from Bio.SeqRecord import SeqRecord


def check_sequence_completeness(sequence_record_object: SeqRecord) -> bool:
    if len(sequence_record_object) != 3822:
        return False
    elif "N" in sequence_record_object:
        return False
    else:
        return True


def remove_duplicate_sequences(
    claded_sequences: List[dict[str, str]]
) -> List[dict[str, str]]:
    return list({frozenset(item.items()): item for item in claded_sequences}.values())
