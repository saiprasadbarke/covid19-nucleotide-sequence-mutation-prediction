# Standard
from typing import List

# Local
from globals.constants import list_of_clades

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


def remove_clades(claded_sequences: List[dict[str, str]]) -> List[dict[str, str]]:
    new_claded_sequences = []
    for claded_sequence in claded_sequences:
        if claded_sequence["clade"] in list_of_clades:
            new_claded_sequences.append(claded_sequence)
    return new_claded_sequences
