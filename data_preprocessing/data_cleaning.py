# Standard

# Local

# External
from Bio.SeqRecord import SeqRecord


def check_sequence_completeness(sequence_record_object: SeqRecord) -> bool:
    if len(sequence_record_object) != 3822:
        return False
    elif "N" in sequence_record_object:
        return False
    else:
        return True


def remove_duplicate_sequences(sequences):
    return None


def clip_sequence(sequence_begin: int, sequence_end: int):
    return None
