# Call the train network function from here

from pathlib import Path

from data_operations.encode_sequences_onehot import kmer_to_onehot
from data_operations.train_val_test_split import get_split_data

path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
print(path)


from data_operations.vocabulary import Vocabulary

print(Vocabulary(3).__str__())
