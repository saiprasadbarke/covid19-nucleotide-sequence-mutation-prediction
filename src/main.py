# Call the train network function from here

from pathlib import Path


from data_operations.train_val_test_split import get_split_data
from model_training.train_model import train_loop
from model_components.create_model import create_model
from data_operations.vocabulary import Vocabulary
from settings.constants import (
    EMBEDDING_SIZE,
    LEN_VOCABULARY,
    MINIBATCH_SIZE,
    RNN_DROPOUT,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
)


dataset_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
print(f"Path to dataset : {dataset_file_path}")
train_dataloader, val_dataloader, test_dataloader = get_split_data(dataset_file_path, MINIBATCH_SIZE)
model = create_model(
    vocab_size=LEN_VOCABULARY,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=RNN_HIDDEN_SIZE,
    num_layers=RNN_NUM_LAYERS,
    dropout=RNN_DROPOUT,
)
dev_preplex = train_loop(model=model, train_dataloader=train_dataloader, validation_dataloader=val_dataloader)
