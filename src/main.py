# Call the train network function from here

from pathlib import Path


from data_operations.train_val_test_split import get_split_data
from model_training.train_model import train_loop
from model_components.create_model import create_model
from data_operations.vocabulary import Vocabulary

path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
print(path)

train_dataloader, val_dataloader, test_dataloader = get_split_data(path, 2)
model = create_model(len(Vocabulary(3)), len(Vocabulary(3)), emb_size=256, hidden_size=256, num_layers=1, dropout=0)
dev_preplex = train_loop(model=model, train_dataloader=train_dataloader, validation_dataloader=val_dataloader)
