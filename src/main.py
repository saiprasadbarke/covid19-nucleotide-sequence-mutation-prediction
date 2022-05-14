# Call the train network function from here

from pathlib import Path


from data_operations.train_val_test_split import get_split_data

path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
print(path)
train_dataloader, val_dataloader, test_dataloader = get_split_data(path, 6)
for xy_values in train_dataloader:
    print()
