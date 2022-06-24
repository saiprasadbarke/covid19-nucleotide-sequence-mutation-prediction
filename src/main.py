# Standard
from pprint import pprint as pp


# Local
from settings.constants import ROUTINES_TO_EXECUTE


pp(ROUTINES_TO_EXECUTE, indent=3)
activity_number = int(input("Choose a task to run :   "))
#################### Create new dataset #############
if activity_number == 1:
    from scripts.create_dataset import create_dataset

    create_dataset()
################ Call the train network function ############
elif activity_number == 2:
    from scripts.train import train

    train()
#################### Inference #####################
elif activity_number == 3:
    from scripts.inference import inference

    inference()
