# Local

from helpers.check_dir_exists import check_dir_exists

# External

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from timeit import default_timer as timer
from argparse import ArgumentParser
