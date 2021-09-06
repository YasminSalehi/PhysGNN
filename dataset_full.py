from dataset_1_formatter import *
from dataset_2_formatter import *

# Generate Dataset 1
for i in range(11): # We have 11 nodes
    dataset_1(i+1)

# Generate Dataset 2
for i in range(11): # We have 11 x 15 directions. forces are saved in 11 batches.
    dataset_2(i+1)
