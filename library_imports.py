import random
from random import shuffle
import math
import statistics

import tensorflow as tf

import os
from os.path import exists

import datetime
import pickle
from joblib import dump, load

import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import JumpingKnowledge

import torch_geometric as tg
import torch_geometric.nn as tg_nn
import torch_geometric.utils as tg_utils
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import Sequential, JumpingKnowledge
from torch_geometric.transforms import NormalizeFeatures

from torch.nn import init
import pdb

from torch_geometric.data import Data, DataLoader

import torch.optim as optim
import torch_geometric.transforms as T

import torchvision
import torchvision.transforms as transforms

import time
from datetime import datetime

import multiprocessing as mp

# For visualizing the results
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pandas as pd
from torch_geometric.utils.convert import to_networkx

