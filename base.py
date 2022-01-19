# Imports
import argparse
import scipy.io
import sys
import os
import copy
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pprint
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import sortedcontainers
import pandas as pd
import seaborn as sns
import random
import itertools
import stan
import pickle
import multiprocessing
import torch
import torch.nn as nn
from sklearn import linear_model, metrics
from scipy.optimize import minimize, curve_fit
from scipy import stats
from scipy import special
from scipy import sparse as scipy_sparse
from tqdm import tqdm
from numba import jit

import sparse

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Headless environment for plotting
matplotlib.use('Agg')

# Seeds
np.random.seed(0)
torch.manual_seed(0)

# Aesthetics
LARGE_SIZE = 20
plt.rc('axes', labelsize=LARGE_SIZE)
plt.rc('axes', titlesize=LARGE_SIZE)
sns.set_theme(palette='colorblind')
