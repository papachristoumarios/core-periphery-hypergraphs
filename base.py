# Imports
import scipy.io
import os
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pandas as pd
import seaborn as sns
import random
import itertools
import pystan
import pickle

from sklearn import linear_model
from scipy.optimize import minimize
from scipy import stats
from scipy import special
from scipy import sparse as scipy_sparse
import sparse

# Headless environment for plotting
matplotlib.set('Agg')

# Aesthetics
LARGE_SIZE = 16
plt.rc('axes', labelsize=LARGE_SIZE)
plt.rc('axes', titlesize=LARGE_SIZE)
sns.set_theme()
