import matplotlib.pyplot as plt

from base import *
from dataloader import *
from cigam import *
from hypergraph import *
from utils import *

N = 10
L = 1
H = [1]
K = 2
c = [1.5]
b = 3

cigam = CIGAM(b=b, c=c, H=H, order=K)
G, ranks = cigam.sample(N, method='naive', return_ranks=True)

M = G.num_simplices()

edges = np.zeros(shape=(M, K), dtype=np.int64)

for i, edge in enumerate(G.edges()):
    edges[i, :] = edge.to_index(np.array)

import pdb; pdb.set_trace()

with open('cigam_model.stan') as f:
    model_code = f.read()

stan_model = pystan.StanModel(model_code=model_code, model_name='fast model')

data = {
    'N' : N,
    'L' : L,
    'H' : H,
    'ranks' : ranks,


}

fit = stan_model.sampling(data=model_data, iter=100, chains=10, n_jobs=10)
