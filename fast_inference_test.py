import matplotlib.pyplot as plt

from base import *
from dataloader import *
from cigam import *
from hypergraph import *
from utils import *

N = 100
H = [1]
K = 3
c = [2.5]
L = len(H)
b = 3

cigam = CIGAM(b=b, c=c, H=H, order=K)
G, ranks = cigam.sample(N, method='naive', return_ranks=True)

M = G.num_simplices()

edges = np.zeros(shape=(M, K), dtype=np.int64)

for i, edge in enumerate(G.edges()):
    edges[i, :] = edge.to_index(np.array)

# import pdb; pdb.set_trace()

with open('cigam_test.stan') as f:
    model_code = f.read()

stan_model = pystan.StanModel(model_code=model_code, model_name='fastmodel')
binomial_coeffs = binomial_coefficients(N, K)


model_data = {
    'N' : N,
    'L' : L,
    'H' : H,
    'K' : K,
    'ranks' : ranks,
    'binomial_coefficients' : binomial_coeffs,
    'M' : M,
    'edges' : edges
}

fit = stan_model.sampling(data=model_data, iter=1000, chains=1, n_jobs=1)

print(fit['c'].mean(0))
print(np.exp(fit['lambda'].mean(0)))
