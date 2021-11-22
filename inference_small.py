from base import *
from cigam import *
from dataloader import *

# G, ranks = cigam.sample(1000, return_ranks=True, method='naive')

order_min, order_max = 2, 2
print('size') 

cigam = CIGAM(constrained=False, order_min=order_min, order_max=order_max)

G, ranks = cigam.sample(N=100, method='naive')
cigam.fit_model_given_ranks_torch(G, H=[1.0], features=ranks, learnable_ranks=False)
