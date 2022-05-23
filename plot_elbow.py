from base import *
from cigam import *
from dataloader import *

num_layers_agg = []
bics_agg = []
log_err_agg = []

for _ in range(10):
    b = 3
    l = 4
    H = np.linspace(0, 1, l + 1)[1:]
    c = np.linspace(1.5, 2.5, len(H))
    cigam = CIGAM(constrained=False, H=H, order_min=2, order_max=2, c=c, b=b)

    G, ranks = cigam.sample(1000, return_ranks=True, method='naive')

    num_layers, log_err, breakpoints = cigam.find_hyperparameters(G=G, eps=0.3, layers_max=6)
    bics = []
    
    import pdb; pdb.set_trace()

    #for l in num_layers:
    #    H = 2.0**(-np.arange(l, dtype=np.float64))[::-1]
    #    c = np.linspace(1.5, 2.5, len(H))
    #    cigam = CIGAM(constrained=False, H=H, order_min=2, order_max=2, c=c, b=b)
    #    fit = cigam.fit_model_bayesian(G, H, ranks=ranks)

    #    bic = (l + 1) * np.log(len(G) + G.num_simplices()) - 2 * fit['lp__'].max()

    #   bics.append(bic)

    num_layers_agg.append(num_layers)
    #bics_agg.append(bics)
    log_err_agg.append(log_err)

# bics_agg = np.array(bics_agg)
log_err_agg = np.array(log_err_agg)

fig, ax1 = plt.subplots(figsize=(10, 10))

# ax2 = ax1.twinx()

ax1.plot(num_layers, log_err_agg.mean(0), color='b', marker='x', linewidth=4)
ax1.fill_between(num_layers, log_err_agg.mean(0) - log_err_agg.std(0), log_err_agg.mean(0) + log_err_agg.std(0), color='b', alpha=0.3)
#ax2.plot(num_layers, bics_agg.mean(0), color='g', marker='o', linewidth=4)
#ax2.fill_between(num_layers, bics_agg.mean(0) - bics_agg.std(0), bics_agg.mean(0) + bics_agg.std(0), color='g', alpha=0.3)
ax1.set_xlabel('Number of Layers', fontsize=16)
ax1.set_ylabel('Log Piecewise Linear Fit Error', fontsize=16, color='b')
#ax2.set_ylabel('Bayesian Information Criterion', fontsize=16, color='g')
plt.title('Hyperparameter Selection on Synthetic Data', fontsize=16) 

plt.savefig('elbow.png')
