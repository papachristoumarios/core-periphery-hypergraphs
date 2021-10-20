from base import *
from cigam import *
from dataloader import *

b = 3
H = [1]
c = np.linspace(1.5, 2.5, len(H))
cigam = CIGAM(constrained=False, H=H, order=2, c=c, b=b)
# G, ranks = cigam.sample(100, return_ranks=True, method='naive')

G, _ = load_world_trade()
fit = cigam.fit_model_bayesian(G, H, ranks=None)

cigam.visualize_posterior(fit, params=['lambda', 'c'], method='pairplot', outfile='posterior_fast')


# cigam.visualize_degree_plot(G.clique_decomposition(), fit) 
# cigam.visualize_degree_plot(G, fit)
# cigam.visualize_posterior(fit, params=['lambda', 'c'], pairplot=True)


#cigam = CIGAM()
#G = load_celegans()
#fit = cigam.fit_model_bayesian(G, [3])
#cigam.visualize_degree_plot(G, fit)

# cigam = CIGAM()
# G = load_faculty(location='/data/mp2242//faculty/History_edgelist.txt', relabel=True)
# fit = cigam.fit_model_bayesian(G, [4])
# cigam.visualize_degree_plot(G, fit)

# cigam = CIGAM()
# G = load_faculty(location='/data/mp2242/faculty/Business_edgelist.txt', relabel=True)
# fit = cigam.fit_model_bayesian(G, [4])
# cigam.visualize_degree_plot(G, fit)
