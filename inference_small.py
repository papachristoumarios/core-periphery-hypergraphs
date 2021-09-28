from base import *
from cigam import *
from dataloader import *

cigam = CIGAM(constrained=True, H=[1], order=2, c=[2.5])
G, _ = load_world_trade()
fit = cigam.fit_model_bayesian(G, np.linspace(0, 1, 6)[1:])
cigam.visualize_degree_plot(G, fit) 
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
