from base import *
from cigam import *

cigam = CIGAM()
G = load_world_trade()
fit = cigam.fit_model_bayesian(G, [5])
cigam.visualize_degree_plot(G, fit)
cigam.visualize_posterior(fit, params=['lambda', 'c'], pairplot=True)

cigam = CIGAM()
G = load_celegans()
fit = cigam.fit_model_bayesian(G, [3])
cigam.visualize_degree_plot(G, fit)

cigam = CIGAM()
G = load_faculty(location='/content/drive/My Drive/NNIM/data/faculty/History_edgelist.txt', relabel=True)
fit = cigam.fit_model_bayesian(G, [4])
cigam.visualize_degree_plot(G, fit)

cigam = CIGAM()
G = load_faculty(location='/content/drive/My Drive/NNIM/data/faculty/Business_edgelist.txt', relabel=True)
fit = cigam.fit_model_bayesian(G, [4])
cigam.visualize_degree_plot(G, fit)
