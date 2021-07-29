from base import *
from cigam import *
from utils import *
from hypergraph import *
import hypernetx as hnx

cigam = CIGAM(H=[0.25, 1], c=[1.5, 2.5], b=3)

n = 100

H, heights = cigam.sample(n, return_ranks=False, method='naive')

G = H.clique_decomposition()
pos = {}
nodecolor = []
nodesize = []

cmap = plt.get_cmap('magma')

for i, h in enumerate(heights):
    theta = np.random.uniform() * 2 * np.pi
    pos[i] = (h * np.cos(theta), h * np.sin(theta))
    nodecolor.append(h)
    nodesize.append(100 * (1-h))

edgecolor = []
edgesize = []

for (u, v) in G.edges():
    edgecolor.append(-np.log(min(heights[u], heights[v])))
    edgesize.append(2 * (1 - min(heights[u], heights[v])))

ax = plt.figure(figsize=(10, 10))

nx.draw_networkx_nodes(G, pos=pos, node_color=nodecolor, node_size=nodesize)
nx.draw_networkx_edges(G, pos=pos, edge_color=edgecolor, width=edgesize, alpha=0.3)

ax.set_facecolor('white')
plt.axis('off')
plt.grid(b=None)

plt.savefig('instance_2_uniform.png')

cigam.order = 3
n = 10

H, heights = cigam.sample(n, return_ranks=False, method='naive')
simplices = {}
simplices_colors = []
pos = {}

for i, h in enumerate(heights):
    theta = np.random.uniform() * 2 * np.pi
    pos[str(i)] = (h * np.cos(theta), h * np.sin(theta))


for i, edge in enumerate(H.edges()):
    simplices[i] = tuple([str(x) for x in edge.to_index(dtype=tuple)])
    simplices_colors.append(-np.log(heights[edge.to_index(dtype=np.array)].min()))

simplices_colors = np.array(simplices_colors)

H_hnx = hnx.Hypergraph(simplices)

norm = plt.Normalize(simplices_colors.min(), simplices_colors.max())

norm_nodes = plt.Normalize(heights.min(), heights.max())

ax = plt.figure(figsize=(10, 10))

hnx.drawing.draw(H_hnx,
                label_alpha=0,
                pos=pos,
                edges_kwargs={
                    'facecolors': cmap(norm(simplices_colors))*(1, 1, 1, 0.2),
                    'edgecolors': 'black',
                    'linewidths': 1
                },
                nodes_kwargs={
                    'facecolor' : cmap(norm_nodes(heights))*(1, 1, 1, 0.2)
                },
                )

plt.savefig('instance_3_uniform.png')
