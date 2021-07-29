from base import *
from cigam import *

LARGE_SIZE = 20
plt.rc('axes', labelsize=LARGE_SIZE)
plt.rc('axes', titlesize=LARGE_SIZE)
plt.rc('figure', facecolor='white')

cigam = CIGAM(H = [1,], c=[1.5,], b=3)
n = 1000

cmap = plt.get_cmap("magma")
P, _ = cigam.bias_matrix(n)

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.grid(b=None)
plt.title('Bias Matrix for an {}-node 2-uniform hypergraph'.format(int(n)))
plt.imshow(cmap(1-P))
plt.savefig('biases_2_uniform.png')

cigam = CIGAM(H = [1,], c=[1.5], b=3, order=3)
n = 20

P, _ = cigam.bias_matrix(n)
x, y, z = np.indices((n, n, n))
voxels = (z <= y) & (z <= x) & (y <= x)

ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
ax.voxels(voxels, facecolors=cmap(P), alpha=0.5, edgecolors=None, shade=False)
ax.set_facecolor('white')
plt.axis('off')
plt.grid(b=None)
plt.title('Voxel Plot for a {}-node 3-uniform hypergraph'.format(int(n)))
plt.savefig('biases_3_uniform.png')

