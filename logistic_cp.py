from base import *
from hypergraph import *
from utils import *
from dataloader import * 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticCP:

    def __init__(self, thetas, order=2):
        self.thetas = - np.sort(-thetas)
        self.order = order

    def sample(self, n):
        G = Hypergraph()

        for edge in itertools.combinations(range(n), self.order):
            edge = np.array(edge)
            p_edge = sigmoid(self.thetas[edge].sum())
            if np.random.unifomr() <= p_edge:
                G.add_simplex_from_nodes(nodes=edge.tolist(), simplex_data = {})

        return G, thetas

    def plot_sample(self, n):
        G, thetas = self.sample(n)
        G = G.clique_decomposition()
        plt.figure(figsize=(10, 10))
        plt.imshow(A)
        plt.title('Adjacency Matrix for G ~ logstic-CP($\mu$={}, $a$={}, $b$={})'.format(self.mu, self.alpha, self.beta))
        plt.xlabel('Ranked Nodes by $\\theta(u)$')
        plt.ylabel('Ranked Nodes by $\\theta(u)$')
        plt.figure(figsize=(10, 10))
        log_rank = np.log(1 + np.arange(A.shape[0]))
        log_degree = np.log(1 + A.sum(0))
        plt.plot(log_rank, log_degree)
        plt.title('Degree Plot')
        plt.xlabel('Node Rank by $\\theta(u)$ (log)')
        plt.ylabel('Node Degree (log)')

        plt.figure(figsize=(10, 10))
        degree_ranks = np.argsort(-log_degree)
        for i in range(A.shape[0]):
            A[i, :] = A[i, degree_ranks]

        for i in range(A.shape[1]):
            A[:, i] = A[degree_ranks, i]

        log_degree = log_degree[degree_ranks]
        plt.imshow(A)
        plt.title('Adjacency Matrix for G ~ logstic-CP($\mu$={}, $a$={}, $b$={})'.format(self.mu, self.alpha, self.beta))
        plt.xlabel('Ranked Nodes by degree')
        plt.ylabel('Ranked Nodes by degree')
        plt.figure(figsize=(10, 10))

        plt.plot(log_rank, log_degree)
        p = np.polyfit(log_rank, log_degree, deg=1)
        alpha_lasso = 0.1
        r2 = np.corrcoef(log_rank, log_degree)[0, 1]
        plt.plot(log_rank, log_degree, linewidth=1, label='Realized Degree $R^2 = {}$'.format(round(r2, 2)))
        plt.plot(log_rank, p[1] + p[0] * log_rank, linewidth=2, label='Linear Regression')
        plt.title('Degree Plot')
        plt.xlabel('Node Rank by degree (log)')
        plt.ylabel('Node Degree (log)')
        plt.legend()

    @staticmethod
    def graph_log_likelihood(edge_set, n, order, thetas, negative_samples):

        log_likelihood_pos = 0
        log_likelihood_neg = 0

        for edge in edge_set:
            edge_index = np.array(edge)
            log_likelihood_pos += np.log(sigmoid(thetas[edge_index].sum()))

        neg_edge_set = set([])
        
        for _ in range(negative_samples):
            while True:
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    break
            neg_edge_index = np.array(neg_edge)
            log_likelihood_neg += np.log(1 - sigmoid(thetas[neg_edge_index].sum()))
            neg_edge_set.add(neg_edge)

        return log_likelihood_pos + (n / negative_samples) * log_likelihood_neg

    @staticmethod
    def graph_log_likelihood_jacobian(edge_set, n, order, thetas, negative_samples):

        log_likelihood_pos_jac = np.zeros(n)
        log_likelihood_neg_jac = np.zeros(n)

        for edge in edge_set:
            edge_index = np.array(edge)
            thetas_sum = thetas[edge_index].sum()
            log_likelihood_pos_jac[edge_index] += (1 - sigmoid(thetas_sum))

        neg_edge_set = set([])

        for _ in range(negative_samples):
            while True:
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    break
            neg_edge_index = np.array(neg_edge)
            thetas_sum = thetas[neg_edge_index].sum()
            log_likelihood_neg_jac[neg_edge_index] += sigmoid(thetas_sum)
            neg_edge_set.add(neg_edge)

        return log_likelihood_pos_jac + (n / negative_samples) * log_likelihood_neg_jac

    def fit(self, G, negative_samples):
        n = len(G)
        edge_set = G.to_index(set)

        res = minimize(lambda x: - LogisticCP.graph_log_likelihood(edge_set, n, self.order, x, negative_samples), self.thetas, jac=lambda x: - LogisticCP.graph_log_likelihood_jacobian(edge_set, n, self.order, x, negative_samples))
        
        self.thetas = res.x

        return -res.fun, res.x

if __name__ == '__main__':

    G = load_world_trade()
    # G, _ = load_coauth_mag_kdd(simplex_min_size=3, simplex_max_size=3, year_min=1995, year_max=2000)
    G = G.deduplicate()
    G = Hypergraph.convert_node_labels_to_integers(G)

    model = LogisticCP(order=3, thetas=np.zeros(len(G)))

    print(model.fit(G, 50))
    
    plt.figure(figsize=(10, 10))
    plt.plot(- np.sort(- model.thetas))
    plt.savefig('thetas_temp.png')
    
    print(model.thetas)


