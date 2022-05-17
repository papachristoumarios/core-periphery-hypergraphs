from base import *
from utils import *
from dataloader import *
from hypergraph import *

class HyperNSM:

    def __init__(self, alpha=10, order_min=2, order_max=2, xi = lambda k: 1 / k):
        self.alpha = alpha
        self.order_min = order_min
        self.order_max = order_max
        self.xi = xi

    def sample(self, n):
        G = Hypergraph()
        ranks = n - (1 + np.arange(n))

        for order in range(self.order_min, self.order_max + 1):
            for edge in itertools.combinations(range(n), order):
                edge = np.array(edge)
                p_edge = sigmoid(self.xi(order) * generalized_mean(ranks[edge], self.alpha))
                if np.random.uniform() <= p_edge:
                    G.add_simplex_from_nodes(nodes=edge.tolist(), simplex_data = {})
        return G, ranks
    
    @staticmethod
    def graph_log_likelihood(edge_set, n, M_neg, order_min, order_max, ranks, alpha, xi, negative_samples):

        log_likelihood_pos = 0
        log_likelihood_neg = 0
        
        for edge in edge_set:
            edge_index = np.array(edge)
            log_likelihood_pos += np.log(sigmoid(xi(len(edge_index)) * generalized_mean(ranks[edge_index], alpha) / n))

        neg_edge_set = set([])
        
        for _ in range(negative_samples):
            while True:
                order = np.random.choice(np.arange(order_min, order_max + 1), p=M_neg/M_neg.sum())
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    neg_edge_index = np.array(neg_edge)
                    log_likelihood_neg += np.log(1 - sigmoid(xi(order) * generalized_mean(ranks[neg_edge_index], alpha) / n))
                    neg_edge_set.add(neg_edge)
                    break
        return log_likelihood_pos + (M_neg.sum() / negative_samples) * log_likelihood_neg

    def display_p(self):
        if np.isinf(self.alpha):
            return '\\infty'
        elif np.isinf(-self.alpha):
            return '- \\infty'
        else:
            return self.alpha

    def plot_sample(self, n):
        G, ranks = self.sample(n)
        G = G.clique_decomposition()
        A = nx.to_numpy_array(G)
        plt.figure(figsize=(10, 10))
        plt.imshow(A)
        plt.title('Adjacency Matrix for G ~ logstic-TH($p={}$) (Clique-decomposition)'.format(self.display_p()))
        plt.xlabel('Ranked Nodes by $\\pi(u)$')
        plt.ylabel('Ranked Nodes by $\\pi(u)$')
        plt.figure(figsize=(10, 10))
        log_rank = np.log(1 + np.arange(A.shape[0]))
        log_degree = np.log(1 + A.sum(0))
        plt.plot(log_rank, log_degree)
        plt.title('Degree Plot')
        plt.xlabel('Node Rank by $\\pi(u)$ (log)')
        plt.ylabel('Node Degree (log)')
        plt.legend()

        plt.figure(figsize=(10, 10))
        degree_ranks = np.argsort(-log_degree)
        for i in range(A.shape[0]):
            A[i, :] = A[i, degree_ranks]

        for i in range(A.shape[1]):
            A[:, i] = A[degree_ranks, i]

        log_degree = log_degree[degree_ranks]
        plt.imshow(A)
        plt.title('Adjacency Matrix for G ~ logstic-TH($p={}$) (Clique-decomposition)'.format(self.display_p()))
        plt.xlabel('Ranked Nodes by degree')
        plt.ylabel('Ranked Nodes by degree')
        plt.figure(figsize=(10, 10))

        plt.plot(log_rank, log_degree)
        p = np.polyfit(log_rank, log_degree, deg=1)
        r2 = np.corrcoef(log_rank, log_degree)[0, 1]
        plt.plot(log_rank, log_degree, linewidth=1, label='Realized Degree $R^2 = {}$'.format(round(r2, 2)))
        plt.plot(log_rank, p[1] + p[0] * log_rank, linewidth=2, label='Linear Regression')
        plt.title('Degree Plot')
        plt.xlabel('Node Rank by degree (log)')
        plt.ylabel('Node Degree (log)')
        plt.legend()

    def fit(self, G, p, negative_samples, max_iters=10000, eval_log_likelihood=True): 
        n = len(G)
        assert(p > max(1, self.alpha))
        q = p / (p - 1)
        x = np.random.uniform(size=n)

        edges = G.to_index()
        M = G.num_simplices(separate=True)

        def fixed_point(x, edges,  M, order_min, order_max, alpha, xi):
            z = np.zeros((edges.shape[0], edges.shape[1]))
            w = np.zeros_like(x)

            for i, k in enumerate(range(order_min, order_max + 1)):
                for m in range(M[i]):
                    for j in edges[i, m, :k]:
                        z[i, m] += x[j]**alpha

            z = z**(1 / alpha - 1)

            for i, k in enumerate(range(order_min, order_max + 1)):
                for m in range(M[i]):
                    for j in edges[i, m, :k]:
                        w[j] += xi(k) * z[i, m]

            y = (x**(alpha - 1)) * w                

            return y

        x_prev = x

        pbar = tqdm(range(max_iters))
        for _ in range(max_iters):
            y = fixed_point(x, edges, M, self.order_min, self.order_max, self.alpha, self.xi)
            x = (y / np.linalg.norm(y, q))**(1 / (p - 1))
            pbar.set_description('Error: {}'.format(np.linalg.norm(x - x_prev) / (1e-5 + np.linalg.norm(x_prev))))
            pbar.update()
            if np.allclose(x, x_prev, rtol=1e-3):
                break
            else:
                x_prev = x
        pbar.close()
        ranks = np.argsort(-x)
        self.ranks = ranks
        x = normalize(x)
        ll = LogisticTH.graph_log_likelihood(G.to_index(set), len(G), G.num_simplices(separate=True, negate=True), self.order_min, self.order_max, ranks, 10, self.xi, negative_samples=negative_samples)
        return ll, x, ranks

