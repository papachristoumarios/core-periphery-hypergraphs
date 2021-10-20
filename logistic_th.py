from base import *
from utils import *
from dataloader import *
from hypergraph import *

class LogisticTH:

    def __init__(self, alpha=10, order=2):
        self.alpha = alpha
        self.order = order

    def sample(self, n):
        G = Hypergraph()
        ranks = n - (1 + np.arange(n))

        for edge in itertools.combinations(range(n), self.order):
            edge = np.array(edge)
            p_edge = sigmoid(generalized_mean(ranks[edge], self.alpha) / n)
            if np.random.unifomr() <= p_edge:
                G.add_simplex_from_nodes(nodes=edge.tolist(), simplex_data = {})

        return G, ranks
    
    @staticmethod
    def graph_log_likelihood(edge_set, n, order, ranks, alpha, negative_samples):

        log_likelihood_pos = 0
        log_likelihood_neg = 0

        for edge in edge_set:
            edge_index = np.array(edge)
            log_likelihood_pos += np.log(sigmoid(generalized_mean(ranks[edge_index], alpha) / n))

        neg_edge_set = set([])
        
        for _ in range(negative_samples):
            while True:
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    break
            neg_edge_index = np.array(neg_edge)
            log_likelihood_neg += np.log(1 - sigmoid(generalized_mean(ranks[edge_index], alpha) / n))
            neg_edge_set.add(neg_edge)

        return log_likelihood_pos + (n / negative_samples) * log_likelihood_neg

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

    def fit(self, G, p, negative_samples): 
        n = len(G)
        assert(p > max(1, self.alpha))
        q = p / (p - 1)
        x = np.random.uniform(size=n)
        order_factorial = scipy.special.factorial(self.order)
        edges = G.to_index()

        def fixed_point(x, edges, order_factorial):
            y = np.zeros_like(x) 

            for m in range(edges.shape[0]):
                generalized_mn = generalized_mean(x[edges[m]], self.alpha)
                y[edges] += order_factorial / generalized_mn**(self.alpha - 1)

            y *= np.abs(x)**(self.alpha - 2) * x

            return y

        x_prev = x

        while True:
            y = fixed_point(x, edges, order_factorial)
            x = np.linalg.norm(y, q)**(1 - q) * np.abs(y)**(q - 2) * y
            if np.allclose(x, x_prev, rtol=1e-3):
                break
            else:
                x_prev = x
            

        ranks = np.argsort(-x)

        x = normalize(x)

        ll = LogisticTH.graph_log_likelihood(G.to_index(set), len(G), self.order, ranks, self.alpha, negative_samples)

        return ll, x, ranks

if __name__ == '__main__':
    # G, labels = load_hypergraph(name='contact-primary-school', simplex_min_size=3, simplex_max_size=3, timestamp_min=31220, timestamp_max=31220 + 500)
    G = load_world_trade()
    G = Hypergraph.convert_node_labels_to_integers(G)
    G = G.deduplicate()
    
    logistic_th = LogisticTH(order=2)

    x, ranks = logistic_th.fit(G, p = 20)
    print(x)
    print(LogisticTH.graph_log_likelihood(G.to_index(set), len(G), 3, ranks, 10, len(G) // 2))
    plt.figure(figsize=(10, 10))

    plt.plot(x[ranks])
    
    plt.savefig('th_ranks_test.png')
