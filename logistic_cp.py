from base import *
from hypergraph import *
from utils import *
from dataloader import * 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticCP:

    def __init__(self, thetas, order_min=2, order_max=2, alpha_theta=0):
        self.thetas = - np.sort(-thetas)
        assert(order_min <= order_max)
        self.order_min = order_min
        self.order_max = order_max
        self.alpha_theta = alpha_theta

    def sample(self, n):
        G = Hypergraph()

        for order in range(self.order_min, self.order_max + 1):
            for edge in itertools.combinations(range(n), order):
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
    def graph_log_likelihood(edge_set, n, M_neg, order_min, order_max, thetas, negative_samples):

        log_likelihood_pos = 0
        log_likelihood_neg = 0

        for edge in edge_set:
            edge_index = np.array(edge)
            log_likelihood_pos += np.log(sigmoid(thetas[edge_index].sum()))

        neg_edge_set = set([])
        
        for _ in range(negative_samples):
            while True:
                order = np.random.choice(np.arange(order_min, order_max + 1), p=M_neg/M_neg.sum())
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    neg_edge_index = np.array(neg_edge)
                    log_likelihood_neg += np.log(1 - sigmoid(thetas[neg_edge_index].sum()))
                    neg_edge_set.add(neg_edge)
                    break

        return log_likelihood_pos + (M_neg.sum() / negative_samples) * log_likelihood_neg

    @staticmethod
    def graph_log_likelihood_jacobian(edge_set, n, M_neg, order_min, order_max, thetas, negative_samples):

        log_likelihood_pos_jac = np.zeros(n)
        log_likelihood_neg_jac = np.zeros(n)

        for edge in edge_set:
            edge_index = np.array(edge)
            thetas_sum = thetas[edge_index].sum()
            log_likelihood_pos_jac[edge_index] += (1 - sigmoid(thetas_sum))

        neg_edge_set = set([])

        for _ in range(negative_samples):
            while True:
                order = np.random.choice(np.arange(order_min, order_max + 1), p=M_neg/M_neg.sum())
                neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                    neg_edge_index = np.array(neg_edge)
                    thetas_sum = thetas[neg_edge_index].sum()
                    log_likelihood_neg_jac[neg_edge_index] += sigmoid(thetas_sum)
                    neg_edge_set.add(neg_edge)
                    break

        return log_likelihood_pos_jac + (M_neg.sum() / negative_samples) * log_likelihood_neg_jac

    def fit(self, G, negative_samples):
        n = len(G)
        edge_set = G.to_index(set)
        M = G.num_simplices(separate=True)
        binomial_coeffs = binomial_coefficients(n, self.order_max)
        M_neg = binomial_coeffs[n, self.order_min:(1 + self.order_max)] - M

        res = minimize(lambda x: - LogisticCP.graph_log_likelihood(edge_set, n, M_neg, self.order_min, self.order_max, x, negative_samples), self.thetas, jac=lambda x: - LogisticCP.graph_log_likelihood_jacobian(edge_set, n, M_neg, self.order_min, self.order_max, x, negative_samples), method='BFGS')
        
        self.thetas = res.x
        lp = - res.fun

        return lp, self.thetas

    def fit_torch(self, G, features, gold_ranks, negative_samples=-1, num_epochs=10, max_patience=5, ranks_col=0, early_stopping='log-posterior', lr=1e-6, learnable_ranks=True):
        # Graph 
        n = len(G)
        edge_set = G.to_index(set)
        M = G.num_simplices(separate=True)
        binomial_coeffs = binomial_coefficients(n, self.order_max)
        M_neg = binomial_coeffs[n, self.order_min:(1 + self.order_max)] - M
        
        # Features and model
        feature_dim = features.shape[-1]

        features = torch.from_numpy(features.astype(np.float32)).cuda()

        if learnable_ranks:
            thetas_model = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1)).cuda()
            optimizer = torch.optim.SGD(thetas_model.parameters(), lr=lr) 
        else:
            thetas = nn.Parameter(data=torch.rand(n, 1), requires_grad=True)
            optimizer = torch.optim.SGD(thetas, lr=lr)

        torch_sigmoid = nn.Sigmoid()

        # Training loop
        pbar = tqdm(range(num_epochs))
        log_posteriors = []
        spearmans = []
        max_log_posterior = - sys.maxsize
        max_spearman = - sys.maxsize
        opt_model = None
        patience = 0

        for i in range(num_epochs):
            loss = torch.tensor(0.0).cuda()
            
            if learnable_ranks:
                thetas = thetas_model(features)

            log_likelihood_pos = torch.tensor(0.0).cuda()
            log_likelihood_neg = torch.tensor(0.0).cuda()

            for edge in edge_set:
                edge_index = torch.tensor(edge).long().cuda()
                log_likelihood_pos += torch.log(torch_sigmoid(torch.sum(thetas[edge_index])))

            neg_edge_set = set([])
            
            for _ in range(negative_samples):
                while True:
                    order = np.random.choice(np.arange(self.order_min, self.order_max + 1), p=M_neg/M_neg.sum())
                    neg_edge = tuple(sorted(sample_combination(n=n, k=order)))
                    if neg_edge not in edge_set and neg_edge not in neg_edge_set:
                        neg_edge_index = torch.tensor(neg_edge).long().cuda()
                        log_likelihood_neg += torch.log(1 - torch_sigmoid(torch.sum(thetas[neg_edge_index])))
                        neg_edge_set.add(neg_edge)
                        break

                loss = - log_likelihood_pos - (M_neg.sum() / negative_samples) * log_likelihood_neg

            loss += self.alpha_theta / 2 * torch.sum(thetas**2)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            log_posteriors.append(-loss.item())
            spearmans.append(scipy.stats.spearmanr(gold_ranks, thetas.clone().detach().cpu().numpy()).correlation)
            pbar.set_description('Loss: {}, Spearman: {}'.format(log_posteriors[-1], spearmans[-1]))
            pbar.update()

            # Early stopping
            if not np.isnan(log_posteriors[-1]) and max_log_posterior <= log_posteriors[-1] and early_stopping == 'log-posterior':
                max_log_posterior = log_posteriors[-1]
                max_spearman = spearmans[-1]
                opt_model = copy.deepcopy(thetas_model)
            if not np.isnan(spearmans[-1]) and max_spearman <= spearmans[-1] and early_stopping == 'spearman':
                max_log_posterior = log_posteriors[-1]
                max_spearman = spearmans[-1]
                opt_model = copy.deepcopy(thetas_model)

            if i > 1:
                if (log_posteriors[-1] < log_posteriors[-2] and early_stopping == 'log-posterior') or (spearmans[-1] > spearmans[-2] and early_stopping == 'spearman'):
                    patience += 1
                else:
                    patience = 0

                if patience == max_patience:
                    break
                
        pbar.close()

        thetas = opt_model(features)
        self.thetas = thetas.detach().cpu().numpy()

        return max_log_posterior, self.thetas
