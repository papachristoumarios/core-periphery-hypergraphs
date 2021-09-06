from base import *
from utils import *

class CIGAM:

    def __init__(self, c=[1.5], b=3, H=[4], order=2):

        if isinstance(c, float):
            self.c = np.array([c])
        else:
            self.c = np.array(c)

        if isinstance(H, float):
            self.H = np.array([H])
        else:
            self.H = np.array(H)

        assert(np.all(self.c == np.sort(self.c)))
        assert(c[0] >= 1 and c[-1] <= b)
        assert(np.all(H == np.sort(H)))
        assert(len(self.H) == len(self.c))
        assert(b > 1)
        assert(order >= 2)

        # Hypergraph Order
        self.order = order

        # Ranks parameter
        self.lambda_ = np.log(b)
        
        # Number of layers for the multi-layer model
        self.num_layers = len(self.c)

        self.stan_definitions = {
                'N' : 'int N;',
                'M' : 'int M;',
                'K' : 'int K;',
                'L' : 'int L;',
                'H' : 'real H[L];',
                'b' : 'real<lower=1> b;',
                'lambda' : 'real<lower=0> lambda;',
                'c' : 'real<lower=1> c[L];',
                'edges' : 'int edges[M, K];',
                'binomial_coefficients' : 'int binomial_coefficients[N + 1, K + 1];',
                'ranks' : 'real<lower=0, upper=H[L]> ranks[N];'
        }

    @staticmethod
    def find_c(h, H, c):
        h_max = np.max(h)
        idx = np.where(h_max <= H)[0][0]
        return c[idx]

    @property
    def b(self):
        return np.exp(self.lamdba_)

    @b.getter
    def b(self):
        return np.exp(self.lambda_)

    @b.setter
    def b(self, b_):
        self.lambda_ = np.log(b_)
        return b_

    def bias_matrix(self, N, return_ranks=True):
        P = np.zeros(tuple(self.order * [N]))
        h = self.continuous_tree_sample(N=N)
        h = np.sort(h)

        for edge in itertools.combinations(range(N), self.order):
            edge = np.array(edge)
            bias = CIGAM.find_c(h[edge], self.H, self.c)**(-1-h[edge].min())
           
            for dedge in itertools.permutations(edge, self.order):
                P[dedge] = bias
        
        if return_ranks:
            return P, self.H[-1] - h
        else:
            return P, h


    def sample(self, N, return_ranks=True, method='ball_dropping'):
        assert(method in ['ball_dropping', 'naive'])
        heights = self.continuous_tree_sample(N=N)
        ordering  = np.argsort(heights)
        heights = heights[ordering]
        ranks = self.H[-1] - heights
        G = Hypergraph()

        layers, num_layers = CIGAM.get_layers(ranks, self.H)

        if method == 'naive':
            for edge in itertools.combinations(range(N), self.order):
                edge = np.array(edge)
                heights_min = heights[edge].min()
                heights_argmax = edge[heights[edge].argmax()]
                p_edge = self.c[layers[heights_argmax]]**(-1-heights_min)
                if np.random.uniform() <= p_edge:
                    G.add_simplex_from_nodes(nodes=edge.tolist(), simplex_data = {})
        else:
            for i in range(heights.shape[0] - self.order + 1):
                j = i + 1
                for l in range(num_layers.shape[-1]):
                    if num_layers[i, l] >= 1 and j + num_layers[i, l] - i >= self.order - 1:
                        batch = ball_dropping_helper(S=[ordering[i]], V=ordering[i+1:j+num_layers[i, l]], k_f=1, n_f=num_layers[i, l], p=self.c[l]**(-1-heights[i]), k=self.order, directed=False)       
                        for edge in batch:
                            G.add_simplex_from_nodes(nodes=edge, simplex_data = {})
                    
                    j += num_layers[i, l]

        if return_ranks:
            return G, ranks
        else:
            return G, heights

    def plot_sample(self, n):
        H, h = self.sample(n)
        G = H.clique_decomposition()
        A = nx.to_numpy_array(G)
        plt.figure(figsize=(10, 10))
        plt.imshow(A)
        plt.title('Adjacency Matrix for G ~ CIGAM($c$={}, $b$={}, $H$={})'.format(self.c, self.b, self.H))
        plt.xlabel('Ranked Nodes by $h(u)$')
        plt.ylabel('Ranked Nodes by $h(u)$')
        plt.savefig('adjacency_matrix.png')
        plt.figure(figsize=(10, 10))
        log_rank = np.log(1 + np.arange(A.shape[0]))
        log_degree = np.log(1 + A.sum(0))
        log_degree = -np.sort(-log_degree)
        p = np.polyfit(log_rank, log_degree, deg=1)
        alpha_lasso = 0.1
        clf_lasso = linear_model.Lasso(alpha=alpha_lasso)
        clf_lasso.fit(log_rank.reshape(-1, 1), log_degree)
        r2 = np.corrcoef(log_rank, log_degree)[0, 1]
        plt.plot(log_rank, log_degree, linewidth=1, label='Realized Degree $R^2 = {}$'.format(round(r2, 2)))
        plt.plot(log_rank, p[1] + p[0] * log_rank, linewidth=2, label='Linear Regression')
        plt.plot(log_rank, clf_lasso.intercept_ + clf_lasso.coef_ * log_rank, linewidth=2, label='Lasso Regression ($a = {}$)'.format(alpha_lasso))
        plt.xlabel('Node Rank by $h(u)$ (log)')
        plt.ylabel('Node Degree (log)')
        plt.title('Degree Plot')
        plt.legend()
        plt.savefig('degree_plot.eps')

    def stan_model(self, known, dump=True, load=True):
            
        with open('cigam_functions.stan') as f:
            functions_segment = f.read()

        with open('cigam_model.stan') as f:
            model_segment = f.read()

        data = []
        params = []
        data_keys = []
        params_keys = []

        for key, val in known.items():
            if val:
                data.append(self.stan_definitions[key])
                data_keys.append(key)
            else:
                params.append(self.stan_definitions[key])
                params_keys.append(key)

        data_text = '\n\t'.join(data)
        params_text = '\n\t'.join(params)

        data_segment = 'data {\n\t' + data_text + '\n}'
        params_segment = 'parameters {\n\t' + params_text + '\n}'

        model_code = '{}\n\n{}\n\n{}\n\n{}'.format(functions_segment, data_segment, params_segment, model_segment)

        model_name = '{}_given_{}'.format('_'.join(params_keys), '_'.join(data_keys))


        if load:
            if os.path.isfile('{}.pickle'.format(model_name)):
                with open('{}.pickle'.format(model_name), 'rb') as f:
                    stan_model = pickle.load(f)
                return stan_model
            else:
                stan_model = pystan.StanModel(model_code=model_code, model_name=model_name)
        else:
            stan_model = pystan.StanModel(model_code=model_code, model_name=model_name)

        if dump:
            with open('{}.pickle'.format(model_name), 'wb+') as f:
                pickle.dump(stan_model, f, protocol=-1)
            with open('{}.stan'.format(model_name), 'w+') as f:
                f.write(model_code)

        return stan_model

    def continuous_tree_sample(self, N):
        u = np.random.uniform(size=N)
        y = np.log(u * (self.b**self.H[-1] - 1) + 1) / np.log(self.b)
        return y

    def stan_model_sample(self, known, model_data, dump=True, load=True):
        stan_model = self.stan_model(known, dump=dump, load=load)
        fit = stan_model.sampling(data=model_data, iter=100, chains=10, n_jobs=10)

        return fit

    def params_posterior(self):
        known = {
                'N' : True,
                'L' : True,
                'K' : True,
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : True,
                'lambda' : False,
                'c' : False
        }

        return known

    def latent_posterior(self):
        known = {
                'N' : True,
                'L' : True,
                'K' : True, 
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : False,
                'lambda' : True,
                'c' : True
        }

        return known

    def params_latent_posterior(self):
        known = {
                'N' : True,
                'L' : True,
                'K' : True,
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : False,
                'lambda' : False,
                'c' : False
        }

        return known

    def visualize_posterior(self, fit, params=None, pairplot=False):

        if params is None:
            params = list(fit.extract().keys())

        df = stanfit_to_dataframe(fit, params)

        params = [col for col in df.columns]


        if pairplot:
            sns.pairplot(df, x_vars=params, y_vars=params, kind='kde')

        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            colors = iter(cm.rainbow(np.linspace(0, 1, len(params))))

            for param in params:
                if param == 'lp__':
                    continue
                c = next(colors)
                param_mean = round(df[param].mean(), 2)
                param_std = round(df[param].std(), 2)
                sns.histplot(df[param], kde=True, label='{} (mean = {}, std = {})'.format(param, param_mean, param_std), ax=ax, color=c)

            plt.xlabel('Parameters')
            plt.ylabel('Posterior')
            plt.legend()

    def visualize_degree_plot(self, G, fit):

        degrees_init = 1 + np.array([G.degree(u) for u in range(len(G))])

        ranks = fit.extract()['ranks']
        degrees = np.zeros_like(ranks)

        A = nx.to_numpy_array(G)
        A_mean = np.zeros_like(A)

        for i in range(ranks.shape[0]):

            positions = np.argsort(-ranks[i, :])
            degrees[i, :] = degrees_init[positions]

            for i in range(A.shape[0]):
                A[i, :] = A[i, positions]

            for i in range(A.shape[1]):
                A[:, i] = A[positions, i]

            A_mean += A

        degrees_mean = degrees.mean(0)
        degrees_std = degrees.std(0)
        x_axis = np.arange(1, len(degrees_mean) + 1)
        log_x_axis = np.log(x_axis)

        plt.figure(figsize=(10, 10))

        plt.plot(log_x_axis, np.log(degrees_mean), 'k-', color='blue', label='Empirical Mean Degrees wrt to Rankings')
        plt.fill_between(log_x_axis, np.log(degrees_mean-degrees_std), np.log(degrees_mean+degrees_std), color='blue', alpha=0.3)

        degrees_init = -np.sort(-degrees_init)

        plt.plot(log_x_axis, np.log(degrees_init), marker=11, color='red', linewidth=0, label='Actual Degrees')

        plt.title('Degree Plot')
        plt.xlabel('Ranking')
        plt.ylabel('Degree')
        plt.legend()

        # plt.figure(figsize=(10, 10))
        # plt.imshow(A_mean)
        # plt.xlabel('Ranking')
        # plt.ylabel('Ranking')
        # plt.title('Expected Adjacency Matrix wrt to Rankings')

    def fit_model_bayesian_em(self, G, H, lambda_init=np.log(3.2), c_init=1.4, epochs=30):

        lambda_old = lambda_init
        b_old = np.exp(lambda_old)
        c_old = c_init
        N = len(G)
        edges = G.to_index()
        M = edges.shape[0]

        optimum = (-np.inf, (lambda_old, b_old, c_old))

        binomial_coefs = binomial_coefficients(N, self.order)

        for _ in range(epochs):
            # E-Step: Sample from p(heights | G, b_old, c_old)
            e_data = {
                    'N' : N,
                    'L' : len(H),
                    'K' : self.order,
                    'M' : M, 
                    'H' : H,
                    'edges' : edges,
                    'binomial_coefficients' : binomial_coeffs,
                    'lambda' : lambda_old,
                    'c' : c_old
            }

            e_fit = self.stan_model_sample(self.latent_posterior(), e_data)

            # For every node let its rank be the mean of the corresponding sampled parameter heights[i]
            e_ranks = e_fit.extract()['ranks'].mean(0)

            # (Bayesian) M-Step: Sample from p(b, c | G, heights_avg)
            m_data = {
                    'N' : N,
                    'L' : len(H),
                    'K' : self.order,
                    'M' : M,
                    'H' : H,
                    'edges' : edges,
                    'binomial_coefficients' : binomial_coeffs,
                    'ranks' : e_ranks
            }

            # Let the parameters of the next iteration be the mean of the predicted parameters
            m_fit = self.stan_model_sample(self.params_posterior(), m_data)
            self.lambda_ = m_fit.extract()['lambda'].mean()
            self.b = np.exp(m_fit.extract()['lambda']).mean()
            self.c = m_fit.extract()['c'].mean(0)
            q_function = CIGAM.q_function(G, e_fit.extract()['ranks'], H, self.b, self.c, self.order)

            print('lambda = {}, c = {}, b = {}, Q = {}'.format(self.lambda_, self.c, self.b, q_function))

            if q_function >= optimum[0]:
                optimum = (q_function, (self.lambda_, self.b, self.c))

            lambda_old, c_old, b_old = self.lambda_, self.c, self.b

        self.lambda_, self.b, self.c = optimum[-1]
        print('Best fit', optimum)

        return optimum

    def fit_model_em(self, G, H, lambda_init=np.log(3.2), c_init=[1.4], epochs=15):
        lambda_old = lambda_init
        c_old = np.array(c_init)
        N = len(G)
        edges = G.to_index()
        M = edges.shape[0]

        binomial_coefs = binomial_coefficients(N, self.order)
        
        optimum = (-np.inf, (lambda_old, c_old))

        bounds = ((1, np.inf), (1, np.inf))
        q_values = []

        for _ in range(epochs):
            # E-Step: Sample from p(ranks | G, lambda_old, c_old)
            e_data = {
                    'N' : N,
                    'L' : len(H),
                    'K' : self.order,
                    'M' : M,
                    'H' : H,
                    'edges' : edges,
                    'binomial_coefficients' : binomial_coeffs,
                    'lambda' : lambda_old,
                    'c' : c_old
            }

            e_fit = self.stan_model_sample(self.latent_posterior(), e_data)
            ranks_post = e_fit.extract()['ranks']
            sizes_post, neg_sizes_post, layers_post, num_layers_post = CIGAM.get_partition_sizes(G, ranks_post, self.order, self.H)
            sum_ranks_post = ranks_post.sum(1)

            # M-Step: Optimize the Q function = E_{posterior latent variables} [complete_log_likelihood]
            res = minimize(lambda x: - CIGAM.q_function(G, ranks_post, H, x[0], x[1:], self.order, sizes_post=sizes_post, neg_sizes_post=neg_sizes_post, sum_ranks=sum_ranks_post), np.hstack(([lambda_old], c_old)), method='L-BFGS-B', bounds=bounds) 

            self.lambda_ = res.x[0]
            self.c = res.x[1:]

            q_function = -res.fun
            q_values.append(q_function)

            print('lambda = {}, c = {}, b = {}, Q = {}'.format(self.lambda_, self.c, self.b, q_function))

            if q_function >= optimum[0]:
                optimum = (q_function, (self.lambda_, self.c))

            if np.allclose(self.c, c_old) and np.allclose(self.lambda_, lambda_old):
                break

            lambda_old, c_old = self.lambda_, self.c

        self.lambda_, self.c = optimum[-1]
        print('Best fit', optimum)

        plt.figure(figsize=(10, 10))
        plt.plot(q_values)

        return optimum

    def fit_model_bayesian(self, G, H):
        edges = G.to_index()
        N = len(G)
        M = edges.shape[0]
        binomial_coeffs = binomial_coefficients(N, self.order) 
        
        data = {
                'N' : len(G),
                'K' : self.order,
                'L' : len(H),
                'M' : M,
                'H' : H,
                'edges' : edges,
                'binomial_coefficients' : binomial_coeffs
        }

        fit = self.stan_model_sample(self.params_latent_posterior(), data)

        self.lambda_ = fit.extract()['lambda'].mean()
        self.b = np.exp(fit.extract()['lambda']).mean()
        self.c = fit.extract()['c'].mean(0)
        self.H = H

        return fit

    def backup_params(self):
        self.b_backup = self.b
        self.c_backup = self.c
        self.lambda_backup = self.lambda_
        self.H_backup = self.H

    def restore_params(self):
        self.b = self.b_backup
        self.c = self.c_backup
        self.lambda_ = self.lambda_backup
        self.H = self.H_backup

    @staticmethod
    def graph_log_likelihood(G, ranks, H, c, order, sizes=None, neg_sizes=None, layers=None, num_layers=None):
    
        if sizes is None or  neg_sizes is None or layers is None or num_layers is None:
            sizes, neg_sizes, layers, num_layers = CIGAM.get_partition_sizes(G, ranks, order, H)

        result = np.sum([sizes[i, l] * np.log(c[l]) * (-1-H[-1]+ranks[i]) + neg_sizes[i, l] * np.log(1 - c[l]**(-1-H[-1]+ranks[i])) for i in range(len(ranks)) for l in range(len(H))])

        return result

    @staticmethod
    @jit(forceobj=True)
    def graph_log_likelihood_jacobian(G, ranks, H, c, order, sizes=None, neg_sizes=None, layers=None, num_layers=None):
        if sizes is None or neg_sizes is None or layers is None or num_layers is None:
            sizes, neg_sizes, layers, num_layers = CIGAM.get_partition_sizes(G, ranks, order, H)
        return np.array([np.sum([
            sizes[u, l] * (-1 - H[-1] + ranks[u]) / (c[l]) -
            #neg_sizes[u] * (-1 - H[-1] + ranks[u]) * c[-1]**(-2 + ranks[u] - H[-1]) / (1 - c[-1]**(-1-H[-1]+ranks[u])) 
            neg_sizes[u, l] / (c[l]) * (-1-H[-1] + ranks[u]) * (c[l])**(-2-H[-1] + ranks[u]) 
            for u in G.nodes()]) for l in range(len(H))])
        
    @staticmethod
    def complete_log_likelihood(G, ranks, H, lambda_, c, order, sizes=None, neg_sizes=None, layers=None, num_layers=None, sum_ranks=None):
        return CIGAM.ranks_log_likelihood(None if sum_ranks is not None else ranks, len(G), lambda_, H, sum_ranks) + CIGAM.graph_log_likelihood(G, ranks, H, c, order, sizes=sizes, neg_sizes=neg_sizes, layers=layers, num_layers=num_layers)

    @staticmethod
    @jit(forceobj=True)
    def q_function(G, ranks_post, H, lambda_, c, order, sizes_post=None, neg_sizes_post=None, sum_ranks=None):
        if sizes_post is None or neg_sizes_post is None:
            sizes_post, neg_sizes_post, layers_post, num_layers_post = CIGAM.get_partition_sizes(G, ranks_post, order, H)
        if sum_ranks is None:
            sum_ranks = ranks_post.sum(1)

        return np.mean([CIGAM.complete_log_likelihood(G, ranks_post[i, :], H, lambda_, c, order, sizes=sizes_post[i, :], neg_sizes=neg_sizes_post[i, :], sum_ranks=sum_ranks[i]) for i in range(ranks_post.shape[0])])

    @staticmethod
    def complete_log_likelihood_jacobian(G, ranks, H, lambda_, c, order, sizes=None, neg_sizes=None, sum_ranks=None, layers=None, num_layers=None):
        return np.hstack(([CIGAM.ranks_log_likelihood_jacobian(None if sum_ranks is not None else ranks, len(G), lambda_, H, sum_ranks)], CIGAM.graph_log_likelihood_jacobian(G, ranks, H, c, order, sizes=sizes, neg_sizes=neg_sizes, layers=layers, num_layers=num_layers)))

    @staticmethod
    def ranks_log_likelihood(ranks, n, lambda_, H, sum_ranks=None, num_terms=2):
        if sum_ranks is None:
            sum_ranks = np.nansum(ranks)
        if num_terms > 0:
            # Use Taylor expansion of log(1 - x)
            Z = np.sum(np.exp(-np.arange(1, 1 + num_terms).astype(np.float64) * lambda_ * H[-1]))
        else:
            Z = np.log(1 - np.exp(-lambda_ * H[-1]))
        result = n * np.log(lambda_) - lambda_ * sum_ranks - Z
        return result
    
    @staticmethod
    @jit(nopython=True)
    def ranks_log_likelihood_jacobian(ranks, n, lambda_, H, sum_ranks=None):
        if sum_ranks is None:
            sum_ranks = np.nansum(ranks)
        return n / lambda_ - sum_ranks - n * H[-1]  / (np.exp(H[-1] * lambda_) - 1)

    @staticmethod
    @jit(nopython=True)
    def get_layers(ranks, H):
        layers = np.zeros(shape=ranks.shape, dtype=np.int64)
        num_layers = np.zeros(shape=ranks.shape + (len(H),), dtype=np.int64)
    
        j = 0
   
        for i in range(ranks.shape[0]):
            if H[-1] - ranks[i] > H[j]:
                j += 1
            layers[i] = j 

        j = 0

        temp = np.zeros(shape=len(H), dtype=np.int64)

        for l in range(len(H)):
            temp[l] = len(np.where(layers == l)[0])

        for i in range(ranks.shape[0] - 1):
            while temp[j] == 0 and j < len(H):
                j += 1
            temp[j] -= 1

            num_layers[i] = np.copy(temp)

        return layers, num_layers

    @staticmethod
    def get_partition_sizes(G, ranks, order, H):
        sizes = np.zeros(shape=ranks.shape + (len(H),))
        neg_sizes = np.zeros(shape=sizes.shape, dtype=np.int64)
        binomial_coeffs = binomial_coefficients(len(G), order - 1)
        layers = np.zeros(shape=ranks.shape, dtype=np.int64)
        num_layers = np.zeros(shape=ranks.shape + (len(H),), dtype=np.int64)

        if len(ranks.shape) == 1:
            layers, num_layers = CIGAM.get_layers(ranks, H)
            if G is not None:
                for edge in G.edges():
                    edge_index = edge.to_index()
                    argmax = edge_index[np.argmax(ranks[edge_index])]
                    argmin = layers[edge_index[np.argmin(ranks[edge_index])]]
                    sizes[argmax, argmin] += 1            

            for i in range(num_layers.shape[0] - 1):
                j = i + 1
                for l in range(num_layers.shape[1]):
                    neg_sizes[i, l] = binomial_coeffs[j + num_layers[i, l] - i, order - 1] - binomial_coeffs[j - i - 1, order - 1]  - sizes[i, l]
                    j += num_layers[i, l]
                    
        else:
            for i in range(ranks.shape[0]):
                sizes_temp, neg_sizes_temp, layers_temp, num_layers_temp = CIGAM.get_partition_sizes(G, ranks[i], order, H)
                sizes[i] = sizes_temp
                neg_sizes[i] = neg_sizes_temp
                layers[i] = layers_temp
                num_layers[i] = num_layers_temp
            
        return sizes, neg_sizes, layers, num_layers


    def fit_model_given_ranks(self, G, ranks, num_layers=1):
        assert(num_layers >= 1)
        self.c = np.array([1.5])
        self.H = np.array([ranks.max()])
        model_ll = self.fit_model_given_ranks_helper(G, ranks)
        model_bic = 2 * np.log(2 * len(G)) - 2 * model_ll
        opt_bic = (model_bic, model_ll, 1, copy.deepcopy(self.lambda_), copy.deepcopy(self.c))
        print('Number of layers: {}, Model BIC: {}, lambda = {}, c = {}, H = {}'.format(1, model_bic, self.lambda_, self.c, self.H))

        for i in range(2, num_layers + 1):
            self.H = np.hstack(([self.H[0] / 2], self.H))
            self.c = np.hstack(([self.c[0]], self.c))
            model_ll = self.fit_model_given_ranks_helper(G, ranks)
            model_bic = (i + 1) * np.log(2 * len(G)) - 2 * model_ll
        
            if model_bic <= opt_bic[0]:
                opt_bic = (model_bic, model_ll, i, copy.deepcopy(self.lambda_), copy.deepcopy(self.c))
    
            print('Number of layers: {}, Model BIC: {}, lambda = {}, b = {}, c = {}, H = {}'.format(i, model_bic, self.lambda_, self.b, self.c, self.H))
    
        return opt_bic

    def fit_model_given_ranks_helper(self, G, ranks):
        sum_ranks = ranks.sum()
        n = len(ranks)
        bounds = ((1e-4, np.inf),) 

        res = minimize(lambda x: - CIGAM.ranks_log_likelihood(ranks, n, x, self.H, sum_ranks), 0.1, bounds=bounds, jac=lambda x: - CIGAM.ranks_log_likelihood_jacobian(ranks, n, x, self.H, sum_ranks=sum_ranks))
        ranks_ll = - res.fun[0]
        self.lambda_ = res.x[0]

        bounds = len(self.H) * ((1 + 1e-4, np.inf),)
        sizes, neg_sizes, _, _ = CIGAM.get_partition_sizes(G, ranks, self.order, self.H)

        
        res = minimize(lambda x: - CIGAM.graph_log_likelihood(G, ranks, self.H, x, self.order, sizes, neg_sizes), self.c, bounds=bounds, jac=lambda x: - CIGAM.graph_log_likelihood_jacobian(G, ranks, self.H, x, self.order, sizes, neg_sizes))
        self.c = res.x

        graph_ll = - res.fun

        return ranks_ll + graph_ll

