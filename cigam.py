from base import *
from utils import *

class CIGAM:

    def __init__(self, c=[1.5], b=3, H=[1], order_min=2, order_max=2, constrained=False, alpha_c=0, alpha_lambda=1, beta_lambda=0):

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
        assert(order_min >= 2 and order_max >= order_min)

        self.constrained = constrained

        # Prior hyperparams
        self.alpha_c = alpha_c
        self.alpha_lambda = alpha_lambda
        self.beta_lambda = beta_lambda

        # Hypergraph Order
        self.order_min = order_min
        self.order_max = order_max

        # Ranks parameter
        self.lambda_ = np.log(b)
        
        # Number of layers for the multi-layer model
        self.num_layers = len(self.c)

        self.stan_definitions = {
                'alpha_c' : 'real alpha_c;',
                'alpha_lambda' : 'real alpha_lambda;',
                'beta_lambda' : 'real beta_lambda;',
                'N' : 'int N;',
                'K_min' : 'int K_min;',
                'K_max' : 'int K_max;',
                'M' : 'int M[K_max - K_min + 1];',
                'L' : 'int L;',
                'H' : 'real H[L];',
                'b' : 'real<lower=1> b;',
                'lambda' : 'real<lower=0.001> lambda;',
                'c_0' : 'real<lower=0> c[L];',
                'edges' : 'int edges[K_max - K_min + 1, max(M), K_max];',
                'binomial_coefficients' : 'matrix[N + 1, K_max + 1] binomial_coefficients;',
                'ranks' : 'vector<lower=0, upper=H[L]>[N] ranks;'
        }

        if self.constrained:
            self.stan_definitions['c0'] = 'positive_ordered[L] c0;'
        else:
            self.stan_definitions['c0'] = 'vector<lower=0>[L] c0;'

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

    def sample(self, N, return_ranks=True, method='ball_dropping'):
        assert(method in ['ball_dropping', 'naive'])
        heights = self.continuous_tree_sample(N=N)
        ordering  = np.argsort(heights)
        heights = heights[ordering]
        ranks = self.H[-1] - heights
        G = Hypergraph()

        layers, num_layers = CIGAM.get_layers(ranks, self.H)

        if method == 'naive':
            for order in range(self.order_min, self.order_max + 1):
                for edge in itertools.combinations(range(N), order):
                    edge = np.array(edge)
                    heights_min = heights[edge].min()
                    heights_argmax = edge[heights[edge].argmax()]
                    p_edge = self.c[layers[heights_argmax]]**(-1-heights_min)
                    if np.random.uniform() <= p_edge:
                        G.add_simplex_from_nodes(nodes=edge.tolist(), simplex_data = {})
        else:
            for order in range(self.order_min, self.order_max + 1):
                for i in range(heights.shape[0] - order + 1):
                    j = i + 1
                    for l in range(num_layers.shape[-1]):
                        if num_layers[i, l] >= 1 and j + num_layers[i, l] - i >= order - 1:
                            batch = ball_dropping_helper(S=[ordering[i]], V=ordering[i+1:j+num_layers[i, l]], k_f=1, n_f=num_layers[i, l], p=self.c[l]**(-1-heights[i]), k=order, directed=False)       
                            for edge in batch:
                                G.add_simplex_from_nodes(nodes=edge, simplex_data = {})
                    
                        j += num_layers[i, l]

        if return_ranks:
            return G, ranks
        else:
            return G, heights

    def plot_sample(self, n, kind='degree'):
        H, h = self.sample(n)
        
        if kind == 'adjacency': 
            G = H.clique_decomposition()
            A = nx.to_numpy_array(G)
            plt.figure(figsize=(10, 10))
            plt.imshow(A)
            plt.title('Adjacency Matrix for G ~ CIGAM($c$={}, $b$={}, $H$={})'.format(self.c, self.b, self.H))
            plt.xlabel('Ranked Nodes by $h(u)$')
            plt.ylabel('Ranked Nodes by $h(u)$')
            plt.savefig('adjacency_matrix.png')
        elif kind == 'degree':
            degrees = H.degrees()

            plt.figure(figsize=(10, 10))
            log_rank = np.log(1 + np.arange(len(degrees)))
            log_degree = np.log(1 + degrees)
            log_degree = -np.sort(-log_degree)
            plt.plot(log_rank, log_degree, linewidth=0, marker='x', label='Realized Degree')
            _, (px, py) = segments_fit(log_rank, Y=log_degree, count=len(self.c))
            plt.plot(px, py, marker='o', linewidth=3, label='Piecewise Linear Fit')

            plt.xlabel('Node Rank by $h(u)$ (log)')
            plt.ylabel('Node Degree (log)')
            plt.title('Degree Plot')
            plt.legend()
            plt.savefig('degree_plot.eps')

    def stan_model(self, known, dump=True, load=True, method='optimized', latent_ranks=False):
            
        with open('cigam_functions.stan') as f:
            functions_segment = f.read()

        with open('cigam_transformed_data.stan') as f:
            transformed_data_segment = f.read()

        with open('cigam_model.stan') as f:
            model_segment = f.read()

        with open('cigam_sample_truncated_ranks.stan') as f:
            sample_ranks_segment = f.read()

        with open('cigam_transformed_parameters.stan') as f:
            transformed_params_segment = f.read()


        if latent_ranks:
            transformed_data_segment = transformed_data_segment.replace('// [SAMPLE RANKS]', sample_ranks_segment)
            model_segment = 'model {\n' + transformed_data_segment + model_segment + '\n}'
            transformed_data_segment = ''
        else:
            model_segment = model_segment.replace('// [SAMPLE RANKS]', sample_ranks_segment)
            model_segment = 'model {\n' + model_segment + '\n}'
            transformed_data_segment = 'transformed data {\n' + transformed_data_segment + '\n}'

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

        model_code = '{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}'.format(functions_segment, data_segment, transformed_data_segment, params_segment, transformed_params_segment, model_segment)

        model_name = '{}_given_{}{}'.format('_'.join(params_keys), '_'.join(data_keys), '_constrained' if self.constrained else '')

        with open(model_name + '.stan', 'w+') as f:
            f.write(model_code)

        return model_code, model_name

    def continuous_tree_sample(self, N):
        u = np.random.uniform(size=N)
        y = np.log(u * (self.b**self.H[-1] - 1) + 1) / np.log(self.b)
        return y

    def stan_model_sample(self, known, model_data, dump=True, load=True, latent_ranks=False):
        model_code, model_name  = self.stan_model(known, dump=dump, load=load, latent_ranks=latent_ranks)
        stan_model = stan.build(program_code=model_code, data=model_data)

        fit = stan_model.sample(num_samples=2000, num_chains=4, init=[{'lambda' : 0.5} for _ in range(4)])
    
        return fit

    def find_number_of_layers(self, G, layers_max=10):
        degrees = G.degrees()
        log_degrees = np.log(1 + degrees) 
        log_degrees = - np.sort(- log_degrees)
        log_num_ranks = np.log(1 + np.arange(0, len(log_degrees)))

        log_err = []
        num_layers = []
        breakpoints = []

        for l in range(1, layers_max + 1):
            err, (px, py) = segments_fit(log_num_ranks, log_degrees, count=l)
            if not np.isnan(err):
                log_err.append(np.log(err))
                num_layers.append(l)
                breakpoints.append(px)

        plt.figure()
        plt.plot(num_layers, log_err)
        plt.savefig('elbow.png') 

        argmax = -1
        maximum = -1

        for i in range(1, len(log_err) - 1):
            if (log_err[i] - log_err[i - 1]) / (log_err[i + 1] - log_err[i]) >= maximum:
                argmax = i
                maximum = (log_err[i] - log_err[i - 1]) / (log_err[i + 1] - log_err[i])

        print('Best number of layers', num_layers[argmax])

        return num_layers[argmax], log_err, num_layers, breakpoints

    def find_hyperparameters(self, G, features, learnable_ranks=False, layers_max=10, step=0.2, bayesian=True, num_epochs=50, criterion='spearman'):

        l, _, _, _ = self.find_number_of_layers(G, layers_max=layers_max)

        h_space = np.arange(0, 1, step)

        argmax = [1]
        maximum = - sys.maxsize

        for h in itertools.product(h_space, repeat=l-1):
            h = list(h)
            if all([h[i] < h[i + 1] for i in range(len(h) - 1)]):
                if len(h) > 0 and h[0] == 0: 
                    h = h[1:]
                print(h + [1])
                if bayesian and not learnable_ranks:
                    fit = self.fit_model_bayesian(G, H=np.array(h + [1.0]), ranks=features)
                    ll = fit['lp__'].max()
                else:   
                    ll, spearman,  _ = self.fit_model_given_ranks_torch(G, H=np.array(h + [1.0]),  features=features, learnable_ranks=learnable_ranks, num_epochs=num_epochs)
                if ((criterion == 'log-posterior' or not learnable_ranks) and ll >= maximum):
                    maximum = ll
                    argmax = np.array(h + [1.0])
                elif spearman >= maximum and learnable_ranks and criterion == 'spearman': 
                    maximum = spearman
                    argmax = np.array(h + [1.0])

        self.H = argmax

        return argmax

    def params_posterior(self):
        known = {
                'alpha_c': True,
                'alpha_lambda': True,
                'beta_lambda' : True,
                'N' : True,
                'L' : True,
                'K_min' : True,
                'K_max' : True,
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : True,
                'lambda' : False,
                'c0' : False
        }

        return known

    def latent_posterior(self):
        known = {
                'alpha_c': True,
                'alpha_lambda': True,
                'beta_lambda' : True,
                'N' : True,
                'L' : True,
                'K_min' : True,
                'K_max' : True,
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : False,
                'lambda' : True,
                'c0' : True
        }

        return known

    def params_latent_posterior(self):
        known = {
                'alpha_c': True,
                'alpha_lambda': True,
                'beta_lambda' : True,
                'N' : True,
                'L' : True,
                'K_min' : True,
                'K_max' : True,
                'M' : True,
                'H' : True,
                'edges' : True,
                'binomial_coefficients' : True,
                'ranks' : False,
                'lambda' : False,
                'c0' : False
        }

        return known

    def visualize_posterior(self, fit, params=None, method='hist', outfile='posterior.png'):

        if params is None:
            params = list(fit.keys())

        if method == 'plot':
            assert(len(params) == 1)

        df = stanfit_to_dataframe(fit, params)

        params = [col for col in df.columns]


        if method == 'pairplot':
            sns.pairplot(df, x_vars=params, y_vars=params, kind='kde')

        elif method == 'hist':
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

        elif method == 'plot':
            plt.figure(figsize=(10, 10))
            means = []
            stds = []
            for param in params:
                if param == 'lp_':
                    continue
                means.append(df[param].mean())
                stds.append(df[param].std())

            means = np.array(means)
            stds = np.array(stds)
            layers = np.arange(1, len(params) + 1)
            plt.plot(layers, means, color='b', marker='o')
            plt.fill_between(layers, means - stds, means + stds, color='b', alpha=0.3)            

            plt.yticks(layers)
            plt.ylabel('Range')
            plt.xlabel('Layer')

        else:
            raise NotImplementedError()

        plt.savefig(outfile)

    def visualize_degree_plot(self, G, fit, outfile='degree.png'):

        degrees_init = 1 + np.array([G.degree(u) for u in range(len(G))])

        ranks = fit['ranks']
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

        plt.savefig(outfile)

    def fit_model_bayesian(self, G, H, ranks=None):
        edges = G.to_index()
        N = len(G)
        M = G.num_simplices(separate=True)
        binomial_coeffs = binomial_coefficients(N, self.order_max) 
       
        data = {
                'alpha_c' : self.alpha_c,
                'alpha_lambda' : self.alpha_lambda,
                'beta_lambda' : self.beta_lambda,
                'N' : len(G),
                'K_min' : self.order_min,
                'K_max' : self.order_max,
                'L' : len(H),
                'M' : M,
                'H' : H,
                'edges' : edges,
                'binomial_coefficients' : binomial_coeffs
        }

        if ranks is None:
            fit = self.stan_model_sample(self.params_latent_posterior(), data, latent_ranks=True)
        else:
            data['ranks'] = ranks
            fit = self.stan_model_sample(self.params_posterior(), data, latent_ranks=False)

        self.lambda_ = fit['lambda'].mean()
        
        # self.b = np.exp(fit['lambda']).mean()
        self.c = fit['c'].mean(0)
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
    def graph_log_likelihood(G, ranks, H, c, order_min, order_max, sizes=None, neg_sizes=None, layers=None, num_layers=None):
    
        if sizes is None or  neg_sizes is None or layers is None or num_layers is None:
            sizes, neg_sizes, layers, num_layers = CIGAM.get_partition_sizes(G, ranks, order_min, order_max, H)

        result = np.sum([sizes[order - order_min, i, l] * np.log(c[l]) * (-1-H[-1]+ranks[i]) + neg_sizes[order - order_min, i, l] * np.log(1 - c[l]**(-1-H[-1]+ranks[i])) for i in range(len(ranks)) for l in range(len(H)) for order in range(order_min, order_max + 1)])
        print(result)
        return result

    @staticmethod
    #@jit(forceobj=True)
    def graph_log_likelihood_jacobian(G, ranks, H, c, order_min, order_max, sizes=None, neg_sizes=None, layers=None, num_layers=None):
        if sizes is None or neg_sizes is None or layers is None or num_layers is None:
            sizes, neg_sizes, layers, num_layers = CIGAM.get_partition_sizes(G, ranks, order_min, order_max, H)
        return np.array([np.sum([
            sizes[order - order_min, u, l] * (-1 - H[-1] + ranks[u]) / (c[l]) -
            #neg_sizes[u] * (-1 - H[-1] + ranks[u]) * c[-1]**(-2 + ranks[u] - H[-1]) / (1 - c[-1]**(-1-H[-1]+ranks[u])) 
            neg_sizes[order - order_min, u, l] / (c[l]) * (-1-H[-1] + ranks[u]) * (c[l])**(-2-H[-1] + ranks[u])
            for u in G.nodes() for order in range(order_min, order_max + 1)]) for l in range(len(H))])
        
    @staticmethod
    def complete_log_likelihood(G, ranks, H, lambda_, c, order_min, order_max, sizes=None, neg_sizes=None, layers=None, num_layers=None, sum_ranks=None):
        return CIGAM.ranks_log_likelihood(None if sum_ranks is not None else ranks, len(G), lambda_, H, sum_ranks) + CIGAM.graph_log_likelihood(G, ranks, H, c, order_min, order_max, sizes=sizes, neg_sizes=neg_sizes, layers=layers, num_layers=num_layers)

    @staticmethod
    @jit(forceobj=True)
    def q_function(G, ranks_post, H, lambda_, c, order_min, order_max, sizes_post=None, neg_sizes_post=None, sum_ranks=None):
        if sizes_post is None or neg_sizes_post is None:
            sizes_post, neg_sizes_post, layers_post, num_layers_post = CIGAM.get_partition_sizes(G, ranks_post, order_min, order_max, H)
        if sum_ranks is None:
            sum_ranks = ranks_post.sum(1)

        return np.mean([CIGAM.complete_log_likelihood(G, ranks_post[i, :], H, lambda_, c, order_min, order_max, sizes=sizes_post[:, i, :], neg_sizes=neg_sizes_post[:, i, :], sum_ranks=sum_ranks[i]) for i in range(ranks_post.shape[0])])

    @staticmethod
    def complete_log_likelihood_jacobian(G, ranks, H, lambda_, c, order_min, order_max, sizes=None, neg_sizes=None, sum_ranks=None, layers=None, num_layers=None):
        return np.hstack(([CIGAM.ranks_log_likelihood_jacobian(None if sum_ranks is not None else ranks, len(G), lambda_, H, sum_ranks)], CIGAM.graph_log_likelihood_jacobian(G, ranks, H, c, order_min, order_max, sizes=sizes, neg_sizes=neg_sizes, layers=layers, num_layers=num_layers)))

    @staticmethod
    @jit(nopython=True)
    def ranks_log_likelihood(ranks, n, lambda_, H, sum_ranks=None, num_terms=2):
        if sum_ranks is None:
            sum_ranks = np.nansum(ranks)
        Z = np.sum(np.exp(-np.arange(1, 1 + num_terms).astype(np.float64) * lambda_ * H[-1]))
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
        layers = np.zeros(shape=ranks.shape, dtype=np.uint64)
        num_layers = np.zeros(shape=ranks.shape + (len(H),), dtype=np.uint64)
    
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
    def get_partition_sizes(G, ranks, order_min, order_max, H):
        sizes = np.zeros(shape=(order_max - order_min + 1,) + ranks.shape + (len(H),))
        neg_sizes = np.zeros(shape=sizes.shape, dtype=np.uint64)
        binomial_coeffs = binomial_coefficients(len(G), order_max - 1)
        layers = np.zeros(shape=ranks.shape, dtype=np.uint64)
        num_layers = np.zeros(shape=ranks.shape + (len(H),), dtype=np.uint64)

        if len(ranks.shape) == 1:
            layers, num_layers = CIGAM.get_layers(ranks, H)
            if G is not None:
                for edge in G.edges():
                    order = len(edge)
                    edge_index = edge.to_index()
                    argmax = edge_index[np.argmax(ranks[edge_index])]
                    argmin = layers[edge_index[np.argmin(ranks[edge_index])]]
                    sizes[order - order_min, argmax, argmin] += 1            

            for order in range(order_min, order_max + 1):
                for i in range(num_layers.shape[0] - 1):
                    j = i + 1
                    for l in range(num_layers.shape[1]):
                        neg_sizes[order - order_min, i, l] = binomial_coeffs[int(j + num_layers[i, l] - i), order - 1] - binomial_coeffs[int(j - i - 1), order - 1]  - sizes[order - order_min, i, l]
                        j += num_layers[i, l]
                    
        else:
            for i in range(ranks.shape[0]):
                sizes_temp, neg_sizes_temp, layers_temp, num_layers_temp = CIGAM.get_partition_sizes(G, ranks[i], order_min, order_max, H)
                sizes[i] = sizes_temp
                neg_sizes[i] = neg_sizes_temp
                layers[i] = layers_temp
                num_layers[i] = num_layers_temp
            
        return sizes, neg_sizes, layers, num_layers

    def fit_model_given_ranks_helper(self, G, ranks):
        sum_ranks = ranks.sum()
        n = len(ranks)
        bounds = ((1e-4, np.inf),) 
        res = minimize(lambda x: - CIGAM.ranks_log_likelihood(ranks, n, x, self.H, sum_ranks), 0.1, bounds=bounds, jac=lambda x: - CIGAM.ranks_log_likelihood_jacobian(ranks, n, x, self.H, sum_ranks=sum_ranks))
        ranks_ll = - res.fun[0]
        self.lambda_ = res.x[0]
        print(ranks_ll)
        bounds = len(self.H) * ((1 + 1e-4, np.exp(self.lambda_)),)
        sizes, neg_sizes, _, _ = CIGAM.get_partition_sizes(G, ranks, self.order_min, self.order_max, self.H)
        
        # res = minimize(lambda x: - CIGAM.graph_log_likelihood(G, ranks, self.H, x, self.order_min, self.order_max, sizes, neg_sizes), self.c, bounds=bounds)
        res = minimize(lambda x: - CIGAM.graph_log_likelihood(G, ranks, self.H, x, self.order_min, self.order_max, sizes, neg_sizes), self.c, bounds=bounds, jac=lambda x: - CIGAM.graph_log_likelihood_jacobian(G, ranks, self.H, x, self.order_min, self.order_max, sizes, neg_sizes))
        self.c = res.x

        graph_ll = - res.fun

        return ranks_ll + graph_ll

    @staticmethod
    def impute_ranks(ranks): 
        ranks_not_nan = ranks[~np.isnan(ranks)]
        sum_ranks_not_nan = np.sum(ranks_not_nan)
        n_not_nan = len(ranks_not_nan)
        n_nan = len(ranks) - n_not_nan 
        bounds = ((1e-4, np.inf),)

        res = minimize(lambda x: - CIGAM.ranks_log_likelihood(ranks_not_nan, n_not_nan, x, [1], sum_ranks_not_nan), 0.1, bounds=bounds, jac=lambda x: - CIGAM.ranks_log_likelihood_jacobian(ranks_not_nan, n_not_nan, x, [1], sum_ranks=sum_ranks_not_nan))

        lambda_not_nan = res.x[0]
        b_not_nan = np.exp(lambda_not_nan)

        u = np.random.uniform(size=n_nan)
        y = 1 - np.log(u * (b_not_nan - 1) + 1) / lambda_not_nan
        
        ranks[np.isnan(ranks)] = y

        return ranks
    
    def fit_model_given_ranks_torch(self, G, H, features, learnable_ranks=False, num_epochs=50, max_patience=5, ranks_col=0, early_stopping='log-posterior'):
        
        if len(features.shape) == 1:
            features = features.reshape(features.shape[0], 1)
        
        gold_ranks = features[:, ranks_col]
        features = torch.from_numpy(features.astype(np.float32)).cuda()
        
        ranks_model = CIGAMRanksTorchModel(feature_dims=features.shape[-1], order_min=self.order_min, order_max=self.order_max, H=H, n=len(G), learnable_ranks=learnable_ranks).cuda()
        graph_model = CIGAMGraphTorchModel(order_min=self.order_min, order_max=self.order_max, H=H).cuda()
        
        edges = torch.from_numpy(G.to_index()).cuda()
        
        if learnable_ranks:
            optimizer = torch.optim.SGD(list(graph_model.parameters()) + list(ranks_model.parameters()), lr=1e-6)
        else:
            optimizer = torch.optim.SGD(graph_model.parameters(), lr=1e-6)
        
        log_posteriors = []
        spearmans = []
        max_log_posterior = - sys.maxsize
        max_spearman = - sys.maxsize
        opt_graph_model = None
        opt_ranks_model = None
        patience = 0

        pbar = tqdm(range(num_epochs))

        for i in range(num_epochs):
            if learnable_ranks or (not learnable_ranks and i == 0):
                ranks, sizes, neg_sizes = ranks_model(features, edges) 
            y_pred, ranks_log_likelihood = graph_model(ranks)

            neg_log_posterior = - torch.sum(sizes.sum(0) * torch.log(y_pred) + neg_sizes.sum(0) * torch.log(1 - y_pred)) + \
                                - ranks_log_likelihood + self.alpha_c * torch.sum(graph_model.c) - (self.alpha_lambda - 1) * torch.log(graph_model.lambda_) + self.beta_lambda * graph_model.lambda_
            if self.constrained:
                neg_log_barrier = - torch.log(graph_model.lambda_) - torch.log(graph_model.c[0] - 1) - torch.log(torch.exp(graph_model.lambda_) - graph_model.c[-1])
                for i in range(len(H) - 1):
                    neg_log_posterior -= torch.log(graph_model.c[i + 1] - graph_model.c[i])
            else:
                neg_log_barrier = torch.tensor(0).cuda()

            loss = neg_log_posterior + neg_log_barrier
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            log_posteriors.append(-neg_log_posterior.item())
            
            if learnable_ranks:
                spearmans.append(scipy.stats.spearmanr(gold_ranks, ranks.clone().detach().cpu().numpy()[:, ranks_col]).correlation)
                pbar.set_description('Loss: {}, Spearman: {}'.format(log_posteriors[-1], spearmans[-1]))
            else:
                spearmans.append(1)
                pbar.set_description('Loss: {}'.format(log_posteriors[-1]))
            pbar.update()

            # Early stopping
            if not np.isnan(log_posteriors[-1]) and max_log_posterior <= log_posteriors[-1] and early_stopping == 'log-posterior':
                max_spearman = spearmans[-1]
                max_log_posterior = log_posteriors[-1]
                opt_graph_model = copy.deepcopy(graph_model)
                opt_ranks_model = copy.deepcopy(ranks_model)
            elif learnable_ranks and not np.isnan(log_posteriors[-1]) and max_spearman <= spearmans[-1] and early_stopping == 'spearman':
                max_spearman = spearmans[-1]
                max_log_posterior = log_posteriors[-1]
                opt_graph_model = copy.deepcopy(graph_model)
                opt_ranks_model = copy.deepcopy(ranks_model)
            if i > 2:
                if (log_posteriors[-1] < log_posteriors[-2] and early_stopping == 'log-posterior') or (spearmans[-1] > spearmans[-2] and learnable_ranks and early_stopping == 'spearman'):
                    patience += 1
                else:
                    patience = 0

                if patience == max_patience:
                    break
        pbar.close()

        self.c = opt_graph_model.c.detach().cpu().numpy()
        self.lamdba_ = opt_graph_model.lambda_.detach().cpu().numpy()
        ranks = ranks.detach().cpu().numpy()

        return max_log_posterior, max_spearman, ranks


class CIGAMRanksTorchModel(nn.Module):

    def __init__(self, feature_dims, order_min, order_max, H, n, learnable_ranks):
        super().__init__()
        self.feature_dims = feature_dims
        self.learnable_ranks = learnable_ranks
        self.ranks_nn = nn.Sequential(nn.Linear(self.feature_dims, 1), nn.Sigmoid())

        self.binomial_coeffs = torch.from_numpy(binomial_coefficients(n, order_max - 1))
        self.order_min = order_min
        self.order_max = order_max
        self.H = H

    def forward(self, features, edges):
        if self.learnable_ranks:
            ranks = self.ranks_nn(features)
            ranks, edges = self.sort_ranks(ranks, edges)
        else:
            ranks = features

        sizes, neg_sizes, layers, num_layers = self.get_partition_sizes(ranks, edges)

        return ranks, sizes, neg_sizes

    def sort_ranks(self, ranks, edges):
        sort_values = ranks.sort(0, descending=True)
        ordered_ranks = sort_values.values
        ordering = sort_values.indices

        ordered_edges = - torch.ones_like(edges) 
        
        for i, order in enumerate(range(self.order_min, self.order_max + 1)):
            for j in range(edges.size(1)):
                if torch.all(edges[i, j, :order] < 0):
                    break
                else:
                    ordered_edges[i, j, :order] = ordering[:, 0][edges[i, j, :order]]

        return ordered_ranks, ordered_edges
    
    def get_partition_sizes(self, ranks, edges):
        sizes = torch.zeros(self.order_max - self.order_min + 1, ranks.size(0), len(self.H)).cuda()
        neg_sizes = torch.zeros(self.order_max - self.order_min + 1, ranks.size(0), len(self.H)).cuda()

        layers, num_layers = self.get_layers(ranks)

        for j, order in enumerate(range(self.order_min, self.order_max + 1)):
            for edge in edges[j]:
                edge_index = edge[:order]
                argmax = edge_index[torch.argmax(ranks[edge_index])]
                argmin = layers[edge_index[torch.argmin(ranks[edge_index])]]
                
                sizes[order - self.order_min, int(argmax.item()), int(argmin.item())] += 1            

            for order in range(self.order_min, self.order_max + 1):
                for i in range(num_layers.size(0) - 1):
                    j = i + 1
                    for l in range(num_layers.size(1)):
                        neg_sizes[order - self.order_min, i, l] = self.binomial_coeffs[int(j + num_layers[i, l] - i), order - 1] - self.binomial_coeffs[int(j - i - 1), order - 1]  - sizes[order - self.order_min, i, l]
                        j += num_layers[i, l]

        return sizes, neg_sizes, layers, num_layers

    def get_layers(self, ranks):
    
        layers = torch.zeros(ranks.size(0)).cuda()
        num_layers = torch.zeros(ranks.size(0), len(self.H)).long().cuda()
        
        j = 0
   
        for i in range(ranks.size(0)):
            if self.H[-1] - ranks[i, 0].item() > self.H[j]:
                j += 1
            layers[i] = j 

        j = 0

        temp = torch.zeros(len(self.H)).long().cuda()

        for l in range(len(self.H)):
            temp[l] = torch.where(layers == l)[0].size(0)
            
        for i in range(ranks.size(0) - 1):
            while temp[j].item() == 0 and j < len(self.H):
                j += 1
            temp[j] -= 1

            num_layers[i] = torch.clone(temp)

        return layers, num_layers


class CIGAMGraphTorchModel(nn.Module):

    def __init__(self, order_min, order_max, H):
        super().__init__()

        self.order_min = order_min
        self.order_max = order_max
        self.H = H
        self.num_layers = len(H)

        self.c = nn.Parameter(torch.linspace(1.1, torch.exp(torch.tensor(1)) - 0.1, self.num_layers).float(), requires_grad=True)
        self.lambda_ = nn.Parameter(torch.tensor(1.1).float(), requires_grad=True) 
    
        self.log_simgoid = nn.LogSigmoid()

    def forward(self, ranks):

        n = ranks.size(dim=0)    

        y_pred = torch.empty(n, self.num_layers).cuda()

        for i in range(n):
            y_pred[i, :] = torch.pow(self.c, -1 - self.H[-1] + ranks[i])
   
        ranks_log_likelihood = n * torch.log(self.lambda_) + n * torch.sum(ranks) + n * self.log_simgoid(self.lambda_)

        return y_pred, ranks_log_likelihood
