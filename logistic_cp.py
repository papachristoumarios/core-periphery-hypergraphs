def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class LogisticCP:

  def __init__(self, lambda_=1, alpha=2, beta=2):
    self.lambda_ = lambda_
    self.alpha = alpha
    self.beta = beta

    self.stan_definitions = {
        'N' : 'int N;',
        'A' : 'int<lower=0, upper=1> A[N, N];',
        'alpha' : 'real<lower=0> alpha;',
        'beta' : 'real<lower=0> beta;',
        'lambda' : 'real<lower=0> lambda;',
        'coins' : 'real<lower=0, upper=1> coins[N];',
        'intermediate_thetas' : 'real<lower=0> intermediate_thetas[N];'
    }

  @property
  def mu(self):
    return 1 / self.lambda_

  @mu.getter
  def mu(self):
    return 1 / self.lambda_

  @mu.setter
  def mu(self, mu_):
    assert(mu_ > 0)
    self.lambda_ = 1 / mu_

  def sample(self, n):
    # thetas = -np.sort(-np.random.normal(loc=self.mu, scale=self.sigma, size=n))
    coins = np.random.beta(self.alpha, self.beta, size=n)
    intermediate_thetas = np.random.exponential(scale=1 / self.lambda_, size=n)
    thetas = - np.sort(-(2 * coins - 1) * intermediate_thetas)

    G = nx.Graph()

    for u in range(n):
      G.add_node(u)
      for v in range(u):
        if u != v and np.random.uniform() <= sigmoid(thetas[u] + thetas[v]):
          G.add_edge(u, v)

    return G, coins, intermediate_thetas, thetas

  def plot_sample(self, n):
    G, coins, thetas = self.sample(n)
    A = nx.to_numpy_array(G)
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
    clf_lasso = linear_model.Lasso(alpha=alpha_lasso)
    clf_lasso.fit(log_rank.reshape(-1, 1), log_degree)
    r2 = np.corrcoef(log_rank, log_degree)[0, 1]
    plt.plot(log_rank, log_degree, linewidth=1, label='Realized Degree $R^2 = {}$'.format(round(r2, 2)))
    plt.plot(log_rank, p[1] + p[0] * log_rank, linewidth=2, label='Linear Regression')
    plt.plot(log_rank, clf_lasso.intercept_ + clf_lasso.coef_ * log_rank, linewidth=2, label='Lasso Regression ($a = {}$)'.format(alpha_lasso))
    plt.title('Degree Plot')
    plt.xlabel('Node Rank by degree (log)')
    plt.ylabel('Node Degree (log)')
    plt.legend()

  def stan_model(self, known, dump=True, load=True):

    functions_segment = '''functions {
      real sigmoid(real x) {
        return 1 / (1 + exp(-x));
      }
}'''

    model_segment = '''model {
      lambda ~ gamma(2, 2);
      alpha ~ lognormal(0, 1);
      beta ~ lognormal(0, 1);

      for (i in 1:N) {
        coins[i] ~ beta(alpha, beta);
        intermediate_thetas[i] ~ exponential(lambda);
      }

      for (i in 1:N) {
        for (j in 1:N) {
          A[i, j] ~ bernoulli(sigmoid((2 * coins[i] - 1) * intermediate_thetas[i] + (2 * coins[j] - 1) * intermediate_thetas[j]));
        }
      }
}
    '''

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

  def stan_model_sample(self, known, model_data, dump=True, load=True):
    stan_model = self.stan_model(known, dump=dump, load=load)
    fit = stan_model.sampling(data=model_data, iter=500, chains=4, seed=1)
    return fit

  def params_posterior(self):
    known = {
        'N' : True,
        'A' : True,
        'coins' : True,
        'intermediate_thetas' : True,
        'alpha' : False,
        'beta' : False,
        'lambda' : False
    }

    return known

  def latent_posterior(self):
    known = {
        'N' : True,
        'A' : True,
        'coins' : False,
        'intermediate_thetas' : False,
        'alpha' : True,
        'beta' : True,
        'lambda' : True
    }

    return known

  def params_latent_posterior(self):
    known = {
        'N' : True,
        'A' : True,
        'coins' : False,
        'intermediate_thetas' : False,
        'alpha' : False,
        'beta' : False,
        'lambda' : False
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
