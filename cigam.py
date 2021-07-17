from base import *
from utils import *

class CIGAM:

  def __init__(self, c=[1.5], b=3, H=[4]):

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

    self.lambda_ = np.log(b)
    self.num_layers = len(self.c)

    self.stan_definitions = {
        'N' : 'int N;',
        'L' : 'int L;',
        'H' : 'real H[L];',
        'b' : 'real<lower=1> b;',
        'lambda' : 'real<lower=0> lambda;',
        'c' : 'real<lower=1> c[L];',
        'A' : 'int<lower=0, upper=1> A[N, N];',
        'heights' : 'real<lower=0, upper=H[L]> heights[N];',
        'ranks' : 'real<lower=0, upper=H[L]> ranks[N];'
    }

  def c_helper(self, h):
    h_max = np.max(h)
    idx = np.where(h_max <= self.H)[0][0]
    return self.c[idx]

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

  def sample(self, N, return_ranks=True):
    h = self.continuous_tree_sample(N=N)
    h = np.sort(h)

    G = nx.Graph()

    for u in range(N):
      G.add_node(u)
      for v in range(u):
        idx = np.array([u, v])
        if u != v and np.random.uniform() <= self.c_helper(h[idx])**(-1-h[idx].min()):
          G.add_edge(u, v)

    if return_ranks:
      return G, self.H[-1] - h
    else:
      return G, h

  def plot_sample(self, n):
    G, h = self.sample(n)
    A = nx.to_numpy_array(G)
    plt.figure(figsize=(10, 10))
    plt.imshow(A)
    plt.title('Adjacency Matrix for G ~ CIGAM($c$={}, $b$={}, $H$={})'.format(self.c, self.b, self.H))
    plt.xlabel('Ranked Nodes by $h(u)$')
    plt.ylabel('Ranked Nodes by $h(u)$')
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

  def stan_model(self, known, dump=True, load=True):

    functions_segment = '''functions {
  int find_c(real h_max, real[] h_values, int len) {
    int j;
    j = 1;
    while (h_max >= h_values[j] && j <= len) j = j + 1;
    return j;

  }
}'''

    model_segment = '''model {

      lambda ~ gamma(2, 2);
      c ~ pareto(1, 2);
      ranks ~ exponential(lambda);

      for (i in 1:N) {
        for (j in 1:N) {
          if (ranks[i] >= ranks[j]) A[i, j] ~ bernoulli(pow(c[find_c(H[L] - ranks[j], H, L)], -1-H[L]+ranks[i]));
          else A[i, j] ~ bernoulli(pow(c[find_c(H[L] - ranks[i], H, L)], -1-H[L]+ranks[j]));
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

  def continuous_tree_sample(self, N):
    u = np.random.uniform(size=N)
    y = np.log(u * (self.b**self.H[-1] - 1) + 1) / np.log(self.b)
    return y

  def stan_model_sample(self, known, model_data, dump=True, load=True):
    stan_model = self.stan_model(known, dump=dump, load=load)
    fit = stan_model.sampling(data=model_data, iter=500, chains=4)

    return fit

  def params_posterior(self):
    known = {
        'N' : True,
        'L' : True,
        'H' : True,
        'A' : True,
        'ranks' : True,
        'lambda' : False,
        'c' : False
    }

    return known

  def latent_posterior(self):
    known = {
        'N' : True,
        'L' : True,
        'H' : True,
        'A' : True,
        'ranks' : False,
        'lambda' : True,
        'c' : True
    }

    return known

  def params_latent_posterior(self):
    known = {
        'N' : True,
        'L' : True,
        'H' : True,
        'A' : True,
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

    print(params)

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
    A = nx.to_numpy_array(G).astype(np.int64)

    optimum = (-np.inf, (lambda_old, b_old, c_old))

    for _ in range(epochs):
      # E-Step: Sample from p(heights | G, b_old, c_old)
      e_data = {
          'N' :  N,
          'L' : len(H),
          'H' : H,
          'A' : A,
          'lambda' : lambda_old,
          'c' : c_old
      }

      e_fit = cigam.stan_model_sample(cigam.latent_posterior(), e_data)

      # For every node let its rank be the mean of the corresponding sampled parameter heights[i]
      e_ranks = e_fit.extract()['ranks'].mean(0)

      # (Bayesian) M-Step: Sample from p(b, c | G, heights_avg)
      m_data = {
          'N' : N,
          'L' : len(H),
          'H' : H,
          'A' : A,
          'ranks' : e_ranks
      }

      # Let the parameters of the next iteration be the mean of the predicted parameters
      m_fit = cigam.stan_model_sample(cigam.params_posterior(), m_data)
      self.lambda_ = m_fit.extract()['lambda'].mean()
      self.b = np.exp(m_fit.extract()['lambda']).mean()
      self.c = m_fit.extract()['c'].mean(0)
      q_function = CIGAM.q_function(G, e_fit.extract()['ranks'], H, self.b, self.c)

      print('lambda = {}, c = {}, b = {}, Q = {}'.format(self.lambda_, self.c, self.b, q_function))

      if q_function >= optimum[0]:
        optimum = (q_function, (self.lambda_, self.b, self.c))

      lambda_old, c_old, b_old = self.lambda_, self.c, self.b

    self.lambda_, self.b, self.c = optimum[-1]
    print('Best fit', optimum)

    return optimum

  def fit_model_em(self, G, H, lambda_init=np.log(3.2), c_init=1.4, epochs=15):

    lambda_old = lambda_init
    b_old = np.exp(lambda_old)
    c_old = c_init
    N = len(G)
    A = nx.to_numpy_array(G).astype(np.int64)

    optimum = (-np.inf, (lambda_old, b_old, c_old))

    bounds = ((1, np.inf), (1, np.inf))
    q_values = []

    for _ in range(epochs):
      # E-Step: Sample from p(heights | G, b_old, c_old)
      e_data = {
          'N' :  N,
          'L' : len(H),
          'H' : H,
          'A' : A,
          'lambda' : lambda_old,
          'c' : c_old
      }

      e_fit = cigam.stan_model_sample(cigam.latent_posterior(), e_data)

      m_objective = lambda x: -CIGAM.q_function(G, e_fit.extract()['ranks'], H, x[0], x[1])

      res = minimize(m_objective, np.array([np.exp(lambda_old), c_old]), method='L-BFGS-B', bounds=bounds)

      self.b = res.x[0]
      self.c = res.x[1]
      self.lambda_ = np.log(self.b)
      q_function = -res.fun
      q_values.append(q_function)

      print('lambda = {}, c = {}, b = {}, Q = {}'.format(self.lambda_, self.c, self.b, q_function))

      if q_function >= optimum[0]:
        optimum = (q_function, (self.lambda_, self.b, self.c))

      if np.allclose(self.c, c_old) and np.allclose(self.b, b_old):
        break

      lambda_old, c_old, b_old = self.lambda_, self.c, self.b

    self.lambda_, self.b, self.c = optimum[-1]
    print('Best fit', optimum)

    plt.figure(figsize=(10, 10))
    plt.plot(q_values)

    return optimum

  def fit_model_bayesian(self, G, H):
    data = {
        'N' : len(G),
        'L' : len(H),
        'H' : H,
        'A' : nx.to_numpy_array(G).astype(np.int64)
    }

    fit = cigam.stan_model_sample(cigam.params_latent_posterior(), data)

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
  def ranks_log_likelihood(ranks, H, b):
    return (np.log(np.log(b)) - np.log(b**H - 1) + (H - ranks) * np.log(b)).sum()

  @staticmethod
  def graph_log_likelihood(G, ranks, H, c):
    heights = H - ranks
    result = 0
    for u in G:
      for v in G:
        if u != v:
          if G.has_edge(u, v):
            result += np.log(c**(-1 - min(heights[u], heights[v])))
          else:
            result += np.log(1 - c**(-1 - min(heights[u], heights[v])))
    return result

  @staticmethod
  def complete_log_likelihood(G, ranks, H, b, c):
    return CIGAM.ranks_log_likelihood(ranks, H, b) + CIGAM.graph_log_likelihood(G, ranks, H, c)

  @staticmethod
  def q_function(G, ranks_post, H, b, c):
    return np.mean([CIGAM.complete_log_likelihood(G, ranks_post[i, :], H, b, c) for i in range(ranks_post.shape[0])])

class CIGAMGeneralizedMean(CIGAM):

  def __init__(self, c=[1.5,], b=3, H=[4,], p=-np.inf):

    self.p = p

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

    self.lambda_ = np.log(b)
    self.num_layers = len(self.c)

    self.stan_definitions = {
        'N' : 'int N;',
        'L' : 'int L;',
        'H' : 'real H[L];',
        'b' : 'real<lower=1> b;',
        'lambda' : 'real<lower=0> lambda;',
        'c' : 'real<lower=1> c[L];',
        'A' : 'int<lower=0, upper=1> A[N, N];',
        'heights' : 'real<lower=0, upper=H[L]> heights[N];',
        'ranks' : 'real<lower=0, upper=H[L]> ranks[N];'
    }

  def c_helper(self, h):
    h_max = h.max()
    idx = np.where(h_max <= self.H)[0][0]
    return self.c[idx]


  def sample(self, N, return_ranks=True):
    h = self.continuous_tree_sample(N=N)
    h = np.sort(h)

    G = nx.Graph()

    for u in range(N):
      G.add_node(u)
      for v in range(u):
        idx = np.array([u, v])
        if u != v and np.random.uniform() <= self.c_helper(h[idx])**(-1-generalized_mean(h[idx], self.p)):
          G.add_edge(u, v)

    if return_ranks:
      return G, self.H[-1] - h
    else:
      return G, h

    def plot_sample(self, n):
      super().plot_sample(n)

class CIGAMHyper:
  def __init__(self, b=3, c=1.5, H=4, order=3, sampling_helper=ball_dropping_helper):
    self.b = b
    self.c = c
    self.H = H
    self.order = order
    self.sampling_helper = ball_dropping_helper

  def sample(self, N, return_ranks=True):
    h = self.continuous_tree_sample(N=N)
    ranks = np.argsort(h)
    h = h[ranks]
    H = Hypergraph()

    for i in range(ranks.shape[0] - self.order):
      batch = self.sampling_helper(S=[ranks[i]], V=ranks[i+1:], p=self.c**(-1-h[i]), k=self.order, directed=False)

      for edge in batch:
        H.add_simplex_from_nodes(nodes=edge, timestamp=None)

    if return_ranks:
      return H, self.H - h
    else:
      return H, h

  def continuous_tree_sample(self, N):
    u = np.random.uniform(size=N)
    y = np.log(u * (self.b**self.H - 1) + 1) / np.log(self.b)
    return y

  def plot_sample(self, n):
    assert(self.order == 2)
    H, h = self.sample(n)
    G = H.clique_decomposition()
    A = nx.to_numpy_array(G)

    degree_ranks = np.argsort(-A.sum(0))

    for i in range(A.shape[0]):
      A[i, :] = A[i, degree_ranks]

    for i in range(A.shape[1]):
      A[:, i] = A[degree_ranks, i]

    plt.figure(figsize=(10, 10))
    plt.imshow(A)
    plt.title('Adjacency Matrix for G ~ CIGAM($c$={}, $b$={}, $H$={}) (w/ ball dropping)'.format(self.c, self.b, self.H))
    plt.xlabel('Ranked Nodes by $h(u)$')
    plt.ylabel('Ranked Nodes by $h(u)$')
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
