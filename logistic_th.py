from base import *
from utils import *

class LogisticTH:

  def __init__(self, p=20):
    self.p = p

  def sample(self, n):
    G = nx.Graph()

    ranks = n - (1 + np.arange(n))

    for u in range(n):
      G.add_node(u)
      for v in range(u):
        if u != v and np.random.uniform() <= sigmoid(generalized_mean([ranks[u], ranks[v]], self.p) / n):
          G.add_edge(u, v)

    return G, ranks

  def display_p(self):
    if np.isinf(self.p):
      return '\\infty'
    elif np.isinf(-self.p):
      return '- \\infty'
    else:
      return self.p

  def plot_sample(self, n):
    G, ranks = self.sample(n)
    A = nx.to_numpy_array(G)
    plt.figure(figsize=(10, 10))
    plt.imshow(A)
    plt.title('Adjacency Matrix for G ~ logstic-TH($p={}$)'.format(self.display_p()))
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
    plt.title('Adjacency Matrix for G ~ logstic-TH($p={}$)'.format(self.display_p()))
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

  def fit(self, G, alpha=10):
    q = self.p / (self.p - 1)
    n = len(G)

    def helper(x, G, alpha):
      F = np.zeros_like(x)

      for i in G:
        for j in G.neighbors(i):
          F[i] += np.abs(x[i])**(alpha - 2) * x[i] * (x[i]**alpha + x[j]**alpha) ** (1 / (alpha) - 1)

      return F

    x = np.ones(shape=n)
    y = np.ones(shape=n)

    x_prev = x

    for i in range(100):
      y = helper(x, G, alpha)
      x = np.linalg.norm(y, q)**(q - 1) * np.abs(y)**(q - 2) * y
      if np.allclose(x, x_prev):
        break
      else:
        x_prev = x

    ranks = np.argsort(-x)[::-1]

    return x, ranks
