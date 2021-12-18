from base import *
from hypergraph import *

def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.001].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return r.fun, func(r.x)

def nanplot(x, y, **kwargs):
    plt.plot(x[~np.isnan(y)], y[~np.isnan(y)], **kwargs)

def savefig(name, ext='png'):
    fig = plt.gcf()

    plt.savefig('{}.{}'.format(name, ext))
    
    with open('{}.fig.pickle'.format(name), 'wb+') as f:
        pickle.dump(fig, f)

    return fig

def normalize_df(df, fields, method='minmax'):
    for field in fields:
        df[field] = normalize(df[field], method=method)
    return df

def normalize_array(x, method='minmax'):
    return np.apply_along_axis(lambda u: normalize(u, method), 0, x)

def normalize(x, method='minmax'):
    if method == 'minmax':
        return (x - x.min()) / (x.max() - x.min())
    elif method == 'z-score':
        return (x - x.mean()) / x.std()


@jit(nopython=True)
def truncated_exp_inverse_cdf(q, lambda_, H):
    if q < 0:
        return -np.inf
    if q > 1:
        return np.inf

    Z = np.log(1 - np.exp(-lambda_ * H[-1]))
    return - np.log(1 - q * Z) / lambda_

def get_num_non_edges(n, order_min, order_max):
    return 2**order_max - 2**(order_min - 1) 

@jit(nopython=True)
def binomial_coefficients(n, k):
        C = np.zeros(shape=(n + 1, k + 1), dtype=np.float64)

        for i in range(0, n + 1):
                for j in range(0, min(i, k) + 1):
                        if j == 0 or j == i:
                                C[i, j] = 1
                        else:
                                C[i, j] = C[i - 1, j - 1] + C[i - 1, j]

        return C

def stanfit_to_dataframe(fit, params=None):
    df = fit.to_frame()

    if params:
        temp = []
        for p in params:
            for c in df.columns:
                if c.startswith(p + '.') or c.startswith(p):
                    temp.append(c)

        df = df[temp]

    return df

@jit(nopython=True)
def generalized_mean(x, p):
    return np.linalg.norm(x.astype(np.float64)) / len(x)**(1 / p)

@jit(nopython=True)
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def mns(H, s):
    if isinstance(H, Hypergraph):
        u0, v0 = random.choice(list(H.graph.edges()))
    elif isinstance(H, nx.Graph):
        u0, v0 = random.choice(list(H.edges()))
    f = set([u0, v0])

    while len(f) < s:
        S = set([])
        for u in f:
            if isinstance(H, Hypergraph):
                for v in H.graph.neighbors(u):
                    S |= {(u, v)}
            elif isinstance(H, nx.Graph):
                for v in H.neighbors(u):
                    S |= {(u, v)}

        if len(S) == 0:
            return mns(H, s)
        else:
            u, v = random.choice(list(S))
            f |= {u, v}

    return f

def cns(H, s):
    f = set(random.choice(H.simplices).nodes)
    v = random.choice(list(f))

    while len(f) < s:
        V = set([])
        for u in f - {v}:
            for smp in H.simplex_neighbors(u):
                for w in smp.nodes:
                    V |= {w}

        if len(V) == 0:
            return cns(H, s)
        else:
            v1 = random.choice(list(V))
            f = (f - {v}) | {v1}
    return f

def int2base(x, base):
        digits = []

        while x:
                digits.append(x % base)
                x //= base

        return digits

def grass_hopping_helper(S, V, k, p, directed=True):
    assert(k > len(S))
    n = len(V)
    s = len(S)
    i = -1

    edges = set()

    q = p

    while True:
        i += np.random.geometric(q)
        decoded = int2base(i, n)

        if len(decoded) > k - s:
            break
        else:
            simplex = tuple(S + [V[int(j)] for j in decoded])
            # Perform rejection sampling if the graph is undirected
            if directed or ((not directed) and decoded == list(sorted(decoded))):
                edges.add(simplex)

    return edges

def sample_combination(n, k, n_f=0, k_f=0):
    S = set([])
    choices = list(range(n))

    # Generate combinations uniformly with rejection sampling
    while len(S) < k:
        if k_f > 0: 
            u = random.choice(choices[-n_f:])
            k_f -= 1
        else:
            u = random.choice(choices)
        
        if u not in S:
            S |= {u}
 
        if k >= len(choices) // 2 and len(S) >= len(choices) // 4:
            choices = list(set(choices) - S)
            k -= len(S)

    return list(S)

def ball_dropping_helper(S, V, k, p, n_f=0, k_f=0, existing_edges=[], directed=True):
        assert(k >= len(S))

        n = len(V)
        s = len(S)
        m_existing = len(existing_edges)

        if directed:
            m = int(np.random.binomial(n**(k - s) - (n - n_f)**(k - s), p))
        else:
            m = int(np.random.binomial(special.comb(n, k - s) - special.comb(n - n_f, k - s), p))


        edges = set(existing_edges)
        while len(edges) < m_existing + m:
                if directed:
                    e_index = np.hstack((np.random.randint(low=n-n_f, high=n, size=k_f), np.random.randint(low=0, high=n, size=k-s-k_f)))
                else:
                    e_index = sample_combination(n, k - s, n_f=n_f, k_f=k_f)
                e = tuple(S + [V[idx] for idx in e_index])
                if e not in edges:
                    edges.add(e)

        return list(edges)
