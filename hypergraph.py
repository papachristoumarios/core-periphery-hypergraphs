from base import *
from utils import *

class Simplex:

    def __init__(self, nodes=[], simplex_data={}):
        self.nodes_ = nodes
        self.simplex_data = simplex_data

    def __len__(self):
        return len(self.nodes_)

    def __repr__(self):
        return '[{}]'.format(', '.join(map(str, self.nodes_)))

    def __eq__(self, other):
        return tuple(sorted(self.nodes_)) == tuple(sorted(other.nodes_))

    def __hash__(self):
        return hash(tuple(sorted(self.nodes_)))

    def add_node(self, node):
        self.nodes_.append(node)

    def nodes(self):
        for node in self.nodes_:
            yield node

    def set_attribute(self, key, val):
        self.simplex_data[key] = val

    def get_attribute(self, key):
        return self.simplex_data.get(key, None)

    def to_index(self, dtype=np.array):
        return dtype(self.nodes_)

class Hypergraph:

    def __init__(self):
        self.nodes_ = collections.defaultdict(bool)
        self.simplices = []
        self.pointers = collections.defaultdict(list)
        self.node_data = collections.defaultdict(dict)
        self.degrees_ = collections.defaultdict(int)
        self.simplex_sizes = collections.defaultdict(int)

    def add_simplex_from_nodes(self, nodes, simplex_data={}):
        simplex = Simplex(nodes=nodes, simplex_data=simplex_data)
        self.add_simplex(simplex)

    def add_simplex(self, simplex):
        self.simplices.append(simplex)
        self.simplex_sizes[len(simplex)] += 1
        for node in simplex.nodes():
            self.nodes_[node] = True
            self.pointers[node].append(len(self.simplices) - 1)
            self.degrees_[node] += 1

    def __getitem__(self, node):
        return self.node_data[node]

    def set_attribute(self, node, key, value):
        self.node_data[node][key] = value

    def __setitem__(self, node, data):
        assert(isinstance(data, dict))
        self.node_data[node] = data
        return data

    def neighbors(self, node):
        for pointer in self.pointers[node]:
            yield self.simplices[pointer]

    def degree(self, node):
        return self.degrees_[node]

    def degrees(self): 
        return np.array([self.degree(u) for u in self.nodes_])

    def degrees_histogram(self, log=True):
        if log:
            return np.histogram(np.log(1 + self.degrees()), bins='auto')
        else:
            return np.histogram(self.degrees(), bins='auto') 

    def nodes(self):
        for key in self.nodes_:
            yield key

    def simplices_iter(self):
        for simplex in self.simplices:
            yield simplex

    def edges(self):
        for simplex in self.simplices:
            yield simplex

    def nodes(self, data=False):
        for node in self.nodes_:
            if data:
                yield node, self.node_data[node]
            else:
                yield node

    def __len__(self):
        return len(self.nodes_)

    def num_simplices(self, separate=False, negate=False):
        k_min = min(self.simplex_sizes.keys())
        k_max = max(self.simplex_sizes.keys())
        num_simplices = np.zeros(k_max - k_min + 1, dtype=int)
        for k in range(k_min, k_max + 1):
            num_simplices[k - k_min] = self.simplex_sizes[k]
        if not negate:
            if separate:
                return num_simplices
            else:
                return num_simplices.sum()
        else:
            binomial_coeffs = binomial_coefficients(self.__len__(), k_max)
            neg_num_simplices = binomial_coeffs[self.__len__(), k_min:(1 + k_max)] - num_simplices
            if separate:
                return neg_num_simplices
            else:
                return neg_num_simplices.sum()

    def get_order_range(self):
        return min(self.simplex_sizes.keys()), max(self.simplex_sizes.keys())

    def get_field_range(self, field='timestamp'):
        minimum = min([s.simplex_data.get(field, np.inf) for s in self.edges()])
        maximum = max([s.simplex_data.get(field, -np.inf) for s in self.edges()])
        return minimum, maximum

    def simplices_to_list_of_lists(self):
        for edge in self.edges():
            yield edge.nodes_

    @staticmethod
    def graph_to_hypergraph(G):
        H = Hypergraph()
        for (u, v) in G.edges():
            smp = Simplex([u, v])
            H.add_simplex(smp)

        return H

    def set_node_attributes_to_graph(self, G):
        nx.set_node_attributes(G, self.node_data, "data")
        return G

    def clique_decomposition(self, dtype=nx.Graph, weighted=True):
        if dtype == nx.Graph:
            G = nx.Graph()
        else:
            G = Hypergraph()

        weights = collections.defaultdict(int)

        for simplex in self.simplices:
            for i in range(len(simplex.nodes_)):
                for j in range(i):
                    weights[i, j] += 1
                    if dtype == nx.Graph:
                        G.add_edge(simplex.nodes_[i], simplex.nodes_[j])
                    else:
                        temp = Simplex([simplex.nodes_[i], simplex.nodes_[j]], simplex_data=simplex.simplex_data)
                        G.add_simplex(temp)

        if dtype == nx.Graph:
            G = self.set_node_attributes_to_graph(G)
            for (u, v) in G.edges():
                G[u][v]['weight'] = weights[i, j]
            return G
        else:
            for u, data in self.nodes(data=True):
                G[u] = copy.deepcopy(data)
            for i in range(G.num_simplices()):
                G.simplices[i].simplex_data['weight'] = weights[G.simplices[i].nodes_[0], G.simplices[i].nodes_[1]]
                
            return G
    
    def star_decomposition(self):
        G = nx.Graph()
        for simplex in self.simplices:
            node_name = ','.join([str(node) for node in simplex.nodes()])
            for u in simplex.nodes():
                G.add_edge(u, node_name)

        G = self.set_node_attributes_to_graph(G)

        return G

    def deduplicate(self):
        H = Hypergraph()
        edges  = set([])

        for e in self.edges():
            edges |= {e}

        for e in edges:
            H.add_simplex(e)

        for u, data in self.nodes(data=True):
            H[u] = copy.deepcopy(data)

        self = H

        return self

    def to_csr(self):
        temp_indices = collections.defaultdict(list)
        for simplex in self.simplices:
            temp_indices[len(simplex)].append(simplex.nodes_)

        csr = {}

        for key in temp_indices:
            coords = np.array(temp_indices[key]).T
            data = 1
            shape = tuple(key * [self.__len__(), ])
            csr[key] = sparse.COO(coords=coords, data=data)

        return csr

    def to_dense(self):
        self = Hypergraph.convert_node_labels_to_integers(self)
        dense = {}
        for edge in self.edges():
            k = len(edge)
            if dense.get(k, None) is None:
                shape = tuple(k * [self.__len__()])
                dense[k] = np.zeros(shape=shape)
            dense[k][edge.to_index(dtype=tuple)] = 1

        return dense

    def edges_to_numpy_array(self):
        numpy_edges = collections.defaultdict(list)

        for edge in self.edges():
            k = len(edge)
            numpy_edges[k].append(edge.to_index(list))

        for key, val in numpy_edges.items():
            numpy_edges[key] = np.array(val).T

        return numpy_edges

    @staticmethod
    def convert_node_labels_to_integers(H, mapping=None):
        if isinstance(H, Hypergraph):
            if mapping is None:
                mapping = dict([(u, i) for i, u in enumerate(H.nodes())])
            
            H_new = Hypergraph()
            for edge in H.edges():
                new_edge = [mapping[u] for u in edge.nodes()]
                H_new.add_simplex_from_nodes(nodes=new_edge, simplex_data=copy.deepcopy(edge.simplex_data))

            for u, data in H.nodes(data=True):
                new_data = copy.deepcopy(data)
                new_data['label'] = u
                H_new[mapping[u]] = new_data
                
            return H_new
        elif isinstance(H, nx.Graph):
            return nx.convert_node_labels_to_integers(H)

    @staticmethod
    def convert_node_labels_to_integers_with_field(H, field, sort=True, sort_col=0):
      
        H = Hypergraph.convert_node_labels_to_integers(H, mapping=None)
        feature_dim = len(next(H.nodes(data=True))[1][field])
        values = np.zeros((len(H), feature_dim))

        for u, data in H.nodes(data=True):
            values[u] = data.get(field, np.nan * np.ones(feature_dim))
    
        if sort:
            ordering = np.argsort(-values[:, sort_col])
        
            mapping = dict([(ordering[i], i) for i in range(values.shape[0])])
         
            return Hypergraph.convert_node_labels_to_integers(H, mapping=mapping), values[ordering]
        else:
            return H, values

    @staticmethod
    def filter_by_field(H, field, minimum, maximum):
        H_new = Hypergraph()

        for e in H.edges():
            if e.simplex_data.get(field, -np.inf) >= minimum and e.simplex_data.get(field, np.inf) <= maximum:
                H_new.add_simplex(e)
        
        # TODO copy node data

        return H_new

    def to_index(self, dtype=np.array):
        if dtype == np.array:
            M = self.num_simplices(separate=True).max()
            lengths = [len(e) for e in self.edges()]
            K_max = max(lengths)
            K_min = min(lengths)
            edges = - np.ones(shape=(K_max - K_min + 1, M, K_max), dtype=np.int64)
            indexes = np.zeros(K_max - K_min + 1, dtype=int)

            for i, edge in enumerate(self.edges()):
                k = len(edge) - K_min
                edges[k, indexes[k], 0:len(edge)] = edge.to_index(np.array)
                indexes[k] += 1
        elif dtype == list:
            edges = []

            for edge in self.edges():
                edges.append(edge.to_index(list))
        elif dtype == set:
            edges = set([])

            for edge in self.edges():
                edges.add(tuple(sorted(edge.to_index(list))))

        return edges

    def incidence_matrix(self):
        H = Hypergraph.convert_node_labels_to_integers(self)
        indexed = H.to_index(dtype=np.array)
        A = np.zeros((indexed.shape[0], len(H)))
        for m in range(self.num_simplices()):
            A[m, indexed[m]] = 1
        
        return A.T

    def domination_curve(self, ordering):

        x_axis = np.arange(0, 1 + len(ordering)).astype(np.float32)
        y_axis = np.zeros(1 + len(ordering))
        
        S = set([])

        for i, v in enumerate(ordering):
            S |= {v} 
            for p in self.pointers[v]:
                for u in self.simplices[p].nodes_:
                    S |= {u}

            y_axis[i + 1] = len(S)

        y_axis = y_axis / y_axis.max()
        x_axis = x_axis / x_axis.max()

        return x_axis, y_axis

    def umhs(self, N):
        S = set([])
        perm = np.arange(0, self.num_simplices())

        for i in range(N):
            np.random.shuffle(perm) 
            S_temp = collections.defaultdict(bool)
            S_temp_ordering = []
            U = set([])

            for j in perm:
                if all([not S_temp[v] for v in self.simplices[j].nodes_]):
                    for v in self.simplices[j].nodes_:
                        S_temp[v] = True
                        S_temp_ordering.append(v)
                        U.add(v)
                        for p in self.pointers[v]:
                            for u in self.simplices[p].nodes_:
                                U.add(u)
                        if len(U) >= self.__len__():
                            break

                    if len(U) >= self.__len__():
                        break

            U = set([])
            i_stop = -1
            for i, v in enumerate(S_temp_ordering, 1):
                for p in self.pointers[v]:
                    for u in self.simplices[p].nodes_:
                        U.add(u)
                        if len(U) >= self.__len__():
                            i_stop = i
                            break
            S_temp_ordering = set(S_temp_ordering[:i])
            S = S | S_temp_ordering

        return list(S)

    def connected_components(self): 
        visited = collections.defaultdict(bool) 
        connected_components = []

        for u in self.nodes():
            if not visited[u]:
                q = collections.deque([u])
                visited[u] = True
                S = set([])
    
                while q:
                    current = q.popleft()
                    S.add(current)

                    for ptr in self.pointers[current]:
                        for v in self.simplices[ptr].nodes_:
                            if not visited[v]:
                                visited[v] = True
                                q.append(v)

                connected_components.append((len(S), S))

        return connected_components

    def largest_connected_component(self):
        _, LCC = max(self.connected_components(), key=lambda x: x[0])
        return self.subhypergraph(LCC)

    def filter_degrees(self, threshold=4):
        S = set([])
        for u in self.nodes_:
            if self.degree(u) >= threshold:
                S.add(u)
        return self.subhypergraph(S)

    def k_core(self, k=2):
        degrees = copy.deepcopy(self.degrees_)
        visited = collections.defaultdict(bool)
        
        for u in self.nodes_:
            if not visited[u]:
                q = collections.deque([u])
                
                while q:
                    current = q.pop()
                    visited[current] = True

                    for ptr in self.pointers[current]:
                        for v in self.simplices[ptr].nodes_:
                            if degrees[current] < k:
                                degrees[v] -= 1
                            if not visited[v]:
                                q.append(v)

        S = set([])
        for u in degrees.keys():
            if degrees[u] >= k:
                S.add(u)
                for ptr in self.pointers[u]:
                    for v in self.simplices[ptr].nodes_:
                        if degrees[v] >= k:
                            S.add(v)
        return self.subhypergraph(S)

    def subhypergraph(self, S):
        H = Hypergraph()
        edges = []

        for edge in self.simplices:
            if set(edge.nodes_).issubset(S):
                edges.append(edge)

        for e in edges:
            H.add_simplex(e)

        for u, data in self.nodes(data=True):
            H[u] = copy.deepcopy(data)

        return H

    def pagerank(self):
        H = self.clique_decomposition(dtype=nx.Graph, weighted=True)
        return nx.algorithms.link_analysis.pagerank(H, weight='weight')

    def clique_graph_eigenvector(self):
        H = self.clique_decomposition(dtype=nx.Graph, weighted=True)
        return nx.eigenvector_centrality(H, weight='weight', max_iter=1000)

    def borgatti_everett(self, max_iter=1000):
        H = self.clique_decomposition(dtype=nx.Graph, weighted=False)
        A = nx.to_numpy_array(H)
        degrees = A.sum(-1)
        c = np.random.rand(A.shape[0], 1)
        c[degrees == 0] = 0
        c /= np.linalg.norm(c)
        c_prev = c
        
        for _ in range(max_iter):
            num = A @ c
            den = np.sum(c**2) - c**2
            c_prev = c
            c = num / den
            c /= np.linalg.norm(c)
            if np.allclose(c, c_prev):
                break

        return c

    def centrality_features(self):
        n = self.__len__()
        assert(list(sorted([u for u in self.nodes_])) == list(range(n)))

        degrees_np = np.array([self.degree(u) for u in range(n)])
        clique_eigenvector = self.clique_graph_eigenvector()
        clique_eigenvector_np = np.array([clique_eigenvector[u] for u in range(n)])
        pagerank = self.pagerank()
        pagerank_np = np.array([pagerank[u] for u in range(n)])

        return np.log(1 +np.vstack((degrees_np, clique_eigenvector_np, pagerank_np)).T)

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
