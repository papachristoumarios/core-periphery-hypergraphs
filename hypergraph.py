from base import *

class Simplex:

    def __init__(self, nodes=[], simplex_data={}):
        self.nodes_ = nodes
        self.simplex_data = simplex_data

    def __len__(self):
        return len(self.nodes_)

    def __repr__(self):
        return '[{}]'.format(', '.join(map(str, self.nodes_)))

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

    def add_simplex_from_nodes(self, nodes, simplex_data={}):
        simplex = Simplex(nodes=nodes, simplex_data=simplex_data)
        self.add_simplex(simplex)

    def add_simplex(self, simplex):
        self.simplices.append(simplex)
        for node in simplex.nodes():
            self.nodes_[node] = True
            self.pointers[node].append(len(self.simplices) - 1)

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
        return len(self.pointers[node])

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

    def num_simplices(self):
        return len(self.simplices)

    def degrees(self):
        degrees = collections.defaultdict(int)
        for simplex in self.simplices:
            for node in simplex.nodes():
                degrees[node] += 1

        return degrees

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

    def clique_decomposition(self):
        G = nx.Graph()
        for simplex in self.simplices:
            for i in range(len(simplex.nodes_)):
                for j in range(i):
                    G.add_edge(simplex.nodes_[i], simplex.nodes_[j])

        G = self.set_node_attributes_to_graph(G)
        return G

    def star_decomposition(self):
        G = nx.Graph()
        for simplex in self.simplices:
            node_name = ','.join([str(node) for node in simplex.nodes()])
            for u in simplex.nodes():
                G.add_edge(u, node_name)

        G = self.set_node_attributes_to_graph(G)

        return G

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
                H_new[mapping[u]] = copy.deepcopy(data)
                
            return H_new
        elif isinstance(H, nx.Graph):
            return nx.convert_node_labels_to_integers(H)

    @staticmethod
    def convert_node_labels_to_integers_with_field(H, field):
      
        H = Hypergraph.convert_node_labels_to_integers(H, mapping=None)
        values = np.zeros(len(H))

        for u, data in H.nodes(data=True):
            values[u] = data[field]
            
        values = np.nan_to_num(values)

        ordering = np.argsort(-values)
        
        mapping = dict([(ordering[i], i) for i in range(len(values))])
         
        return Hypergraph.convert_node_labels_to_integers(H, mapping=mapping), values[ordering]

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
