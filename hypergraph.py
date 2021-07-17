from base import *

class Simplex:

  def __init__(self, nodes=[], timestamp=None):
    self.nodes = nodes
    self.timestamp = timestamp

  def __len__(self):
    return len(self.nodes)

  def add_node(self, node):
    self.nodes.append(node)

class Hypergraph:

  def __init__(self):
    self.nodes = collections.defaultdict(bool)
    self.simplices = []
    self.pointers = collections.defaultdict(list)
    self.graph = nx.Graph()
    self.node_data = collections.defaultdict(dict)

  def add_simplex_from_nodes(self, nodes, timestamp=None):
    simplex = Simplex(nodes=nodes, timestamp=timestamp)
    self.add_simplex(simplex)

  def add_simplex(self, simplex):
    self.simplices.append(simplex)
    for node in simplex.nodes:
      self.nodes[node] = True
      self.pointers[node].append(len(self.simplices) - 1)

    for u in simplex.nodes:
      for v in simplex.nodes:
        if u != v:
          self.graph.add_edge(u, v, timestamp=simplex.timestamp)

  def __getitem__(self, node):
    return self.node_data[node]

  def set_attribute(self, node, key, value):
    self.node_data[node][key] = value

  def __setitem__(self, node, data):
    assert(isinstance(data, dict))
    self.node_data[node] = data
    return data

  def simplex_neighbors(self, node):
    for pointer in self.pointers[node]:
      yield self.simplices[pointer]

  def nodes(self):
    for key in self.nodes:
      yield key

  def simplices_iter(self):
    for simplex in self.simplices:
      yield simplex

  def __len__(self):
    return len(self.nodes)

  def num_simplices(self):
    return len(self.simplices)

  def degrees(self):
    degrees = collections.defaultdict(int)
    for simplex in self.simplices:
      for node in simplex.nodes:
        degrees[node] += 1

    return degrees

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
      for i in range(len(simplex.nodes)):
        for j in range(i):
          G.add_edge(simplex.nodes[i], simplex.nodes[j])

    G = self.set_node_attributes_to_graph(G)

    return G

  def star_decomposition(self):
    G = nx.Graph()
    for simplex in self.simplices:
      node_name = ','.join([str(node) for node in simplex.nodes])
      for u in simplex.nodes:
        G.add_edge(u, node_name)

    G = self.set_node_attributes_to_graph(G)

    return G

  def to_csr(self):
    temp_indices = collections.defaultdict(list)
    for simplex in self.simplices:
      temp_indices[len(simplex)].append(simplex.nodes)

    csr = {}

    for key in temp_indices:
      coords = np.array(temp_indices[key]).T
      data = 1
      shape = tuple(key * [self.__len__(), ])
      csr[key] = sparse.COO(coords=coords, data=data)

    return csr

  def to_dense(self):
    csr = self.to_csr()
    dense = {}
    for key, val in csr.items():
      dense[key] = val.todense()

    return dense

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
