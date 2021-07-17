from base import *

def stanfit_to_dataframe(fit, params=None):
  data = fit.extract()
  result = {}
  if params is None:
    params = data.keys()

  for key in params:
    val = data[key]
    if len(val.shape) == 1:
      result[key] = val
    else:
      for i in range(val.shape[-1]):
        result['{}[{}]'.format(key, i)] = val[:, i]
  return pd.DataFrame.from_dict(data=result)

def load_world_trade(location='/data/mp2242/world-trade/world-trade.csv', relabel=True):
  df = pd.read_csv(location)
  G = nx.convert_matrix.from_pandas_edgelist(df, source='from', target='to')

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_faculty(location='/data/mp2242/faculty/ComputerScience_edgelist.txt', relabel=True):
  df = pd.read_csv(location, sep='\t')
  G = nx.convert_matrix.from_pandas_edgelist(df, source='# u', target='v')
  vertexlist_filename = location.replace('edgelist', 'vertexlist')
  vertex_df = pd.read_csv(vertexlist_filename, sep='\t')
  vertex_df.set_index('# u', inplace=True)
  mapping = vertex_df['institution'].to_dict()
  nx.set_node_attributes(G, mapping, 'name')

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')
  return G

def load_polblogs(location='/data/mp2242/polblogs/polblogs.mtx', relabel=True):
  df = pd.read_csv(location, sep=' ', comment='%', header=None)
  G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1)
  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_airports(location='/data/mp2242/airports/USairport500.txt', relabel=True):
  df = pd.read_csv(location, sep=' ', header=None)
  G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1)
  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')
  return G

def load_pvc_enron(location='/data/mp2242/pvc-enron/pvc-enron.csv', relabel=True):
  df = pd.read_csv(location, sep=' ', header=None)
  G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)
  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')
  return G

def load_pvc_text_Reality(location='/data/mp2242/pvc-text-Reality/pvc-text-Reality.csv', relabel=True):
  df = pd.read_csv(location, sep=' ', header=None)
  G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)
  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')
  return G

def load_celegans(location='/data/mp2242/celegans', relabel=True):
  A = np.genfromtxt(os.path.join(location, 'celegans_matrix.csv'), delimiter=',', dtype=np.int64).astype(np.int64)
  locs = np.genfromtxt(os.path.join(location, 'celegans_positions.csv'), delimiter=',').astype(np.float64)
  mapping = {}
  for i, loc in enumerate(locs):
    mapping[i] = loc

  G = nx.from_numpy_array(A)

  Gccs = sorted(nx.connected_components(G), key=len, reverse=True)
  G = G.subgraph(Gccs[0])

  nx.set_node_attributes(G, mapping, "location")

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_london_underground(location='/data/mp2242/london_underground', relabel=True):
  A = np.genfromtxt(os.path.join(location, 'london_underground_network.csv'), delimiter=',', dtype=np.int64).astype(np.int64)
  locs = np.genfromtxt(os.path.join(location, 'london_underground_tubes.csv'), delimiter=',').astype(np.float64)
  names = np.genfromtxt(os.path.join(location, 'london_underground_names.csv'), delimiter='\t', dtype=str)

  mapping = {}
  for i, loc in enumerate(locs):
    mapping[i] = loc

  names_mapping = {}
  for i, name in enumerate(names):
    names_mapping[i] = name

  G = nx.from_numpy_array(A)

  Gccs = sorted(nx.connected_components(G), key=len, reverse=True)
  G = G.subgraph(Gccs[0])

  nx.set_node_attributes(G, mapping, "location")
  nx.set_node_attributes(G, names_mapping, "name")

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_open_airlines(location='/data/mp2242/open_airlines', relabel=True):
  airports = pd.read_csv(os.path.join(location, 'airports.dat'), header=None).iloc[:, [4, 6, 7]]
  routes = pd.read_csv(os.path.join(location, 'routes.dat'), header=None).iloc[:, [2, 4]]
  G = nx.convert_matrix.from_pandas_edgelist(routes, source=2, target=4, create_using=nx.Graph)
  Gccs = sorted(nx.connected_components(G), key=len, reverse=True)
  G = G.subgraph(Gccs[0])
  mapping = {}
  for i, x in airports.iterrows():
    if G.has_node(x[4]):
      mapping[x[4]] = np.array([x[6], x[7]])

  nx.set_node_attributes(G, mapping, "location")

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_fungal(location='/data/mp2242/fungal_networks', fungus='Pv_M_I_U_N_42d_1.mat', relabel=True):
  mat = scipy.io.loadmat(os.path.join(location, fungus))
  G = nx.from_scipy_sparse_matrix(mat['A'], create_using=nx.Graph)
  mapping = {}

  for i in range(mat['coordinates'].shape[0]):
      mapping[i] = mat['coordinates'][i]

  nx.set_node_attributes(G, mapping, 'location')

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_ca_netscience(location='/data/mp2242/ca-netscience/ca-netscience.mtx', relabel=True):
  df = pd.read_csv(location, sep=' ', header=None, skiprows=2)
  G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)

  if relabel:
    G = nx.convert_node_labels_to_integers(G, label_attribute='name')

  return G

def load_ghtorrent_projects(location='/data/mp2242/ghtorrent-projects-hypergraph', simplex_min_size=2, simplex_max_size=2, relabel=True):
  df = pd.read_csv(os.path.join(location, 'project_members.txt'), sep='\t')
  df = df[(df.simplex_size >= simplex_min_size) & (df.simplex_size <= simplex_max_size)]

  H = Hypergraph()

  for i, row in df.iterrows():
    simplex = Simplex([int(x) for x in row['simplex'].split(',')], timestamp=row['created_at'])
    H.add_simplex(simplex)

  num_followers = pd.read_csv(os.path.join(location, 'num_followers.txt'), sep='\t')
  num_followers = num_followers[num_followers['user_id'].isin(list(H.nodes.keys()))]
  num_followers['log_num_followers'] = np.log(num_followers['num_followers'])

  for i, row in num_followers.iterrows():
    H.set_attribute(row['user_id'], 'log_num_followers', row['log_num_followers'])
    H.set_attribute(row['user_id'], 'num_followers', row['num_followers'])

  return H, num_followers


def load_hypergraph(name='email-Enron', location='/data/mp2242'):
  nverts = np.genfromtxt(os.path.join(location, name, '{}-nverts.txt'.format(name)), delimiter=',', dtype=np.int64)
  simplices = np.genfromtxt(os.path.join(location, name, '{}-simplices.txt'.format(name)), delimiter=',', dtype=np.int64)
  times = np.genfromtxt(os.path.join(location, name, '{}-times.txt'.format(name)), delimiter=',', dtype=np.int64)

  H = Hypergraph()
  j = 0

  for nvert, timestamp in zip(nverts, times):
    simplex = Simplex([], timestamp)
    for i in range(nvert):
      simplex.add_node(simplices[j])
      j += 1

    H.add_simplex(simplex)

  return H
