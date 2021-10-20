from base import *
from hypergraph import *
from utils import *

def remove_self_loops(df, source, dest):
    return df[df[source] != df[dest]]

def load_dataset(name, **kwargs):
    simplex_min_size = kwargs.get('simplex_min_size', 2)
    simplex_max_size = kwargs.get('simplex_max_size', 25)
    timestamp_min = kwargs.get('timestamp_min', -np.inf)
    timestamp_max = kwargs.get('timestamp_max', np.inf)

    if name == 'world-trade':
        return load_world_trade()
    elif name == 'polblogs':
        return load_polblogs()
    elif name == 'ca-netscience':
        return load_ca_netscience()
    elif name in  ['cs-faculty', 'business-faculty', 'history-faculty']:
        if name == 'cs-faculty':
            return load_faculty(location='/data/mp2242/faculty/ComputerScience_edgelist.txt')
        elif name == 'history-faculty':
            return load_faculty(location='/data/mp2242/faculty/History_edgelist.txt')
        elif name == 'business-faculty':
            return load_faculty(location='/data/mp2242/faculty/Business_edgelist.txt')
    elif name == 'celegans':
        return load_celegans()
    elif name == 'open-airlines':
        return load_open_airlines()
    elif name == 'pvc-Enron':
        return load_pvc_Enron()
    elif name == 'pvc-text-Reality':
        return load_pvc_text_Reality()
    elif name == 'coauth-MAG-KDD':
        return load_coauth_mag_kdd(simplex_min_size=simplex_min_size, simplex_max_size=simplex_max_size, timestamp_min=timestamp_min, timestamp_max=timestamp_max, completed=True)
    elif name in ['congress-bills', 'contact-high-school', 'contact-primary-school', 'email-Eu', 'email-Enron']:
        return load_hypergraph(name=name, simplex_min_size=simplex_min_size, simplex_max_size=simplex_max_size, timestamp_min=timestamp_min, timestamp_max=timestamp_max)
    elif name == 'ghtorrent':
        return load_ghtorrent_projects(simplex_min_size=simplex_min_size, simplex_max_size=simplex_max_size, timestamp_min=timestamp_min, timestamp_max=timestamp_max, completed=True)


def load_world_trade(location='/data/mp2242/world-trade/world-trade.csv', relabel=True):
    df = pd.read_csv(location)
    df = remove_self_loops(df, 'from', 'to')
    G = nx.convert_matrix.from_pandas_edgelist(df, source='from', target='to')
    labels = {}

    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u, data in G.nodes(data=True):
            labels[u] = data['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_faculty(location='/data/mp2242/faculty/ComputerScience_edgelist.txt', relabel=True):
    df = pd.read_csv(location, sep='\t')
    df = remove_self_loops(df, '# u', 'v')
    G = nx.convert_matrix.from_pandas_edgelist(df, source='# u', target='v')
    vertexlist_filename = location.replace('edgelist', 'vertexlist')
    vertex_df = pd.read_csv(vertexlist_filename, sep='\t')
    vertex_df.set_index('# u', inplace=True)
    mapping = vertex_df['institution'].to_dict()
    nx.set_node_attributes(G, mapping, 'name')
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u, data in G.nodes(data=True):
            labels[u] = data['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_polblogs(location='/data/mp2242/polblogs/polblogs.mtx', relabel=True):
    df = pd.read_csv(location, sep=' ', comment='%', header=None)
    G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1)
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_airports(location='/data/mp2242/airports/USairport500.txt', relabel=True):
    df = pd.read_csv(location, sep=' ', header=None)
    G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1)
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_pvc_Enron(location='/data/mp2242/pvc-enron/pvc-enron.csv', relabel=True):
    df = pd.read_csv(location, sep=' ', header=None)
    G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_pvc_text_Reality(location='/data/mp2242/pvc-text-Reality/pvc-text-Reality.csv', relabel=True):
    df = pd.read_csv(location, sep=' ', header=None)
    G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

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

    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u, data in G.nodes(data=True):
            labels[u] = data['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

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

    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

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

    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_fungal(location='/data/mp2242/fungal_networks', fungus='Pv_M_I_U_N_42d_1.mat', relabel=True):
    mat = scipy.io.loadmat(os.path.join(location, fungus))
    G = nx.from_scipy_sparse_matrix(mat['A'], create_using=nx.Graph)
    mapping = {}

    for i in range(mat['coordinates'].shape[0]):
            mapping[i] = mat['coordinates'][i]

    nx.set_node_attributes(G, mapping, 'location')

    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_ca_netscience(location='/data/mp2242/ca-netscience/ca-netscience.mtx', relabel=True):
    df = pd.read_csv(location, sep=' ', header=None, skiprows=2)
    G = nx.convert_matrix.from_pandas_edgelist(df, source=0, target=1, create_using=nx.Graph)
    labels = {}
    
    if relabel:
        G = nx.convert_node_labels_to_integers(G, label_attribute='name')
        for u in G.nodes():
            labels[u] = G[u]['name']
    else:
        for u in G.nodes():
            labels[u] = u

    return Hypergraph.graph_to_hypergraph(G), labels

def load_ghtorrent_projects(location='/data/mp2242/ghtorrent-projects-hypergraph', simplex_min_size=2, simplex_max_size=2, relabel=True):
    df = pd.read_csv(os.path.join(location, 'project_members.txt'), sep='\t')
    df = df[(df.simplex_size >= simplex_min_size) & (df.simplex_size <= simplex_max_size)]

    H = Hypergraph()

    for i, row in df.iterrows():
            simplex = Simplex([int(x) for x in row['simplex'].split(',')], simplex_data={'timestamp' :row['created_at']})
            H.add_simplex(simplex)

    num_followers = pd.read_csv(os.path.join(location, 'num_followers.txt'), sep='\t')
    num_followers = num_followers[num_followers['user_id'].isin(list(H.nodes()))]
    num_followers['log_num_followers'] = np.log(num_followers['num_followers'])

    for i, row in num_followers.iterrows():
        H.set_attribute(row['user_id'], 'log_num_followers', row['log_num_followers'])
        H.set_attribute(row['user_id'], 'num_followers', row['num_followers'])

    return H, num_followers

def load_bills(name='house-bills', location='/data/mp2242', simplex_min_size=2, simplex_max_size=2):

    H = Hypergraph()

    with open(os.path.join(location, name, 'hyperedges-{}.txt'.format(name))) as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.split(', ') 
        
        simplex = Simplex([], simplex_data = {})

        for v in line:
            simplex.add_node(int(v))
            
        if simplex_min_size <= len(line) <= simplex_max_size:
            H.add_simplex(v)

    labels = {}

    with open(os.path.join(location, name, 'node-labels-{}.txt'.format(name))) as f1:
        with open(os.path.join(location, name, 'node-names-{}.txt'.format(name))) as f2:

            node_labels = f1.read().splitlines()
            node_names = f2.read().splitlines()

            labels = dict([(i, (int(node_label), node_name)) for i, (node_label, node_name) in enumerate(zip(node_labels, node_names))])

    return H, labels

def load_hypergraph(name='email-Enron', location='/data/mp2242', simplex_min_size=2, simplex_max_size=2, timestamp_min=-np.inf, timestamp_max=np.inf, load_features=True):
    nverts = np.genfromtxt(os.path.join(location, name, '{}-nverts.txt'.format(name)), delimiter=',', dtype=np.int64)
    simplices = np.genfromtxt(os.path.join(location, name, '{}-simplices.txt'.format(name)), delimiter=',', dtype=np.int64)
    times = np.genfromtxt(os.path.join(location, name, '{}-times.txt'.format(name)), delimiter=',', dtype=np.int64)
    
    if load_features:
        features = pd.read_csv(os.path.join(location, name, '{}-features.txt'.format(name)), delimiter=' ', header=None).set_index(0)
        features = features.to_dict()


    H = Hypergraph()
    j = 0

    for nvert, timestamp in zip(nverts, times):
        simplex = Simplex([], simplex_data={'timestamp' : timestamp})
        for i in range(nvert):
            simplex.add_node(simplices[j])
            j += 1
        if timestamp_min <= timestamp <= timestamp_max and simplex_min_size <= nvert <= simplex_max_size:
            H.add_simplex(simplex)


    try:
        labels = {}
         
        with open(os.path.join(location, name, '{}-node-labels.txt'.format(name))) as f:
            lines = f.read().splitlines()

            for line in lines:
                line = line.split(' ')
                labels[int(line[0])] = ' '.join(line[1:])
    except:
        labels = {}
        for u in H.nodes():
            labels[u] = u

    for key, val in features.items():
        H.set_attribute(key, 'features', val)

    return H, labels

def load_coauth_mag_kdd(location='/data/mp2242/coauth-MAG-KDD', simplex_min_size=2, simplex_max_size=2, timestamp_min=1970, timestamp_max=2100, completed=True):
        H = Hypergraph()

        with open(os.path.join(location, 'coauth-MAG-KDD.txt')) as f:
                lines = f.read().splitlines()
        with open(os.path.join(location, 'coauth-MAG-KDD-years.txt')) as f:
                years = f.read().splitlines()

        for line, year in zip(lines, years):
                line = line.split(' ')
                if simplex_min_size <= len(line) <= simplex_max_size and timestamp_min <= int(year) <= timestamp_max:
                        simplex = Simplex(line)
                        H.add_simplex(simplex)

        stats = pd.read_csv(os.path.join(location, 'coauth-MAG-KDD-node-labels{}.txt'.format('-imputed' if completed else '')), sep='\t')
        # stats = normalize_df(stats, fields=['h_index', 'n_citation', 'n_pubs'])

        for i, row in stats.iterrows():
                H.set_attribute(row['aminer_id'], 'h_index', row['h_index'])
                H.set_attribute(row['aminer_id'], 'n_citation', row['n_citation'])
                H.set_attribute(row['aminer_id'], 'n_pubs', row['n_pubs'])

        return H, stats
