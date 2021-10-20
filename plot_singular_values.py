from base import *
from cigam import *
from utils import *
from dataloader import *

def get_argparser():
    parser = argparse.ArgumentParser(usage='Plot singular values between fitted network and real network')
    parser.add_argument('--name', help='Name of dataset to load', default='world-trade') 
    parser.add_argument('--field', default=None, help='Field to use for ranks (Latent inference if left empty)')
    parser.add_argument('--bayesian', action='store_true', help='Use bayesian inference')
    parser.add_argument('--num_layers', default=6, type=int, help='Number of layers to use')
    parser.add_argument('--simplex_size', default=2, type=int)
    parser.add_argument('--timestamp_min', default=-np.inf, type=float)
    parser.add_argument('--timestamp_max', default=np.inf, type=float)

    return parser.parse_args()

def plot_singular_values(G_list, ranks_list, labels_list, cigam, name):
    S_list = [np.log(np.linalg.svd(G.incidence_matrix())[1]) for G in G_list]

    plt.figure(figsize=(10, 10))
    plt.xlabel('Singular Value Numeric Rank')
    plt.ylabel('Log Singular Value')
    plt.title('Singular Values')

    colors = iter(cm.rainbow(np.linspace(0, 1, len(G_list))))

    for (log_S, ranks, label) in zip(S_list, ranks_list, labels_list):
        color = next(colors)
        num_ranks = 1 + np.arange(0, len(log_S))
        p = np.polyfit(num_ranks, log_S, deg=1)
        plt.plot(num_ranks, log_S, color=color, marker='x', linewidth=0, label='Singular Values ({})'.format(label))
        plt.plot(num_ranks, num_ranks * p[0] + p[1], color=color, label='Linear Fit, $R^2 = {}$ ({})'.format(round(np.corrcoef(num_ranks, log_S)[0, 1], 2), label))
    plt.legend()
    savefig('singular_values_numeric_rank_{}'.format(name))

if __name__ == '__main__':
    args = get_argparser()
    
    G, _ = load_dataset(name=args.name, simplex_min_size=args.simplex_size, simplex_max_size=args.simplex_size, timestamp_min=args.timestamp_min, timestamp_max=args.timestamp_max)
    G = G.deduplicate()

    if not(args.field is None):
        G, ranks = Hypergraph.convert_node_labels_to_integers_with_field(G, field=args.field)
        ranks = CIGAM.impute_ranks(ranks)
        ranks = normalize(ranks)
    else:
        G = Hypergraph.convert_node_labels_to_integers(G)
        ranks = None

    # Fit model
    H = np.linspace(0, 1, args.num_layers + 1)[1:] 
    c = 2.5 * np.ones(len(H))
    b = 3
    cigam = CIGAM(c=c, b=b, H=H, order=args.simplex_size)
    fit = cigam.fit_model_bayesian(G=G, ranks=ranks, H=H)

    # Keep mean ranks 
    if args.field is None:
        ranks = fit.extract()['ranks'].mean(0)

    # Generate artificial sample
    G_gen, ranks_gen = cigam.sample(N=len(G), return_ranks=True, method='naive')

    # Plot singular values
    plot_singular_values([G, G_gen], [ranks, ranks_gen], [args.name, 'artificial'], cigam, args.name)
