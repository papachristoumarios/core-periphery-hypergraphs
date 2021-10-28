from base import *
from cigam import *
from hypergraph import *
from dataloader import *
from utils import *

def get_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='coauth-MAG-KDD', help='Dataset to load')
    parser.add_argument('-r', default='known', choices=['latent', 'known'], help='Rank variables are either latent or known')
    parser.add_argument('-f', default='n_citation', help='Field to use for rank variables when known')
    parser.add_argument('-o', default='layers_temporal', help='Outfile name')
    parser.add_argument('-l', default=6, type=int, help='Number of layers') 
    parser.add_argument('--step', default=1, help='Timestamp increment')
    parser.add_argument('--window', default=5, help='Sliding Window')
    parser.add_argument('--pieces', default=-1, help='Number of pieces to split timestamps')
    parser.add_argument('--bayesian', action='store_true', help='Use Bayesian inference')
    parser.add_argument('--order_min', default=-np.inf)
    parser.add_argumetn('--order_max', default=np.inf) 
    return parser.parse_args()

def visualize_parameter_layer_evolution(fits, Hs, timestamps, title='', outfile='layers_temporal'): 

    plt.figure(figsize=(10, 10))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(fits))))
    
    for i, (fit, H, timestamp) in enumerate(zip(fits, Hs, timestamps)):
        color = next(colors)
        means = fit['c'].mean(1)
        plt.plot(np.log(H), np.log(means), color=color, marker='x', label='{} - {}'.format(timestamp, timestamp + 5))
    
    plt.xlabel('Log Thresholds $H_i$', fontsize=16)
    plt.ylabel('Log Mean Parameter $c_i$', fontsize=16)
    plt.title(title)
    plt.legend(fontsize=14)

    savefig(outfile)

if __name__ == '__main__':

    args = get_argparser()

    num_layers = args.l

    # Get timestamp range
    G, _ = load_dataset(args.name, timestamp_min=-np.inf, timestamp_max=np.inf)
    timestamp_min, timestamp_max = G.get_field_range(field='timestamp')
        
    
    assert(args.pieces == -1 ^ args.step == -1)

    if args.step > 0:
        timestamps = list(range(timestamp_min, timestamp_max, args.step))
    elif args.pieces == -1:
        timestamps = np.linspace(timestamp_min, timestamp_max, args.pieces, dtype=np.int64)


    fits = []
    Hs = []
    for timestamp in timestamps:
        G, stats = load_coauth_mag_kdd(simplex_min_size=args.order_min, simplex_max_size=args.order_max, timestamp_min=timestamp, timestamp_max=timestamp + args.window_size)

        if len(G) == 0:
            continue

        print('Year = {}, n = {}'.format(timestamp, len(G)))

        G, ranks = Hypergraph.convert_node_labels_to_integers_with_field(G, field=args.f)
        k_min, k_max = G.get_order_range()

        H = np.linspace(0, 1, num_layers + 1)[1:]
        cigam = CIGAM(order_min=k_min, order_max=k_max, H=H)
        ranks = CIGAM.impute_ranks(ranks)
        ranks = normalize(ranks)
        
        # Call fitting
        if args.r == 'known':
            fit = cigam.fit_model_bayesian(G=G, ranks=ranks, H=H)
        elif args.r == 'latent':
            fit = cigam.fit_model_bayesian(G=G, ranks=None, H=H)

        fits.append(fit)
        Hs.append(H)
    
    visualize_parameter_layer_evolution(fits, Hs, timestamps, title='{}: Parameter Evolution (log-log plot) for {} layers'.format(args.name, num_layers), outfile=args.o)
