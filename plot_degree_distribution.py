from base import *
from cigam import *
from hypergraph import *

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simplex_min_size', default=2, type=int)
    parser.add_argument('--simplex_max_size', default=3, type=int)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--layers_step', default=2, type=int)
    parser.add_argument('-n', default=1000, type=int)
    parser.add_argument('--numeric_rank', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_argparser()

    layers_range = np.arange(args.layers_step, args.num_layers + 1, args.layers_step) 
    simplex_size_range = np.arange(args.simplex_min_size, args.simplex_max_size + 1, 1)

    plt.figure(figsize=(10, 10))
    plt.ylabel('(Log) Normalized Degree', fontsize=16) 
    if args.numeric_rank:
        plt.xlabel('(Log) Numeric Rank', fontsize=16) 
    else:
        plt.xlabel('(Log) Rank', fontsize=16)
        
    plt.title('Degree Distributions', fontsize=16)

    binomial_coeffs = binomial_coefficients(args.n, args.simplex_max_size).astype(np.float64)

    for l in layers_range:
        H = (2.0)**(-np.arange(l, dtype=np.float64))[::-1]
        c = np.linspace(1.5, 3 - 0.1, len(H))
        
        for k in simplex_size_range:
            cigam = CIGAM(c=c, H=H, b=3, order_min=k, order_max=k, constrained=True)

            print(k, l)
            G, ranks = cigam.sample(N=args.n, return_ranks=True, method='naive')
            degrees = G.degrees()
            degrees = degrees / binomial_coeffs[args.n, k - 1]
            log_degrees = np.log(degrees + 1)
            if args.numeric_rank:
                log_degrees = -np.sort(-log_degrees) 
                log_num_ranks = np.log(1 + np.arange(len(degrees)))

                error, (px, py) = segments_fit(log_num_ranks, log_degrees, count=l)        
                plt.plot(log_num_ranks, log_degrees, linewidth=0, marker='x', label='k = {}, L = {} (Sample)'.format(k, l))
            else:
                log_ranks = np.log(ranks)
                error, (px, py) = segments_fit(log_ranks, log_degrees, count=l)
                plt.plot(log_ranks, log_degrees, linewidth=0, marker='x', label='k = {}, L = {} (Sample)'.format(k, l))

            plt.plot(px, py, linewidth=4, marker='o', label='k = {}, L = {} (Fit), log(error) = {}'.format(k, l, round(np.log(error), 1)))

    plt.legend(fontsize=14)
    savefig('degree_distributions_{}_{}_{}_{}'.format(args.simplex_min_size, args.simplex_max_size, args.num_layers, args.layers_step))
