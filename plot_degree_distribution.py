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

    return parser.parse_args()

if __name__ == '__main__':

    args = get_argparser()

    layers_range = np.arange(args.layers_step, args.num_layers + 1, args.layers_step) 
    simplex_size_range = np.arange(args.simplex_min_size, args.simplex_max_size + 1, 1)

    plt.figure(figsize=(10, 10))
    plt.ylabel('(Log) Normalized Degree') 
    plt.xlabel('(Log) Numeric Rank') 
    plt.title('Degree Distributions')

    binomial_coeffs = binomial_coefficients(args.n, args.simplex_max_size).astype(np.float64)

    for l in layers_range:
        H = (2.0)**(-np.arange(l, dtype=np.float64))[::-1]
        c = np.linspace(1.5, 3 - 0.1, len(H))
        
        for k in simplex_size_range:
            cigam = CIGAM(c=c, H=H, b=3, order_min=k, order_max=k, constrained=True)

            G, ranks = cigam.sample(N=args.n, return_ranks=True, method='naive')
            #degrees = G.degrees()
            #degrees = degrees / binomial_coeffs[args.n, k - 1]
            #log_degrees = np.log(degrees + 1)
            #log_degrees = -np.sort(-log_degrees) 
            #log_num_ranks = np.log(1 + np.arange(len(degrees)))

            #_, (px, py) = segments_fit(log_num_ranks, log_degrees, count=l)
            
            #plt.plot(log_num_ranks, log_degrees, linewidth=0, marker='x', label='k = {}, L = {} (Sample)'.format(k, l))
            #plt.plot(px, py, linewidth=4, marker='o', label='k = {}, L = {} (Fit)'.format(k, l))
            freqs, bins = G.degrees_histogram(log=False)
            
            bins = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
            
            plt.plot(bins, freqs, marker='o')

    plt.yscale('log') 
    plt.xscale('log')
    
    plt.legend()
    savefig('degree_distributions_{}_{}_{}_{}'.format(args.simplex_min_size, args.simplex_max_size, args.num_layers, args.layers_step))
