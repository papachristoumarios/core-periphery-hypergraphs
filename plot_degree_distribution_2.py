from base import *
from utils import * 
from hypergraph import *
from cigam import *

def get_argparser():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--simplex_min_size', default=2, type=int)
    parser.add_argument('--simplex_max_size', default=2, type=int)
    parser.add_argument('--layers_step', default=2, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--c_start', default=1.5, type=float)
    parser.add_argument('--c_end', default=2.5, type=float)
    parser.add_argument('-b', default=3, type=float)
    return parser.parse_args()

if __name__ == '__main__': 

    args = get_argparser() 

    simplex_size_range = np.arange(args.simplex_min_size, args.simplex_max_size + 1)
    layers_range = np.arange(args.layers_step, args.num_layers + 1, args.layers_step) 

    r_axis = np.linspace(1e-1, 1, 100)

    plt.figure(figsize=(10, 10))

    for k in simplex_size_range:
        for l in layers_range:
            # H = (args.b)**(-np.arange(l, dtype=np.float64))[::-1]
            H = np.linspace(0.1, 1, 9)

            c = np.linspace(args.c_start, args.c_end, len(H))

            cigam = CIGAM(c=c, b=args.b, H=H, order_min=k, order_max=k, constrained=True)
            
            s = np.zeros((k, args.num_samples))

            for j in range(k):
                s[j, :] = cigam.continuous_tree_sample(N=args.num_samples)

            s_max = s.max(0)
            s_min = s.min(0)
            f = np.zeros_like(r_axis)

            for i, r in enumerate(r_axis):
                for j in range(args.num_samples):
                    ll = CIGAM.get_layers(np.array([min(r, s_min[j])]), H)[0]
                    f[i] += c[ll]**(-2+max(s_max[j], r))

                f[i] /= args.num_samples

            plt.plot(r_axis, f, label='$k = {}, L = {}$'.format(k, l))
    plt.xlabel('$r$', fontsize=16)
    plt.ylabel('$\\hat d$', fontsize=16)
    plt.legend(fontsize=14)
    plt.title('Normalized Degree', fontsize=16)
    plt.savefig('foo.png')

            


