from base import *
from cigam import * 
from utils import *
from hypergraph import * 

def special_binom(x, y):
    return scipy.special.gamma(x + 1) / (scipy.special.gamma(y + 1) * scipy.special.gamma(x - y + 1))

def CDF(b, t):
    lambda_ = np.log(b)
    return (1 - np.exp(-lambda_ * t)) / (1 - np.exp(-lambda_))

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('-b', default=3, type=float)
    parser.add_argument('--c_min', default=1.5, type=float) 
    parser.add_argument('--c_max', default=2.5, type=float)
    parser.add_argument('--order_min', default=2, type=int)
    parser.add_argument('--order_max', default=4, type=int)
    parser.add_argument('-n', default=100, type=int)
    parser.add_argument('--log_log', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_argparser()
    plt.figure(figsize=(10, 10))
    plt.xlabel('Percentage of nodes used', fontsize=16)
    plt.ylabel('Percentage of nodes dominated', fontsize=16)

    for order in range(args.order_min, args.order_max + 1):
        c = np.linspace(args.c_min, args.c_max, args.num_layers)
        H = np.linspace(0, 1, args.num_layers + 1)[1:]

        cigam = CIGAM(c=c, b=args.b, H=H, order_min=order, order_max=order)

        G, ranks = cigam.sample(args.n, method='naive')
        ordering = np.arange(len(ranks))
        
        x_axis, y_axis = G.domination_curve(ordering) 
        emp_threshold_index = np.where(y_axis == y_axis.max())[0][0]
        emp_x_threshold = x_axis[emp_threshold_index]
        emp_y_threshold = y_axis[emp_threshold_index]
        emp_t_threshold = ranks[emp_threshold_index]
        plt.scatter(x=emp_x_threshold, y=emp_y_threshold, marker='x', label='$t = {}$ (k = {})'.format(round(float(emp_t_threshold), 3), order), color='r')  
        plt.plot(x_axis, y_axis, label='k = {}'.format(order), linewidth=2, alpha=0.6)
        # plt.scatter(x=th_x_threshold, y=th_y_threshold, marker='o', label='Theoretical $t =  {}$ (k = {})'.format(round(th_x_threshold, 2), order), color='k') 
    if args.log_log:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend(fontsize=16) 
    plt.title('Domination Curve', fontsize=16)

    plt.savefig('domination_curve_synthetic.png')
