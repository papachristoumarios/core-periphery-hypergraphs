from base import *
from logistic_cp import *
from hypergraph import *
from dataloader import *
from utils import *
from cigam import * 

def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='n_citation', choices=['n_citation', 'h_index'], help='Field to use for rank variables when known')
    parser.add_argument('-o', default='layers_temporal', help='Outfile name')
    parser.add_argument('--order_min', default=2, type=int)
    parser.add_argument('--order_max', default=10, type=int)
    parser.add_argument('--timestamp_min', default=1995, type=int)
    parser.add_argument('--timestamp_max', default=2014, type=int)
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--num_simulations', default=6, type=int)
    parser.add_argument('--negative_samples', default=100, type=float)
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--top_k_step', default=0.25, type=float, help='Step for top-k')
    parser.add_argument('--name', default='congress-bills')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_argparser()

    timestamps = np.array(list(range(args.timestamp_min, args.timestamp_max + 1, args.step)))
    orders = np.array(list(range(args.order_min, args.order_max + 1)))
    num_simulations = args.num_simulations
    
    means = np.zeros((len(orders), len(timestamps)))
    stds = np.zeros_like(means)
    medians_means = np.zeros_like(means)
    medians_stds = np.zeros_like(means)
    sizes = np.zeros_like(means)

    core_means = np.zeros_like(means)
    core_stds = np.zeros_like(means)

    top_k_means = np.zeros((len(orders), len(timestamps), args.top_k))
    top_k_stds = np.zeros_like(top_k_means)

    for i1, i in enumerate(orders):
        for timestamp1, timestamp in enumerate(timestamps):
            if args.name == 'coauth-MAG-KDD':
                G, stats = load_coauth_mag_kdd(simplex_min_size=i, simplex_max_size=i, timestamp_min=timestamp, timestamp_max=timestamp + args.window_size)
            else:
                G, labels = load_hypergraph(name=args.name, simplex_min_size=i, simplex_max_size=i, timestamp_min=timestamp, timestamp_max=timestamp + args.window_size)
            
            if len(G) == 0:
                print('Skip Timestamps {} - {}'.format(timestamp, timestamp + args.window_size))
nohup python3.6 inference_temporal_logistic_cp.py --timestamp_min 0 --timestamp_max 19 --window 0 --step 1 --order_min 2 --order_max 25 --negative_samples 0.5  --num_simulations 10 --name email-Eu &                continue

            G = G.deduplicate()
            G = Hypergraph.convert_node_labels_to_integers(G)

            print('Timestamps = {} - {}, n = {}, m = {}, k = {}'.format(timestamp, timestamp + args.window_size, len(G), G.num_simplices(), i))
            logistic_cp = LogisticCP(order=i, thetas=np.zeros(len(G)))
            
            thetas = np.zeros((num_simulations, len(G)))

            for j in range(num_simulations):
                fit = logistic_cp.fit(G=G, negative_samples=int(args.negative_samples * len(G)) if 0 <= args.negative_samples <= 1 else int(args.negative_samples))
                thetas[j, :] = fit[-1]

            sizes[i1, timestamp1] = G.num_simplices()
            means[i1, timestamp1]  = thetas.mean(0).mean()
            stds[i1, timestamp1] = thetas.std(0).std()
            medians_means[i1, timestamp1] = np.median(thetas, axis=-1).mean()
            medians_stds[i1, timestamp1] = np.median(thetas, axis=-1).std()
            core_means[i1, timestamp1] = thetas[thetas >= 0].mean(0).mean()
            core_stds[i1, timestamp1] = thetas[thetas >= 0].std(0).std()

            runs_means = thetas.mean(0)
            indexes = np.argsort(-runs_means)

            for k in range(args.top_k):
                top_k_means[i1, timestamp1, k] = runs_means[indexes[:int((k + 1) * args.top_k_step * len(G))]].mean()
                top_k_stds[i1, timestamp1, k] = runs_means[indexes[:int((k + 1) * args.top_k_step * len(G))]].std()

    sizes_sum = sizes.sum(0) 
    means_agg = np.zeros(len(timestamps))
    stds_agg = np.zeros_like(means_agg)

    medians_means_agg = np.zeros(len(timestamps))
    medians_stds_agg = np.zeros_like(means_agg)

    core_means_agg = np.zeros_like(means_agg)
    core_stds_agg = np.zeros_like(means_agg)
    # periphery_means_agg = np.zeros_like(means_agg) 

    top_k_means_agg = np.zeros((len(timestamps), args.top_k))
    top_k_stds_agg = np.zeros_like(top_k_means_agg)

    for timestamp in range(len(timestamps)):
        means_agg[timestamp] = (sizes[:, timestamp] * means[:, timestamp]).sum() / sizes_sum[timestamp]
        stds_agg[timestamp] = (sizes[:, timestamp] * stds[:, timestamp]).sum() / sizes_sum[timestamp]
        medians_means_agg[timestamp] = (sizes[:, timestamp] * medians_means[:, timestamp]).sum() / sizes_sum[timestamp]
        medians_stds_agg[timestamp] = (sizes[:, timestamp] * medians_stds[:, timestamp]).sum() / sizes_sum[timestamp]
        core_means_agg[timestamp] = (sizes[:, timestamp] * core_means[:, timestamp]).sum() / sizes_sum[timestamp]
        core_stds_agg[timestamp] = (sizes[:, timestamp] * core_stds[:, timestamp]).sum() / sizes_sum[timestamp]
        # periphery_means_agg[timestamp] = (sizes[:, timestamp] * core_means[:, timestamp]).sum() / sizes_sum[timestamp]
        
        for k in range(args.top_k):
            top_k_means_agg[timestamp, k] = (sizes[:, timestamp] * top_k_means[:, timestamp, k]).sum() / sizes_sum[timestamp]
            top_k_stds_agg[timestamp, k] = (sizes[:, timestamp] * top_k_stds[:, timestamp, k]).sum() / sizes_sum[timestamp]

    plt.figure(figsize=(10, 10))
    nanplot(timestamps, means_agg, label='Mean core score', color='b', marker='x')
    plt.fill_between(timestamps, means_agg - stds_agg, means_agg + stds_agg, color='b', alpha=0.3) 
    # nanplot(timestamps, medians_means_agg, label='Median core score (all nodes)', color='r', marker='x')
    # plt.fill_between(timestamps, medians_means_agg - medians_stds_agg, medians_means_agg + medians_stds_agg, color='r', alpha=0.3) 

    # nanplot(timestamps, core_means_agg, marker='o', color='k', label='Mean core scores (nodes with $\\theta_u > 0$)')
    # plt.fill_between(timestamps, core_means_agg - core_stds_agg, core_means_agg + core_stds_agg, color='k', alpha=0.3)

    colors = iter(cm.rainbow(np.linspace(0, 1, args.top_k)))
    
    for k in range(args.top_k):
        color = next(colors)
        nanplot(timestamps, top_k_means_agg[:, k], color=color, label='Top {}% of core nodes'.format(int(k + 1) * args.top_k_step * 100), marker='x')
        plt.fill_between(timestamps, top_k_means_agg[:, k] - top_k_stds_agg[:, k], top_k_means_agg[:, k] + top_k_stds_agg[:, k], color=color, alpha=0.3)

    plt.ylabel('Core scores') 
    plt.xlabel('Timestamp')

    plt.title(args.name)

    plt.legend()


    savefig(args.o)

