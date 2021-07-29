from base import *
from cigam import * 
from hypergraph import *
from utils import *
from collections import defaultdict
from datetime import datetime

def sample_helper(args):
    cigam, n = args
    start = datetime.now()
    H, h = cigam.sample(n, method='ball_dropping')
    end = datetime.now()
    eta_ball_dropping = (end - start).microseconds
    return eta_ball_dropping

if __name__ == '__main__':
    orders = [2, 3, 4]
    sizes = [10, 50, 100, 500, 1000]

    n_trials = 50
    results_mean = collections.defaultdict(list)
    results_std = collections.defaultdict(list)
    mean_edges = collections.defaultdict(list)

    n_jobs = 2
    pool = multiprocessing.Pool(n_jobs)
        
    for order in orders:
        cigam = CIGAM(order=order) 
        for n in sizes:
            start = datetime.now()
            # H, h = cigam.sample(n, method='naive')
            end = datetime.now()
            eta_naive = (end - start).microseconds

            P, _ = cigam.bias_matrix(n)
            
            mean_edges[order].append(P.sum())
            
            trials = pool.map(sample_helper, [(cigam, n) for _ in range(n_trials)])
            results_mean[order].append(np.mean(trials))
            results_std[order].append(np.std(trials))

    pool.close()

    ax = plt.figure(figsize=(10, 10))

    cmap = plt.get_cmap('viridis')

    print(results_mean)
    print(results_std)

    for key, val in results_mean.items():
        plt.plot(sizes, np.log(val), color=cmap(key), marker='x', label='Average Runtime (k = {})'.format(key))
        plt.fill_between(sizes, np.log(np.array(val) - np.array(results_std[key])), np.log(np.array(val) + np.array(results_std[key])), color=cmap(key), alpha=0.3) 
        mean_edges_temp = np.array(mean_edges[key])
        plt.plot(sizes, np.log(mean_edges_temp * np.log(mean_edges_temp)), linestyle='dashed', color=cmap(key), label='Theoretical Runtime (k = {})'.format(key))


    plt.title('Ball Dropping Performance')
    plt.xlabel('Network Size ($n$)')
    plt.ylabel('Runtime (us) (log-scale)')
    plt.legend()
    ax.set_rasterized(True)
    plt.savefig('eta_sampling.eps')

