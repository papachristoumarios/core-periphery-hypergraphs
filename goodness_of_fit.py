from hypergraph import *
from dataloader import *
from utils import *
from cigam import *
from logistic_th import *
from logistic_cp import *

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--alpha_c', default=0, type=float)
    parser.add_argument('--alpha_lambda', default=1, type=float)
    parser.add_argument('--beta_lambda', default=2, type=float)
    parser.add_argument('--learnable_ranks', action='store_true', help='Use NN to learn ranks')
    parser.add_argument('--centrality_features', action='store_true')
    parser.add_argument('--combine_centrality_features', action='store_true')
    parser.add_argument('--ranks_col', default=0, help='Column to use for ranks')
    parser.add_argument('--scatter_plot', action='store_true')
    parser.add_argument('--domination_curve', action='store_true')
    parser.add_argument('-H', default='', type=str)
    parser.add_argument('--core_profile', action='store_true', help='Plot core profile')
    parser.add_argument('--name', default='coauth-MAG-KDD', help='Dataset to load')
    parser.add_argument('-r', default='known', choices=['latent', 'known', 'degree', 'pagerank', 'eigenvector'], help='Rank variables are either latent or known')
    parser.add_argument('-f', default='features', help='Field to use for rank variables when known')
    parser.add_argument('-o', default='layers_orders', help='Outfile name')
    parser.add_argument('-l', default=6, type=int, help='Number of layers') 
    parser.add_argument('--step', default=1, type=int, help='Timestamp increment')
    parser.add_argument('--window', default=np.inf, type=int, help='Sliding Window')
    parser.add_argument('--pieces', default=-1, type=int, help='Number of pieces to split timestamps')
    parser.add_argument('--bayesian', action='store_true', help='Use Bayesian inference')
    parser.add_argument('--order_min', default=2, type=int)
    parser.add_argument('--order_max', default=2, type=int) 
    parser.add_argument('--timestamp_min', default=-np.inf, type=int)
    parser.add_argument('--timestamp_max', default=np.inf, type=int)
    parser.add_argument('--pipeline', default='cigam,logistic-cp,logistic-th', type=str)
    parser.add_argument('--unconstrained', action='store_true', help='Uncostrained CIGAM')
    parser.add_argument('--grid_search_step', default=0.5, type=float)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--clique_decomposition', action='store_true') 
    parser.add_argument('--k_core', default=0, type=int)
    parser.add_argument('--degree_threshold', default=0, type=int)
    parser.add_argument('--no_lcc', action='store_true')
    parser.add_argument('--exclude_ranks_col', action='store_true')

    parser.add_argument('--negative_samples', default='0.2N', type=str)
    parser.add_argument('--logistic_cp_lr', default=1e-6, type=float)
    parser.add_argument('--logistic_cp_alpha_theta', default=0, type=float)

    parser.add_argument('--logistic_th_p', default=20, type=float)
    parser.add_argument('--logistic_th_no_eval_log_likelihood', action='store_true', help='Skip LL evaluation for Logistic-TH')

    parser.add_argument('--cigam_lr', default=1e-3, type=float)
    parser.add_argument('--cigam_smoothing', default=1e-10, type=float)
    parser.add_argument('--cigam_early_stopping', default='log-posterior', type=str) 

    parser.add_argument('--rank_correlation_plots', action='store_true')    

    return parser.parse_args()

def pprint_number(x, bf):
    if abs(x) <= 1:
        return '${}{{{}}}$'.format('\\mathbf' if bf else '', round(x, 2))
    else:
        x = int(x)
        e = len(str(abs(x)))
        x_p = round(x / 10**(e - 1), 1)
        return '{}{{{}}} {{{}}}'.format('\\mixed' if not bf else '\\mixedbf', x_p, e + 1)

def pprint_table(objective, spearman):
    max_objective = max(objective)
    max_spearman = max(spearman)
    result = ''
    for obj, sp in zip(objective, spearman):
        result = result + '{} / {} & '.format(pprint_number(obj, obj==max_objective), pprint_number(sp, sp==max_spearman))

    return result

if __name__ == '__main__':
    args = get_argparser()
    print_args(args)

    torch.cuda.set_device(args.cuda)

    num_layers = args.l
    pipeline = args.pipeline.split(',')

    objective = []
    spearman = []

    if args.H == '':
        H = []
    else:
        H = [float(x) for x in args.H.split(',')]

    # Get timestamp range
    G, _ = load_dataset(args.name, timestamp_min=args.timestamp_min, timestamp_max=args.timestamp_max)
    order_min, order_max = G.get_order_range()

    order_min = max(order_min, args.order_min)
    order_max = min(order_max, args.order_max)

    if args.step > 0:
        order_range = range(order_min, order_max + 1, args.step)
    elif args.pieces > 0:
        order_range = np.linspace(order_min, order_max, args.pieces, dtype=int)

    fits = []
    Hs = []
    for order in order_range:
        G, _ = load_dataset(args.name, timestamp_min=args.timestamp_min, timestamp_max=args.timestamp_max, simplex_min_size=order, simplex_max_size=order + args.window)

        G = G.deduplicate()

        if args.degree_threshold > 0:
            G = G.filter_degrees(threshold=args.degree_threshold)
        
        if not args.no_lcc:
            G = G.largest_connected_component()
        if args.k_core > 0:
            G = G.k_core(k=args.k_core)

        if args.clique_decomposition:
            G = G.clique_decomposition(dtype=Hypergraph)
        
        k_min, k_max = G.get_order_range()

        if len(G) == 0:
            continue

        print('Order Range = {} - {}, n = {}, m = {}'.format(k_min, k_max, len(G), G.num_simplices(separate=False)))
        G, features = Hypergraph.convert_node_labels_to_integers_with_field(G, field=args.f, sort_col=args.ranks_col)

        if args.r == 'known':
            ranks = features[:, args.ranks_col]
        elif args.r == 'degree':
            ranks = np.log(1 + np.atleast_2d([G.degree(u) for u in range(len(G))]).T)
        elif args.r == 'pagerank':
            pagerank = G.pagerank()
            ranks = np.log(1 + np.atleast_2d([pagerank[u] for u in range(len(G))]).T)
        elif args.r == 'eigenvector':
            eigenvector = G.clique_graph_eigenvector()
            ranks = np.log(1 + np.atleast_2d([eigenvector[u] for u in range(len(G))]).T)

        ranks = normalize(ranks, method='minmax')
        ranks = CIGAM.impute_ranks(ranks)
           

        if args.exclude_ranks_col:
            features = np.delete(features, args.ranks_col, -1)

        features = np.nan_to_num(features)
        
        if args.centrality_features:
            if args.combine_centrality_features:
                features = np.hstack((features, G.centrality_features()))
            else:
                features = G.centrality_features()

        features = normalize_array(features, 'z-score')

        for algorithm in pipeline:
            print('Algorithm', algorithm)
            if algorithm == 'cigam':  
                cigam = CIGAM(order_min=k_min, order_max=k_max, constrained=(not args.unconstrained), alpha_lambda=args.alpha_lambda, beta_lambda=args.beta_lambda, alpha_c=args.alpha_c)


                if args.H == '':
                    if args.learnable_ranks:
                        cigam.find_hyperparameters(G, features=features, gold_ranks=ranks, learnable_ranks=args.learnable_ranks, layers_max=5, step=args.grid_search_step, bayesian=args.bayesian, num_epochs=args.num_epochs, criterion=args.cigam_early_stopping)
                    else:
                        cigam.find_hyperparameters(G, features=features, gold_ranks=ranks, learnable_ranks=args.learnable_ranks, layers_max=5, step=args.grid_search_step, bayesian=args.bayesian, num_epochs=args.num_epochs, criterion=args.cigam_early_stopping)
                else:
                    cigam.H = H

                # Call fitting
                if args.r != 'latent':
                    if args.bayesian:
                        fit = cigam.fit_model_bayesian(G=G, ranks=ranks, H=cigam.H)
                        print('LL CIGAM: ', fit['lp__'].max())
                    else:
                        if args.learnable_ranks:
                            lp_cigam, spearman_cigam, learned_cigam = cigam.fit_model_given_ranks_torch(G, features=features, gold_ranks=ranks, H=cigam.H, learnable_ranks=args.learnable_ranks, num_epochs=args.num_epochs, early_stopping=args.cigam_early_stopping, ranks_col=args.ranks_col, lr=args.cigam_lr, smoothing=args.cigam_smoothing)
                            plt.figure(figsize=(10, 10))
                            plt.plot(learned_cigam)
                            plt.title(args.name)
                            plt.plot(np.sort(learned_cigam))
                            plt.xlabel('Ordering') 
                            plt.ylabel('1D Learned Representation')
                            plt.savefig('learned_cigam_ranks_{}.png'.format(args.name))
                        else:
                            lp_cigam, spearman_cigam, learned_cigam = cigam.fit_model_given_ranks_torch(G, features=features, gold_ranks=ranks, H=cigam.H, learnable_ranks=args.learnable_ranks, num_epochs=args.num_epochs, early_stopping=args.cigam_early_stopping, ranks_col=args.ranks_col, lr=args.cigam_lr, smoothing=args.cigam_smoothing)
                        print('LL CIGAM: ', lp_cigam, 'Spearman:' , spearman_cigam, 'H:', cigam.H)
                elif args.r == 'latent':
                    fit = cigam.fit_model_bayesian(G=G, ranks=None, H=cigam.H)
                    print('LL CIGAM: ', fit['lp__'].max(), 'H:', cigam.H)

                print('CIGAM Core profile: ', cigam.c)
                print('CIGAM Lambda: ', cigam.lambda_) 
                objective.append(lp_cigam)
           
            elif algorithm == 'logistic-th':
                if k_min == 2 and k_max == 2:
                    logistic_th = LogisticTH(order_min=k_min, order_max=k_max)
                else:
                    logistic_th = HyperNSM(order_min=k_min, order_max=k_max)
                lp_logistic_th, learned_logistic_th, _ = logistic_th.fit(G, p=args.logistic_th_p, negative_samples=int(G.num_simplices(negate=args.negative_samples[-1] == 'N') * float(args.negative_samples[:-1])), eval_log_likelihood=not args.logistic_th_no_eval_log_likelihood)

                print('LL Logistic-TH:', lp_logistic_th)
                
                objective.append(lp_logistic_th)
            elif algorithm == 'logistic-cp':
                logistic_cp = LogisticCP(thetas=np.zeros(len(G)), order_min=k_min, order_max=k_max, alpha_theta=args.logistic_cp_alpha_theta)
                if args.learnable_ranks:
                    lp_logistic_cp, learned_logistic_cp = logistic_cp.fit_torch(G, gold_ranks=ranks, features=features, negative_samples=int(G.num_simplices(negate=args.negative_samples[-1] == 'N') * float(args.negative_samples[:-1])), num_epochs=args.num_epochs, lr=args.logistic_cp_lr)
                else:
                    lp_logistic_cp, learned_logistic_cp  = logistic_cp.fit_torch(G, gold_ranks=ranks, features=features, lr=args.logistic_cp_lr, negative_samples=int(G.num_simplices(negate=args.negative_samples[-1] == 'N') * float(args.negative_samples[:-1])), num_epochs=args.num_epochs)

                learned_logistic_cp = normalize(learned_logistic_cp)

                print('LL Logistic-CP:', lp_logistic_cp)
        
                objective.append(lp_logistic_cp)
        if args.scatter_plot and args.learnable_ranks:
            if 'cigam' in pipeline and 'logistic-cp' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Logistic-CP Rank Embeddings', fontsize=16)
                plt.ylabel('CIGAM Rank Embeddings', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_cp[:, 0], learned_cigam[:, 0])[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_logistic_cp, y=learned_cigam)
                plt.savefig('{}_scatter_logistic_cp_cigam.png'.format(args.name))
            
            if 'cigam' in pipeline and 'logistic-th' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Logistic-CP Rank Embeddings', fontsize=16)
                plt.ylabel('CIGAM Rank Embeddings', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_th, learned_cigam[:, 0])[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_logistic_th, y=learned_cigam)
                plt.savefig('{}_scatter_logistic_th_cigam.png'.format(args.name))
            
            if 'cigam' in pipeline and 'logistic-th' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Logistic-CP Rank Embeddings', fontsize=16)
                plt.ylabel('Logistic-TH Rank Embeddings', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_cp[:, 0], learned_logistic_th)[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_logistic_cp, y=learned_logistic_th)
                plt.savefig('{}_scatter_logistic_cp_logistic_th.png'.format(args.name))

            if 'logistic-cp' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Learned Rank', fontsize=16)
                plt.ylabel('True Rank', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_cp[:, 0], ranks)[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_logistic_cp, y=ranks)
                plt.savefig('{}_scatter_logistic_cp_real.png'.format(args.name))
            
            if 'cigam' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Learned Rank', fontsize=16)
                plt.ylabel('True Rank', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_cp[:, 0], ranks)[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_cigam, y=ranks)
                plt.savefig('{}_scatter_cigam_real.png'.format(args.name))

            if 'logistic-th' in pipeline:
                plt.figure(figsize=(10, 10))
                plt.xlabel('Learned Rank', fontsize=16)
                plt.ylabel('True Rank', fontsize=16)
                plt.title('{}, $R^2 = {}$'.format(args.name, round(np.corrcoef(learned_logistic_th, ranks)[0, 1], 2)), fontsize=16)
                sns.regplot(x=learned_logistic_th, y=ranks)
                plt.savefig('{}_scatter_cigam_real.png'.format(args.name))


    inferred_ranks = {
        'true': ranks
    }

    centrality_features = G.centrality_features()
    
    if 'cigam' in pipeline:
        inferred_ranks['cigam'] = learned_cigam[:, 0]
    if 'logistic-th' in pipeline:
        inferred_ranks['logistic-th'] = learned_logistic_th
    if 'logistic-cp' in pipeline:
        inferred_ranks['logistic-cp'] = learned_logistic_cp
    
    inferred_ranks['degree'] = centrality_features[:, 0]
    inferred_ranks['eigenvector'] = centrality_features[:, 1]
    inferred_ranks['pagerank'] = centrality_features[:, 2]

    spearmanr = {}

    for i, a in enumerate(pipeline + ['true', 'degree', 'eigenvector', 'pagerank']):
        for j, b in enumerate(pipeline + ['true', 'degree', 'eigenvector', 'pagerank']):
            spearmanr[a, b] = scipy.stats.spearmanr(inferred_ranks[a], inferred_ranks[b])
            print('Spearman rho: {} / {} : {} (p = {})'.format(a, b, round(spearmanr[a, b].correlation, 2), round(spearmanr[a, b].pvalue, 3)))
            spearman.append(spearmanr[a, b].correlation)

    orderings = {}
    # orderings['random'] = np.random.permutation(ranks.shape[0])

    for algorithm in pipeline + ['true', 'degree', 'pagerank', 'eigenvector']:
        if algorithm == 'true':
            orderings[algorithm] = np.argsort(-ranks)
        elif algorithm == 'degree':
            orderings[algorithm] = np.argsort(-centrality_features[:, 0])
        elif algorithm == 'eigenvector':
            orderings[algorithm] = np.argsort(-centrality_features[:, 1])
        elif algorithm == 'pagerank':
            orderings[algorithm] = np.argsort(-centrality_features[:, 2])
        elif algorithm == 'cigam':
            orderings[algorithm] = np.argsort(-learned_cigam[:, 0])
        elif algorithm == 'logistic-cp':
            orderings[algorithm] = np.argsort(-learned_logistic_cp[:, 0])
        elif algorithm == 'logistic-th':
            orderings[algorithm] = np.argsort(-learned_logistic_th)

    if args.domination_curve:
        plt.figure(figsize=(10, 10))
        plt.title('Domination Curve for {}'.format(args.name), fontsize=16)
        plt.xlabel('Percentage of nodes included', fontsize=16) 
        plt.ylabel('Percentage of nodes dominated', fontsize=16)
        for algorithm, ordering in orderings.items():
            x_axis, y_axis = G.domination_curve(ordering)
            plt.plot(x_axis, y_axis, label=algorithm)

        plt.legend(fontsize=14)
        plt.savefig('{}_domination_curve_{}.png'.format(args.name, '_'.join(pipeline)))
    
    if args.core_profile:
        plt.figure(figsize=(10, 10))
        plt.title('Core Profile Score: {}'.format(args.name.rstrip('-filtered')), fontsize=16)
        plt.xlabel('Node $i$', fontsize=16)
        plt.ylabel('$\\gamma(S_i)$', fontsize=16)
        for algorithm, ordering in orderings.items():
            if algorithm in ['true', 'cigam', 'logistic-th', 'logistic-cp']:
                x_axis, y_axis = G.core_profile(ordering[::-1])
                plt.plot(x_axis, y_axis, label=algorithm if algorithm != 'logistic-th' else 'hyper-nsm', linewidth=4)
        plt.xlim(x_axis.min(), x_axis.max())
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=14)
        plt.savefig('{}_core_profile_degree_threshold_{}_k_core_{}_{}.png'.format(args.name, args.degree_threshold, args.k_core, '_'.join(pipeline)))
    
    orderings = pd.DataFrame.from_dict(orderings)

    if args.learnable_ranks:
        fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
        for a in pipeline:
            for i, b in enumerate(['true', 'degree', 'pagerank', 'eigenvector']):
                sns.regplot(data=orderings, x=a, y=b, label='{} ($R^2$ = {}, p = {})'.format(a, round(spearmanr[a, b].correlation, 2), round(spearmanr[a, b].pvalue, 4)), scatter_kws={'s' : 2}, ax=ax[i])

        plt.suptitle(args.name.rstrip('-filtered'))
        for i, b in enumerate(['true', 'degree', 'pagerank', 'eigenvector']):
            ax[i].set_xlabel('algorithm')
            ax[i].set_ylabel(b)
            ax[i].legend()
        
        plt.savefig('regplot_centrality_degree_threshold_{}_k_core_{}_{}'.format(args.degree_threshold, args.k_core, args.name)) 

        plt.figure(figsize=(10, 10))
    
        for i, a in enumerate(pipeline):
            for j, b in enumerate(pipeline):
                if i < j:
                    sns.regplot(data=orderings, x=a, y=b, label='{}/{} ($R^2 = {}$)'.format(a, b, round(spearmanr[a, b].correlation, 2)), scatter_kws={'s' : 2})
        
        plt.xlabel('ordering #1', fontsize=20)
        plt.ylabel('ordering #2', fontsize=20)
        plt.legend(fontsize=16)
        plt.title(args.name.rstrip('-filtered') + (' (projected)' if args.clique_decomposition else ''), fontsize=20)
        
        plt.savefig('regplot_algorithms_degree_threshold_{}_k_core_{}_{}'.format(args.degree_threshold, args.k_core, args.name)) 
