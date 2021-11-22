	int ordering[N]; // ordering of ranks
	vector[N] sorted_ranks; // sorted ranks
	int layers[N];
	int num_layers[N, L];
	matrix[N, L] sizes;
	matrix[N, L] binomial_sizes; 
  int ordered_edges[max(M), K_max];
  int j;

  // [SAMPLE RANKS]

	ordering = sort_indices_desc(ranks);  // argsort
	sorted_ranks = sort_desc(ranks); // sort

  layers = get_layers(sorted_ranks, H, N, L); // create layers
  
  num_layers = get_num_layers(layers, N, L);
  
  for (i in 1:N) {
    for (l in 1:L) {
      sizes[i, l] = 0;
      binomial_sizes[i, l] = 0;
    }
  }
  
  for (k in K_min:K_max) {
    j = k - K_min + 1;
  
    ordered_edges = order_edges(edges[j, 1:M[j], 1:k], ordering, M[j], k);
    sizes = sizes + get_partition_sizes(ordered_edges[1:M[j], 1:k], sorted_ranks,  layers, H, N, L, M[j], k);
    binomial_sizes= binomial_sizes + get_binomial_sizes(num_layers, binomial_coefficients, N, L, k);
  }
