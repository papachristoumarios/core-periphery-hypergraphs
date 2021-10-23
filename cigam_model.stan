model {
	int ordering[N]; // ordering of ranks
	real sorted_ranks[N]; // sorted ranks
	int layers[N];
	int num_layers[N, L];
	real sizes[N, L];
	real binomial_sizes[N, L]; 
  int ordered_edges[max(M), K_max];
  int j;

	lambda ~ gamma(1, 2);
	ranks ~ exponential(lambda);
	ordering = sort_indices_desc(ranks);  // argsort
	// print("Ordering");
	// print(ordering);

	sorted_ranks = sort_desc(ranks); // sort
	// print("Sorted ranks");
	// print(sorted_ranks);


  layers = get_layers(sorted_ranks, H, N, L); // create layers
  // print("Layers");
  // print(layers);
  
  num_layers = get_num_layers(layers, N, L);
  // print("Num layers");
  // print(num_layers);
  
  for (k in K_min:K_max) {
    j = k - K_min + 1;
  
    for (i in 1:N) {
      for (l in 1:L) {
        sizes[i, l] = 0;
        binomial_sizes[i, l] = 0;
      }
    }

    // TODO initialize arrays
    ordered_edges  = order_edges(edges[j, 1:M[j], 1:k], ordering, M[j], k);
    sizes = get_partition_sizes(ordered_edges, sorted_ranks,  layers, H, N, L, M[j], k);
    // print("Sizes");
    // print(sizes);

    binomial_sizes = get_binomial_sizes(num_layers, binomial_coefficients, N, L, k);
    // print("Binomial sizes");
    // print(binomial_sizes);

    // Sample hypergraph
    c0 ~ exponential(1); 
    for (i in 1:N) {
      for (l in 1:L) {
          sizes[i, l] ~ binomial_real(binomial_sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
        }
    }
  }
}
