model {
	int ordering[N]; // ordering of ranks
	real sorted_ranks[N]; // sorted ranks
	int layers[N];
	int num_layers[N, L];
	int sizes[N, L];
	int binomial_sizes[N, L]; 
  int ordered_edges[M, K];

	lambda ~ gamma(1, 2);
	ranks ~ exponential(lambda);
	ordering = sort_indices_desc(ranks);  // argsort
	// print("Ordering");
	// print(ordering);

	sorted_ranks = sort_desc(ranks); // sort
	// print("Sorted ranks");
	// print(sorted_ranks);

  ordered_edges = order_edges(edges, ordering, M, K);

	layers = get_layers(sorted_ranks, H, N, L); // create layers
	// print("Layers");
	// print(layers);

	num_layers = get_num_layers(layers, N, L);
	// print("Num layers");
	// print(num_layers);

	sizes = get_partition_sizes(ordered_edges, sorted_ranks,  layers, H, N, L, M, K);
	// print("Sizes");
	// print(sizes);

	binomial_sizes = get_binomial_sizes(num_layers, binomial_coefficients, N, L, K);
	// print("Binomial sizes");
	// print(binomial_sizes);

	// Sample hypergraph
	c0 ~ pareto(0.5, 2); 
	for (i in 1:N) {
		for (l in 1:L) {
			if (sizes[i, l] > binomial_sizes[i, l]) {
        // print("Difference", sizes[i, l] - binomial_sizes[i, l]);
        sizes[i, l] ~ binomial(sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
      }
      else {
        sizes[i, l] ~ binomial(binomial_sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
		  }
      }
	}
}
