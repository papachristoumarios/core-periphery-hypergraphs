model {
	int ordering[N]; // ordering of ranks
	real sorted_ranks[N]; // sorted ranks
	int layers[N];
	int num_layers[N, L];
	int sizes[N, L];
	int binomial_sizes[N, L]; 

	ranks ~ exponential(lambda);
	ordering = sort_indices_desc(ranks);  // argsort
	// print("Ordering");
	// print(ordering);

	sorted_ranks = sort_desc(ranks); // sort
	// print("Sorted ranks");
	// print(ranks);

	layers = get_layers(sorted_ranks, H, N, L); // create layers
	// print("Layers");
	// print(layers);

	num_layers = get_num_layers(layers, N, L);
	// print("Num layers");
	// print(num_layers);

	sizes = get_partition_sizes(edges, sorted_ranks, ordering, layers, H, N, L, M, K);
	// print("Sizes");
	// print(sizes);

	binomial_sizes = get_binomial_sizes(num_layers, binomial_coefficients, N, L, K);
	// print("Binomial sizes");
	// print(binomial_sizes);

	// Sample hypergraph
	for (i in 1:N) {
		for (l in 1:L) {
			sizes[i, l] ~ binomial(binomial_sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
		}
	}
}
