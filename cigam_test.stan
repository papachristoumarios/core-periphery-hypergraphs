functions {

	int[] get_layers(real[] ranks_vector, real[] thresholds, int N_size, int L_size) {
		int layers[N_size];
		int j = 1;

		for (i in 1:N_size) {
			if (thresholds[L_size] - ranks_vector[i] > thresholds[j]) {
				j = j + 1;
			}
			layers[i] = j;
		}
		return layers;
	}

	int[,] get_num_layers(int[] layers, int N_size, int L_size) {
		int temp[L_size];
		int i = 1;
		int j = 1;
		int result[N_size, L_size];
		
		for (l in 1:L_size) {
			temp[l] = 0;
			for (i1 in 1:N_size) {
				result[i1, l] = 0;
			} 
		}

		for (l in 1:L_size) {
			while (i <= N_size && layers[i] == l) {
				temp[l] = temp[l] + 1;
				i = i + 1;
			}
		}

		// print("Temp ", temp);

		for (i1 in 1:(N_size - 1)) {
			while (temp[j] == 0 && j <= L_size) {
				j = j + 1;
			}
			temp[j] = temp[j] - 1;
			// for (l in 1:L_size) {
			//	result[i1, l] = temp[l];
			// }
			result[i1, 1:L_size] = temp;
		}

		return result;
	}

data {
	int N; // number of nodes
	int L; // number of layers
	int M; // number of edges
	int K; // hypergraph order
	int binomial_coefficients[N + 1, K + 1]; // precalculated binomial coefficients
	int edges[M, K]; // matrix with edge indices
	real H[L]; // multi-core thresholds
	real ranks[N]; // ranks
}

parameters {
	real<lower=1.003> c[L]; // bias bases
	real<lower=0, upper=H[L]> lambda; // ranks exponent
}

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
