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

		print("Temp ", temp);

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


	int[,] get_partition_sizes(int[,] edges_vector, real[] ranks_vector, int[] ordering_vector, int[] layers_vector, real[] H_vector, int N_size, int L_size, int M_size, int K_size) {
		int sizes[N_size, L_size];
		int j;
		real min_value;
		real max_value;
		int argmin;
		int argmax;

		for (i in 1:N_size) {
			for (l in 1:L_size) {
				sizes[i, l] = 0;
			}
		}

		for (m in 1:M_size) {
			argmin = -1;
			argmax = -1;

			for (k in 1:K_size) {

				if (argmin == -1 || ranks_vector[ordering_vector[edges_vector[m, k] + 1]] <= min_value) {
					argmin = edges_vector[m, k] + 1;
					min_value = ranks_vector[ordering_vector[argmin]];
				}

				if (argmax == -1 || ranks_vector[ordering_vector[edges_vector[m, k] + 1]] >= max_value) {
					argmax = edges_vector[m, k] + 1;
					max_value = ranks_vector[ordering_vector[argmax]];
				}
			}

			sizes[argmax, layers_vector[argmin]] += 1;

		}

		return sizes;
	}

	int[,] get_binomial_sizes(int[,] num_layers_vector, int[,] binomial_coefficients_vector, int N_size, int L_size, int K_size) {
		int binomial_sizes[N_size, L_size];
		int j;

		for (i in 1:N_size) {
			for (l in 1:L_size) {
				binomial_sizes[i, l] = 0;
			}
		}

		for (i in 1:(N_size - 1)) {
			j = i + 1;
			for (l in 1:L_size) {
				binomial_sizes[i, l] = binomial_coefficients_vector[j + num_layers_vector[i, l] - i + 1, K_size - 1 + 1] - binomial_coefficients_vector[j - i - 1 + 1, K_size - 1 + 1];
				j += num_layers_vector[i, l];
			}
		}

		return binomial_sizes;
	}

}
