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
}
