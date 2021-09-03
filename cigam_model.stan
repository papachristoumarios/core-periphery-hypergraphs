functions {

  int[] get_layers(real[] ranks_vector, real[] thresholds, int len) {
      int layers[len];
      int j = 1;

      for (i in 1:len) {
          if H[len] - ranks_vector[i] > thresholds[j] {
              j = j + 1;
          }
          layers[i] = j
      }
      return layers;
  }

  int[][] get_num_layers(int[] layers, int len, int num_layers) {
    int temp[num_layers];
    int i = 1;

    for (l in 1:num_layers) {
      while (layers[i] == l) {
        temp[l] = temp[l] + 1;
        i = i + 1;
      }
    }

    int result[len, num_layers];
    int j = 1;

    for (i in 1:(len - 1)) {
      while (temp[j] == 0 && j <= num_layers) {
        j = j + 1;
      }
      temp[j] = temp[j] - 1;
    }

    return result;
  }

  int[][] get_partition_sizes(int[][] edges_vector, real[] ranks_vector, int[] ordering_vector, int[] layers_vector, real[] thresholds, int len, int num_layers, int num_edges, int simplex_size) {
    int sizes[len, num_layers];

    for (i in 1:len) {
      for (j in 1:num_layers) {
        sizes[i, j] = 0;
      }
    }

    int max_value;
    int argmax;
    int min_value;
    int argmin;

    for (i in 1:num_edges) {
      max_value = 0;
      argmax = -1;

      min_value = ranks_vector[len - 1];
      argmin = -1;


      for (k in 1:simplex_size) {
        if (ranks_vector[ordering_vector[edge_set[i, j] + 1]] >= max_value) {
          max_value = ranks_vector[ordering_vector[edges_vector[i, j] + 1]];
          argmax = k;
        }

        if (ranks_vector[ordering_vector[edge_set[i, j] + 1]] <= min_value) {
          min_value = ranks_vector[ordering_vector[edges_vector[i, j] + 1]];
          argmin = k;
        }

      }

      sizes[argmax, layers_vector[argmin]] = sizes[argmax, layers_vector[argmin]] + 1;

    }

    return sizes;
  }


  int[][] get_binomial_sizes(int[][] num_layers_vector, int[][] binomial_coefficients_vector, int len, int num_layers) {
    int binomial_sizes[len, num_layers];
    int j;

    for (i in 1:len) {
      j = i + 1;
      for l in 1:num_layers) {
          binomial_sizes[i, l] = binomial_coefficients_vector[j + num_layers_vector[i, l] - i, simplex_size - 1] - binomial_coefficients_vector[j - i - 1, order - 1];
          j = j + num_layers[i, l];
      }

    }

    return binomial_sizes;
  }


}

data {
  int N; // number of nodes
  int L; // number of layers
  int M; // number of edges
  int K; // hypergraph order
  int binomial_coefficients[N, K]; // precalculated binomial coefficients
  int edges[M, K]; // matrix with edge indices
  real H[L]; // multi-core thresholds
  real ranks[N]; // ranks
}

parameters {
  real<lower=1.003> c[L]; // bias bases
  real<lower=0> lambda; // ranks exponent
}

model {

  ranks ~ exponential(lambda);
  int ordering[N]; // ordering of ranks
  ordering = sort_indices_desc(ranks);  // argsort
  print("Ordering");
  print(ordering);

  ranks = sort_desc(ranks); // sort
  print("Sorted ranks");
  print(ranks);

  int layers[N];
  layers = get_layers(ranks, H, N); // create layers
  print("Layers");
  print(layers);

  int num_layres[N, L];
  num_layers = get_num_layers(layers, N, L);
  print("Num layers");
  print(num_layers);

  int sizes[N, L];
  sizes = get_partition_sizes(edges, ranks, ordering, layers, H, N, L, M, K)
  print("Sizes");
  print(sizes);

  int binomial_sizes[N, L];
  binomial_sizes = get_neg_partition_sizes(sizes, num_layers, binomial_coefficients, N, L);
  print("Binomial sizes");
  print(binomial_sizes);

  // Sample hypergraph
  for (i in 1:N) {
    for (l in 1:L) {
      sizes[i, l] ~ binomial(binomial_sizes[i, l], pow(c[l], -1 - H + ranks[i]));
    }
  }
}
