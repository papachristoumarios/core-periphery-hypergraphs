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

