	lambda ~ gamma(1, 2);
	ranks ~ exponential(lambda);
  c0 ~ exponential(1); 
  for (i in 1:N) {
    for (l in 1:L) {
        sizes[i, l] ~ binomial_real(binomial_sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
      }
  }
