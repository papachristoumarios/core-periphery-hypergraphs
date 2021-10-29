	lambda ~ gamma(1, 2);
	ranks ~ exponential(lambda);
  for (k in K_min:K_max) {
    c0 ~ exponential(1); 
    for (i in 1:N) {
      for (l in 1:L) {
          sizes[k - K_min + 1, i, l] ~ binomial_real(binomial_sizes[k - K_min + 1, i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
        }
    }
  }
