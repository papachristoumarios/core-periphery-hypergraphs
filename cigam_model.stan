	lambda ~ gamma(alpha_lambda, beta_lambda);
	//ranks ~ exponential(lambda) T[,H[L]];
  c0 ~ exponential(alpha_c); 
  
  // [SAMPLE RANKS]

  for (i in 1:N) {
    for (l in 1:L) {
        sizes[i, l] ~ binomial_real(binomial_sizes[i, l], pow(c[l], -1 - H[L] + sorted_ranks[i]));
      }
  }
