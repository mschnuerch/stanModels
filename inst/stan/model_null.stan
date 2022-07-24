data {
  int<lower=2> K;
  int<lower=0> y1[K];
  int<lower=0> y2[K];
  real<lower=0> b1;
}

parameters {
  ordered[K-1] alpha;
}

transformed parameters {
  vector[K] lambda;
  lambda[1] = Phi_approx(alpha[1]);
  for (k in 2:(K-1)){
    lambda[k] = Phi_approx(alpha[k]) - Phi_approx(alpha[k-1]);
  }
  lambda[K] = 1 - Phi_approx(alpha[K-1]);
}

model {
  target += normal_lpdf(alpha | 0, b1);
  target += multinomial_lpmf(y1 | lambda);
  target += multinomial_lpmf(y2 | lambda);
}
