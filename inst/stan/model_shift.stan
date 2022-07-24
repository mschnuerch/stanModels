data {
  int<lower=2> K;
  int<lower=0> y1[K];
  int<lower=0> y2[K];
  real<lower=0> b1;
  real<lower=0> b2;
}

parameters {
  ordered[K-1] alpha;
  real theta;
}

transformed parameters {
  ordered[K-1] gamma1;
  ordered[K-1] gamma2;
  vector[K] lambda1;
  vector[K] lambda2;
   
  for (k in 1:(K-1)){
    gamma2[k] = alpha[k] - .5*theta;
    gamma1[k] = alpha[k] + .5*theta;
  }
      
  lambda1[1] = Phi_approx(gamma1[1]);
  lambda2[1] = Phi_approx(gamma2[1]);
  for (k in 2:(K-1)){
    lambda1[k] = Phi_approx(gamma1[k]) - Phi_approx(gamma1[k-1]);
    lambda2[k] = Phi_approx(gamma2[k]) - Phi_approx(gamma2[k-1]);
  }
  lambda1[K] = 1 - Phi_approx(gamma1[K-1]);
  lambda2[K] = 1 - Phi_approx(gamma2[K-1]);
}

model {
  target += normal_lpdf(alpha | 0, b1);
  target += normal_lpdf(theta | 0, b2);
  target += multinomial_lpmf(y1 | lambda1);
  target += multinomial_lpmf(y2 | lambda2);
}
