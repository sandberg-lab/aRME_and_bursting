data {
    int<lower=0> y[3];
 }
parameters {
    simplex[3] theta;
}
transformed parameters{
    real allele1_normed;
    real allele2_normed;
    real allelic_imbalance;
    real allelic_imbalance_log10;
    allele1_normed = theta[1]/(1-theta[3]);
    allele2_normed = theta[2]/(1-theta[3]);
    allelic_imbalance = theta[1]/theta[2];
    allelic_imbalance_log10 = log(allelic_imbalance)/log(10);
}
model {
    y ~ multinomial(theta);
}
