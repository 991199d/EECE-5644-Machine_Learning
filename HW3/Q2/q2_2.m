clf
Mu = [1 1; 1 4; 4 2; 6 0.5];
Sig(:, :, 1) = [1, 0; 0, 1];
Sig(:, :, 2) = [1.8, -1; -1, 0.8]; Sig(:, :, 3) = [0.4, 0.5; 0.5, 2]; Sig(:, :, 4) = [2, 0.75; 0.75, 1];
P = [0.35, 0.1, 0.3, 0.25];
gmm = gmdistribution(Mu, Sig, P);

%{From the matlab doccumentation:The software optimizes the Gaussian mixture model likelihood using the iterative Expectation-Maximization (EM) algorithm.fitgmdist fits GMMs to data using the iterative Expectation-Maximization (EM) algorithm. Using initial values for component means, covariance matrices, and mixing proportions, the EM algorithm proceeds using these steps.For each observation, the algorithm computes posterior probabilities of component memberships. You can think of the result as an n-by-k matrix, where element (i,j) contains the posterior probability that observation i is from component j. This is the E-step of the EM algorithm.Using the component-membership posterior probabilities as weights, the algorithm estimates the component means, covariance matrices, and mixing proportions by applying maximum likelihood. This is the M-step of the EM algorithm.The algorithm iterates over these steps until convergence. The likelihood surface is complex, and the algorithm might converge to a local optimum.fitgmdist() uses maximum likelihood parameter estimation using the EM algorithm}%





Ns = [10, 100, 1000, 10000]; eMat = zeros(6, length(Ns)); choice = eMat;
index = 1;

 for size = Ns
[X, Y] = random(gmm, size); comps = 1:6;
err = comps * 0;
for n = comps
newTest = @(a,b,c,d)(testQ1(a,b,c,d,n)); cvmdl = crossval(newTest, X, Y, 'KFold', 10); err(n) = sum(cvmdl)/10;
end
[~, mx] = min(err);
choice(:, index) = (comps == mx); eMat(:, index) = 10.^(err / max(err)); index = index + 1;
end
subplot(1, 2, 1);
bar3(eMat);
title("relative nLogL graph"); xlabel('number of samples 10^n'); zlabel('relative nlogL mean cross-value'); ylabel('number of gausian components');
subplot(1, 2, 2);
bar3(choice);
title("choice graph")
xlabel('number of samples 10^n');
zlabel('minimum value (chosen value)');
ylabel('number of gausian components');
