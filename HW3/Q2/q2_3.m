close all,
N = 10;
delta = 1e0; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 3-component GMM
alpha_true = [0.2,0.3,0.4,0.1];
mu_true = [-7 7 7 -7;7 7 -7 -7];
mu_real= [-7,7;7,-7;7,7;-7,-7];
Sigma_true(:,:,1) = [20 1;10 3];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 10;1 16];
Sigma_true(:,:,4) = [2 1;1 7];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
figure(1);
figure(1),scatter(x(1,:),x(2,:),'ob'), hold on,
figure(1),legend('sample')
d = 2;
K = 10;
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

avgp = zeros(1,6);
for M = 1:6
    psum = zeros(1,10);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [[1:indPartitionLimits(k-1,2)],[indPartitionLimits(k+1,1):N]];
        end
        xTrain = x(:,indTrain); % using all other folds as training set
        Ntrain = length(indTrain); Nvalidate = length(indValidate);

        [alpha,mu,Sigma] = EMforGMM(Ntrain,xTrain,M,d,delta,regWeight);% determine dimensionality of samples and number of GMM components
        p = zeros(1,Nvalidate);
        for j = 1:Nvalidate
            for i = 1:M
                p(j) = p(j) + alpha(i)*evalGaussian(xValidate(:,j),mu(:,i),Sigma(:,:,i));
            end
            p(j) = log(p(j));
        end
        psum(k) = sum(p);
        
    end
    avgp(M) = sum(psum)/10;
    if (avgp(M)== -inf)
        avgp(M) = -1e5;
    end
end

figure(2),scatter([1,2,3,4,5,6],avgp),set(gca,'yscale','log'),
figure(2),legend('order'),title('Orders log-liklihood '),
xlabel('order'), ylabel('logp')




function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end