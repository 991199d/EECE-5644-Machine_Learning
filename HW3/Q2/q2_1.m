close all; clear; clc;
N=10;
d=2;%experiments = 30;
%num_GMM_picks = zeros(length(N) ,6) ;
iter = 300;
delta = 1e0; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
a_real =[0.3,0.3,0.2,0.2]; 
mu_real = [8,4;6,12;12,5;13,14]; 
cov_real(:,:,1) = [4 1;1 20]; 
cov_real(:,:,2) = [6 1;1 3]; 
cov_real(:,:,3) = [7 1;1 4]; 
cov_real(:,:,4) = [5 1;1 20];
mu_true=[8 4 6 12;12 5 13 14];
thr = [0,cumsum(a_real)];
x = randGMM(N,a_real,mu_true,cov_real);
u = rand(N,1);
sample_num = zeros(1,length(a_real)); 
label = zeros(N,1);
X = zeros(N,2);
for l = 1:length(a_real)
indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample willget omitted - needs to be fixed
sample_num(l) = length(indices);
fprintf('sample %d number: %d \n',l, sample_num(l)); label(indices,1) = l*ones(length(indices),1);
disp(['mu_real: ' num2str(mu_real(l,:))]);
fprintf('cov_real: \n');
disp(cov_real(:,:,l));
fprintf('alpha_real: %5f \n\n',a_real(l));
X(indices,:) = mvnrnd(mu_real(l,:),cov_real(:,:,l),length(indices));
end
%% Sample Ploting
figure(1); plotclass(X,4,label,a_real,mu_real,cov_real);
 sample_mark = zeros(10,1); %% 10-fold-cv
for C=4:4
for t = 0:9
X_train = X;
X_test = X((N*t/10)+1:N*(t+1)/10,:); X_train((N*t/10)+1:N*(t+1)/10,:) = [];
label_train = label;
label_test = label((N*t/10)+1:N*(t+1)/10,:); label_train((N*t/10)+1:N*(t+1)/10,:) = [];
%% EM
[mu, cov, a] = expec_max(C,iter, X_train);
pdf_all = zeros(length(X_test),C);
for i = 1:C
pdf_all(:,i) = log(mvnpdf(X_test,mu(i,:),cov(:,:,i))*a(i));
end
[pdfvalue, estimate] = max(pdf_all,[],2);
sample_mark(t+1) = sum(pdfvalue);
%% EM-Plot
% figure(t+2);
% plotclass(X_test,C,estimate,a,mu,cov); end
cv10 = mean(sample_mark);
fprintf("Component %d, MSE= %5f \n",C, cv10);
end
end
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

        [alpha,mu0,Sigma] = EMforGMM(Ntrain,xTrain,M,d,delta,regWeight);% determine dimensionality of samples and number of GMM components
        p = zeros(1,Nvalidate);
        for j = 1:Nvalidate
            for i = 1:M
                p(j) = p(j) + alpha(i)*evalGaussian(xValidate(:,j),mu0(:,i),Sigma(:,:,i));
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
%figure(3),bar(avgp),set(gca,'yscale','log');
%legend( '10', '100' ,'1000','10000') ;
%title( 'GMM Model Order Selection' ) ;
%xlabel ( 'GMM Model Order') ; ylabel('logp' ) ;






















 