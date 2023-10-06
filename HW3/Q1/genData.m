function [x,labels,N_l,p_hat]= genData(N,p,mu,Sigma,Lx,d)
%Generates data and labels for random variable x from multiple gaussian
%distributions
numD = length(Lx);
cum_p = [0,cumsum(p)];
u = rand(1,N);
x = zeros(d,N);
labels = zeros(1,N);
for ind=1:numD
 pts = find(cum_p(ind)<u & u<=cum_p(ind+1));
 N_l(ind)=length(pts);
 x(:,pts) = mvnrnd(mu.(Lx{ind}),Sigma.(Lx{ind}),N_l(ind))';
 labels(pts)=ind-1;

 p_hat(ind)=N_l(ind)/N;
end
end

