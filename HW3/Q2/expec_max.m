function[mu, cov, a] = expec_max(C,iter, X_train)
%% Parameter InitializationC 
a = ones(1,C).*1/C;
cov(:,:,C) = zeros(2,2);
mu = zeros(C,2);
for i = 1:C
cov(:,:,i) = [1,0;0,1];
mu_y_init = (max(X_train(:,2))+min(X_train(:,2)))/2;
mu_x_init = i*max(X_train(:,1))/(C+1)+(C+1-i)*min(X_train(:,1))/(C+1);

 mu(i,:) = [mu_x_init,mu_y_init]; 
end
w = zeros(size(X_train,1),length(a)); %%w(i,j),i is sample num, j is cluster's num，w(i,j) is the probability of sample i belongs to cluster j, max of w is i's cluster
%% EM Implementation 
Q = zeros(iter,1);
for i = 1:iter
%% Expectaion: 
for j = 1 : length(a)
w(:,j)=a(j)*mvnpdf(X_train,mu(j,:),cov(:,:,j)+.001 * eye(2)); 
end
w=w./repmat(sum(w,2),1,size(w,2));
%% Maximum:
a = sum(w,1)./size(w,1);
mu = w'*X_train;
mu= mu./repmat((sum(w,1))',1,size(mu,2));
itersum = zeros(size(X_train,1),1); 
for j = 1 : length(a)
vari = repmat(w(:,j),1,size(X_train,2)).*(X_train- repmat(mu(j,:),size(X_train,1),1)); cov(:,:,j) = (vari'*vari)/sum(w(:,j),1);
itersum = itersum + a(j).*mvnpdf(X_train,mu(j,:),cov(:,:,j)+.001 * eye(2));
end
itersum = log(itersum); Q(i) = sum(itersum,1);
fprintf("Iteration %d, Qi= %5f \n", i,Q(i)); 
if (i>1 && abs(Q(i) - Q(i-1))<0.01 || i==iter)
fprintf("Break at Iteration %d, Qi= %5f \n", i,Q(i));
return; 
end
end
end
