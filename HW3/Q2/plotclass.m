function [] = plotclass(X_all,classnum,label,alpha,mu,cov)
x = linspace(0,20); y = linspace(0,20); [x, y]=meshgrid(x,y); mesh = [x(:),y(:)];
s=('rbcgykm');
u=('do*x+vp');
subplot(2,1,1);
legend_label = strings(1,classnum); z_val = zeros(length(mesh),1);
for i = 1:classnum
X_loop = X_all(label==i,:); plot(X_loop(:,1),X_loop(:,2),[s(i) u(i)]);
hold on;
legend_label(i) = join(['X_' num2str(i)]);
z_val = z_val + alpha(i)*mvnpdf(mesh,mu(i,:),cov(:,:,i));
end
legend(legend_label); title('GMM of 4 component'); xlabel('x1');
ylabel('x2');
hold off;
subplot(2,1,2); surf(x,y,reshape(z_val,size(x,2),size(y,2))); title('surface plot of 4 component'); xlabel('x1');
ylabel('x2');
zlabel('log-likelihood function');
end
