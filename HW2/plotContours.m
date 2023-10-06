function plotContours(x,alpha,mu,Sigma)
figure
if size(x,1)==2
	plot(x(1,:),x(2,:),'b.');
	xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
	axis equal, hold on;
	rangex1 = [min(x(1,:)),max(x(1,:))];
	rangex2 = [min(x(2,:)),max(x(2,:))];
	[x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
	contour(x1Grid,x2Grid,zGMM); axis equal,
end
end