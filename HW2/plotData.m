function plotData(x, decisions, labels, theta, type, graphTitle)
 global C x1grid x2grid shapes x1gridMatrix x2gridMatrix mu sigma
 figure
 % Plot every point colored according to correct or incorrect
 for i = 1:C %each decision
 for j = 1:C %each class label
 if i == j
 plot(x(1,decisions==(i-1) & labels==(j-1)),x(2,decisions==(i-1) & labels==(j-1)),append('g',shapes(j)),'DisplayName', ['Class ' num2str(j-1) ' Correct Classification']);
 hold on
 else
 plot(x(1,decisions==(i-1) & labels==(j-1)),x(2,decisions==(i-1) & labels==(j-1)),append('r',shapes(j)),'DisplayName', ['Class ' num2str(j-1) ' Incorrect Classification']);
 hold on
 end
 end
 end
 % Plot decision boundary
 axis manual
 switch type
 case 'L'
 for i = 1:length(x1grid)
 for j = 1:length(x2grid)
 dscScrGrid(j,i) = theta.'*[1; x1grid(i); x2grid(j)];
 end
 end
 case 'Q'
 for i = 1:length(x1grid)
 for j = 1:length(x2grid)
 dscScrGrid(j,i) = theta.'*[1; x1grid(i); x2grid(j); x1grid(i)^2; 
x1grid(i)*x2grid(j); x2grid(j)^2];
 end
 end
 case 'ERM'
 dscScrGrid = log(mvnpdf([x1gridMatrix(:).';x2gridMatrix(:).']', mu(:,3)', sigma(:,:,3))')-log(.5*mvnpdf([x1gridMatrix(:).';x2gridMatrix(:).']', mu(:,1)', sigma(:,:,1))' + .5*mvnpdf([x1gridMatrix(:).';x2gridMatrix(:).']', mu(:,2)', sigma(:,:,2))')-theta;
dscScrGrid = reshape(dscScrGrid,length(x1grid),length(x2grid));
 
 end
 dscScrGrid = reshape(dscScrGrid,length(x1grid),length(x2grid));
 contour(x1grid,x2grid,dscScrGrid,[0 0],'k','DisplayName','Descision Boundary')
 % Clean up graph
 title(graphTitle)
 legend
 xlabel('x1')
 ylabel('x2')
 hold off
end

