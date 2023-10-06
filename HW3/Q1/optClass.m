function [minPFE,decisions]=optClass(lossMatrix,x,mu,Sigma,p,labels,Lx)
% Determine optimal probability of error
symbols='ox+*v';
numLabels=length(Lx);
N=length(x);
for ind = 1:numLabels
 pxgivenl(ind,:) =...
 evalGaussian(x,mu.(Lx{ind}),Sigma.(Lx{ind})); % Evaluate p(x|L=l)
end
px = p*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(p',1,N)./repmat(px,numLabels,1); %P(L=l|x)
% Expected Risk for each label (rows) for each sample (columns)
expectedRisks = lossMatrix*classPosteriors;
% Minimum expected risk decision with 0-1 loss is the same as MAP
[~,decisions] = min(expectedRisks,[],1);
decisions=decisions-1; %Adjust to account for L0 label
fDecision_ind=(decisions~=labels);%Incorrect classificiation vector
minPFE=sum(fDecision_ind)/N;
%Plot Decisions with Incorrect Results
figure;
for ind=1:numLabels
 class_ind=decisions==ind-1;
 plot3(x(1,class_ind & ~fDecision_ind),...
 x(2,class_ind & ~fDecision_ind),...
 x(3,class_ind & ~fDecision_ind),...
 symbols(ind),'Color',[0.39 0.83 0.07],'DisplayName',...
 ['Class ' num2str(ind) ' Correct Classification']);
 hold on;
 plot3(x(1,class_ind & fDecision_ind),...
 x(2,class_ind & fDecision_ind),...
 x(3,class_ind & fDecision_ind),...
 ['r' symbols(ind)],'DisplayName',...
 ['Class ' num2str(ind) ' Incorrect Classification']);
 hold on;
end
xlabel('x1');
ylabel('x2');
grid on;
title('X Vector with Incorrect Classifications');
legend 'show';
if 0
 %Plot Decisions with Incorrect Decisions
 figure;
 for ind2=1:numLabels
 subplot(3,2,ind2);
 for ind=1:numLabels
 class_ind=decisions==ind-1;
 plot3(x(1,class_ind),x(2,class_ind),x(3,class_ind),...
 '.','DisplayName',['Class ' num2str(ind)]);
 hold on;
 end
 plot3(x(1,fDecision_ind & labels==ind2),...
 x(2,fDecision_ind & labels==ind2),...
 x(3,fDecision_ind & labels==ind2),...
 'kx','DisplayName','Incorrectly Classified','LineWidth',2);
 ylabel('x2');
 grid on;
 title(['X Vector with Incorrect Decisions for Class '
num2str(ind2)]);
 if ind2==1
 legend 'show';
 elseif ind2==4
 xlabel('x1');
 end
 end
end
end
